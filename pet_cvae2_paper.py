"""
pet_cvae2_paper.py
==================
Conditional VAE, rebuilt. Reported as 'cvae2' so it sits alongside your
original cVAE in the results table rather than replacing it -- the pair makes
a clean ablation.

WHY THE ORIGINAL SCORED 0.291 (below the trivial baseline)
Your comparison figure shows the diagnosis clearly: the predictions are
STRUCTURALLY CORRECT -- tumour in the right place, right shape, sensible decay
-- but the whole image carries a purple cast where ground truth is bright blue.
The error maps are green/magenta, which is channel disagreement, not spatial
error. So the model was not failing to see the anatomy; it was failing to
reproduce absolute colour. Four causes, each addressed here:

 1. POSTERIOR COLLAPSE. With a strong conditional decoder the KL term drives
    the latent to the prior and the decoder learns the dataset mean, which is a
    washed-out average frame. Fixed with KL warm-up (beta ramps from 0) plus
    free bits, which reserves a minimum information budget per latent dimension
    so the KL cannot crush it to zero.

 2. STOCHASTIC INFERENCE. The original sampled z ~ N(mu, sigma) at test time,
    so every predicted frame carried an independent random offset -- visible as
    a global tint. At inference this version uses the prior mean (z = 0), which
    is the standard deterministic choice for a conditional VAE used as a
    predictor rather than a sampler.

 3. MSE RECONSTRUCTION. Squared error on RGB rewards predicting the conditional
    mean, which is exactly the grey/washed appearance. Replaced with L1 + SSIM.

 4. WEAK CONDITIONING. The decoder had to reconstruct everything from the
    latent. Here the decoder is a U-Net over the context stack with skip
    connections, so z only needs to carry what the context cannot explain.

# --- PATCHED-BY-02_patch_scripts --- (already LOOCV-correct; patcher skips)
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pet_common import (
    ALL_TUMORS, DATA_ROOT, IMG_SUBFOLDER, OUTPUT_DIR, MAX_TIME_MIN,
    N_CONTEXT, IMG_SIZE, BATCH_SIZE, LOOCV_EPOCHS, EPOCHS, MIN_EPOCHS,
    PATIENCE, SEED, N_VAL,
    SinusoidalTimeEmbedding, EarlyStopper,
    autodetect_max_time, setup_device, denorm, ssim_loss_fn, finalize_fold,
    build_loaders, load_test_tumor, make_run_loocv, push_config,
    save_training_curves, save_comparison_grid, save_metrics_plot, env_int,
)
import pet_common as pc

MODEL_NAME = 'cvae2'

LATENT_DIM = 64
FEATURES = 64
TIME_EMB_DIM = 64
LR = 2e-4
BETA_MAX = 0.05          # deliberately small: reconstruction dominates
KL_WARMUP_FRAC = 0.3     # ramp beta over the first 30% of epochs
FREE_BITS = 0.5          # nats per latent dim the KL may not penalise
LAMBDA_SSIM = 0.5


def block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1),
        nn.GroupNorm(min(8, cout), cout), nn.SiLU())


class PosteriorEncoder(nn.Module):
    """q(z | context, target). Only used during training."""

    def __init__(self, ctx_ch, latent=64, f=64):
        super().__init__()
        self.net = nn.Sequential(
            block(ctx_ch + 3, f, 2), block(f, f * 2, 2),
            block(f * 2, f * 4, 2), block(f * 4, f * 4, 2))
        self.mu = nn.Linear(f * 4 * 4 * 4, latent)
        self.logvar = nn.Linear(f * 4 * 4 * 4, latent)

    def forward(self, ctx_flat, tgt):
        h = self.net(torch.cat([ctx_flat, tgt], 1)).flatten(1)
        return self.mu(h), self.logvar(h).clamp(-8, 8)


class ConditionalDecoder(nn.Module):
    """U-Net over the context, modulated by z and the query time."""

    def __init__(self, ctx_ch, latent=64, f=64, emb=64):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(emb)
        self.cond = nn.Sequential(
            nn.Linear(latent + emb, f * 4), nn.SiLU(), nn.Linear(f * 4, f * 4))

        self.e1 = block(ctx_ch, f)
        self.e2 = block(f, f * 2, 2)
        self.e3 = block(f * 2, f * 4, 2)
        self.mid = block(f * 4, f * 4)
        self.d3 = block(f * 8, f * 2)
        self.d2 = block(f * 4, f)
        self.out = nn.Sequential(nn.Conv2d(f * 2, f, 3, 1, 1), nn.SiLU(),
                                 nn.Conv2d(f, 3, 3, 1, 1), nn.Tanh())

    def forward(self, ctx_flat, z, t):
        c = self.cond(torch.cat([z, self.time_emb(t)], -1))[:, :, None, None]
        e1 = self.e1(ctx_flat)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        m = self.mid(e3 + c)
        d3 = F.interpolate(self.d3(torch.cat([m, e3], 1)), scale_factor=2,
                           mode='nearest')
        d2 = F.interpolate(self.d2(torch.cat([d3, e2], 1)), scale_factor=2,
                           mode='nearest')
        return self.out(torch.cat([d2, e1], 1))


class CVAE2(nn.Module):
    def __init__(self, ctx_ch, latent=64, f=64, emb=64):
        super().__init__()
        self.latent = latent
        self.enc = PosteriorEncoder(ctx_ch, latent, f)
        self.dec = ConditionalDecoder(ctx_ch, latent, f, emb)

    def forward(self, ctx_flat, t, tgt=None):
        if tgt is None:
            # Inference: prior mean. Sampling here is what produced the global
            # colour tint in the original model.
            z = torch.zeros(ctx_flat.size(0), self.latent,
                            device=ctx_flat.device)
            return self.dec(ctx_flat, z, t), None, None
        mu, logvar = self.enc(ctx_flat, tgt)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.dec(ctx_flat, z, t), mu, logvar


def kl_free_bits(mu, logvar, free_bits=FREE_BITS):
    """Per-dimension KL, floored at free_bits nats before summing."""
    kl_d = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return torch.clamp(kl_d, min=free_bits).sum(1).mean()


def train_one_fold(train_tumors, val_tumors, test_tumor, epochs, out_dir,
                   device, fold_label=''):
    push_config(globals())
    os.makedirs(out_dir, exist_ok=True)

    tr_loader, va_loader, n_tr, n_va = build_loaders(train_tumors, val_tumors,
                                                     device)
    if n_tr == 0 or n_va == 0:
        print("  [warn] empty train or val set")
        return None

    model = CVAE2(N_CONTEXT * 3, LATENT_DIM, FEATURES, TIME_EMB_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    l1 = nn.L1Loss()

    print(f"\ncVAE2 params      : "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f"Training pairs    : {n_tr}")
    print(f"Validation pairs  : {n_va}")
    print(f"Epochs            : {epochs} "
          f"(patience={pc.PATIENCE}, min={pc.MIN_EPOCHS})")

    # Early stopping uses RECONSTRUCTION only, evaluated the way the model is
    # actually used at test time (z = prior mean). Including the KL term would
    # let the model win by collapsing the latent, which is the failure mode
    # being fixed.
    stopper = EarlyStopper(patience=pc.PATIENCE, min_epochs=pc.MIN_EPOCHS)
    history = {'recon': [], 'kl': [], 'val_recon': []}
    warm = max(int(epochs * KL_WARMUP_FRAC), 1)

    for ep in range(1, epochs + 1):
        beta = BETA_MAX * min(ep / warm, 1.0)
        model.train()
        r_run = k_run = 0.0
        for ctx, ctx_t, t_norm, tgt in tr_loader:
            ctx = ctx.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            t_norm = t_norm.to(device, non_blocking=True)
            ctx_flat = ctx.view(ctx.size(0), -1, IMG_SIZE, IMG_SIZE)

            pred, mu, logvar = model(ctx_flat, t_norm, tgt)
            recon = l1(pred, tgt) + LAMBDA_SSIM * ssim_loss_fn(denorm(pred),
                                                               denorm(tgt))
            kl = kl_free_bits(mu, logvar)
            loss = recon + beta * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            r_run += recon.item(); k_run += kl.item()

        model.eval()
        v_run = 0.0
        with torch.no_grad():
            for ctx, ctx_t, t_norm, tgt in va_loader:
                ctx = ctx.to(device); tgt = tgt.to(device)
                t_norm = t_norm.to(device)
                ctx_flat = ctx.view(ctx.size(0), -1, IMG_SIZE, IMG_SIZE)
                pred, _, _ = model(ctx_flat, t_norm, None)
                v_run += (l1(pred, tgt).item()
                          + LAMBDA_SSIM * ssim_loss_fn(denorm(pred),
                                                       denorm(tgt)).item())
        v_loss = v_run / max(len(va_loader), 1)

        history['recon'].append(r_run / max(len(tr_loader), 1))
        history['kl'].append(k_run / max(len(tr_loader), 1))
        history['val_recon'].append(v_loss)

        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep:4d}/{epochs} | recon={history['recon'][-1]:.4f} "
                  f"kl={history['kl'][-1]:.2f} beta={beta:.4f} "
                  f"val={v_loss:.4f} patience={stopper.counter}/{pc.PATIENCE}")

        if stopper.step(v_loss, model):
            print(f"Early stop at epoch {ep} (best val {stopper.best:.4f})")
            break

    model = stopper.restore(model)
    torch.save(model.state_dict(), Path(out_dir) / 'cvae2_best.pt')
    save_training_curves(history, Path(out_dir) / 'training_curves.png')

    loaded = load_test_tumor(test_tumor, device)
    if loaded is None:
        return None
    ctx, ctx_t, ctx_times_min, tgt_times_min, tgt_norm, gt_arr = loaded
    ctx_flat = ctx.view(1, -1, IMG_SIZE, IMG_SIZE)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(tgt_times_min)):
            p, _, _ = model(ctx_flat, tgt_norm[i].unsqueeze(0), None)
            preds.append(denorm(p[0]).cpu().permute(1, 2, 0).numpy())
    pred_arr = np.asarray(preds, dtype=np.float32)

    ssim_s, psnr_s, mae_s = finalize_fold(pred_arr, gt_arr, tgt_times_min,
                                          out_dir, MODEL_NAME)
    save_comparison_grid(pred_arr, gt_arr, tgt_times_min, test_tumor,
                         Path(out_dir) / 'comparison.png')
    save_metrics_plot(ssim_s, psnr_s, mae_s, tgt_times_min,
                      Path(out_dir) / 'metrics.png')
    return ssim_s, psnr_s, mae_s


run_loocv = make_run_loocv(MODEL_NAME, train_one_fold, globals())


def main():
    global MAX_TIME_MIN
    device = setup_device()
    if MAX_TIME_MIN == 0:
        MAX_TIME_MIN = autodetect_max_time(ALL_TUMORS, DATA_ROOT, IMG_SUBFOLDER)
    run_loocv(device)


if __name__ == '__main__':
    main()
