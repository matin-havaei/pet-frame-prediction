"""
pet_cgan_paper.py
=================
Conditional GAN for PET frame prediction, with leave-one-out cross-validation.

Architecture is unchanged from the user's original script: a U-Net generator
with sinusoidal time conditioning injected at the bottleneck, a PatchGAN
discriminator, LSGAN objective, and an L1 + SSIM reconstruction term. What is
added here is the train_one_fold / run_loocv pair the LOOCV harness requires,
plus per-fold checkpointing and figures.

Relationship to pet_pix2pix_paper.py: both are conditional GANs. This one adds
(a) explicit sinusoidal encoding of the target acquisition time, so the
generator is conditioned on WHEN the frame occurs rather than only on the
context stack, and (b) an SSIM term alongside L1. State that distinction
explicitly in the paper -- a reviewer comparing the two tables will otherwise
ask why the same model appears twice.

# --- PATCHED-BY-02_patch_scripts --- (already LOOCV-correct; patcher skips)
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pet_common import (
    ALL_TUMORS, DATA_ROOT, IMG_SUBFOLDER, OUTPUT_DIR, MAX_TIME_MIN,
    N_CONTEXT, IMG_SIZE, BATCH_SIZE, LOOCV_EPOCHS, EPOCHS, MIN_EPOCHS,
    PATIENCE, SEED, N_VAL,
    SinusoidalTimeEmbedding, EarlyStopper,
    autodetect_max_time, setup_device, denorm, compute_metrics, ssim_loss_fn,
    build_loaders, load_test_tumor, make_run_loocv, push_config,
    save_training_curves, save_comparison_grid, save_metrics_plot,
    env_int,
)
import pet_common as pc

MODEL_NAME = 'cgan'

# ── model hyperparameters (unchanged from the original) ──────────────────────
TIME_EMB_DIM = 64
GEN_FEATURES = 64
DIS_FEATURES = 64
LR_G = 2e-4
LR_D = 1e-4
LAMBDA_L1 = 100.0
LAMBDA_SSIM = 0.3
N_CRITIC = 1


# ═════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════

class UNetGeneratorFixed(nn.Module):
    def __init__(self, in_ch, out_ch=3, features=64, time_emb_dim=64):
        super().__init__()
        f = features
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, f * 8)

        self.e1 = nn.Sequential(nn.Conv2d(in_ch, f, 4, 2, 1),
                                nn.LeakyReLU(0.2, True))
        self.e2 = nn.Sequential(nn.Conv2d(f, f * 2, 4, 2, 1, bias=False),
                                nn.InstanceNorm2d(f * 2), nn.LeakyReLU(0.2, True))
        self.e3 = nn.Sequential(nn.Conv2d(f * 2, f * 4, 4, 2, 1, bias=False),
                                nn.InstanceNorm2d(f * 4), nn.LeakyReLU(0.2, True))
        self.e4 = nn.Sequential(nn.Conv2d(f * 4, f * 8, 4, 2, 1, bias=False),
                                nn.InstanceNorm2d(f * 8), nn.LeakyReLU(0.2, True))
        self.e5 = nn.Sequential(nn.Conv2d(f * 8, f * 8, 4, 2, 1),
                                nn.LeakyReLU(0.2, True))

        self.d1 = nn.Sequential(nn.ConvTranspose2d(f * 8, f * 8, 4, 2, 1, bias=False),
                                nn.InstanceNorm2d(f * 8), nn.ReLU(True), nn.Dropout(0.5))
        self.d2 = nn.Sequential(nn.ConvTranspose2d(f * 16, f * 8, 4, 2, 1, bias=False),
                                nn.InstanceNorm2d(f * 8), nn.ReLU(True), nn.Dropout(0.5))
        self.d3 = nn.Sequential(nn.ConvTranspose2d(f * 12, f * 4, 4, 2, 1, bias=False),
                                nn.InstanceNorm2d(f * 4), nn.ReLU(True), nn.Dropout(0.5))
        self.d4 = nn.Sequential(nn.ConvTranspose2d(f * 6, f * 2, 4, 2, 1, bias=False),
                                nn.InstanceNorm2d(f * 2), nn.ReLU(True))
        self.d5 = nn.Sequential(nn.ConvTranspose2d(f * 3, f, 4, 2, 1, bias=False),
                                nn.InstanceNorm2d(f), nn.ReLU(True))
        self.out_layer = nn.Sequential(nn.Conv2d(f, out_ch, 3, 1, 1), nn.Tanh())

    def forward(self, ctx, t):
        t_bias = self.time_proj(self.time_emb(t))
        e1 = self.e1(ctx); e2 = self.e2(e1)
        e3 = self.e3(e2); e4 = self.e4(e3)
        b = self.e5(e4) + t_bias[:, :, None, None]
        d1 = torch.cat([self.d1(b), e4], dim=1)
        d2 = torch.cat([self.d2(d1), e3], dim=1)
        d3 = torch.cat([self.d3(d2), e2], dim=1)
        d4 = torch.cat([self.d4(d3), e1], dim=1)
        return self.out_layer(self.d5(d4))


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_ch, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, features, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features * 4, features * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(features * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(features * 8, 1, 4, 1, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ctx, frame):
        return self.net(torch.cat([ctx, frame], dim=1))


class GANLoss(nn.Module):
    """LSGAN: least-squares objective, more stable than BCE at this scale."""

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_one_fold(train_tumors, val_tumors, test_tumor, epochs, out_dir,
                   device, fold_label=''):
    push_config(globals())
    os.makedirs(out_dir, exist_ok=True)

    tr_loader, va_loader, n_tr, n_va = build_loaders(train_tumors, val_tumors,
                                                     device)
    if n_tr == 0 or n_va == 0:
        print("  [warn] empty train or val set")
        return None

    in_ch = N_CONTEXT * 3
    G = UNetGeneratorFixed(in_ch, 3, GEN_FEATURES, TIME_EMB_DIM).to(device)
    D = PatchGANDiscriminator(in_ch + 3, DIS_FEATURES).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    gan_loss = GANLoss()
    l1 = nn.L1Loss()

    n_params = sum(p.numel() for p in G.parameters())
    print(f"\ncGAN generator params : {n_params:,}")
    print(f"Training pairs        : {n_tr}")
    print(f"Validation pairs      : {n_va}")
    print(f"Epochs                : {epochs} "
          f"(patience={pc.PATIENCE}, min={pc.MIN_EPOCHS})")

    # Early stopping is driven by the RECONSTRUCTION loss on validation, not by
    # the adversarial loss. GAN losses are not comparable across epochs -- they
    # measure a moving target -- so using them to select a checkpoint is
    # meaningless. L1+SSIM on held-out tumours is a real quantity.
    stopper = EarlyStopper(patience=pc.PATIENCE, min_epochs=pc.MIN_EPOCHS)
    history = {'G_total': [], 'D': [], 'val_recon': []}

    for ep in range(1, epochs + 1):
        G.train(); D.train()
        g_run = d_run = 0.0
        for step, (ctx, ctx_t, t_norm, tgt) in enumerate(tr_loader):
            ctx = ctx.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            t_norm = t_norm.to(device, non_blocking=True)
            ctx_flat = ctx.view(ctx.size(0), -1, IMG_SIZE, IMG_SIZE)

            fake = G(ctx_flat, t_norm)

            # --- discriminator ---
            opt_D.zero_grad(set_to_none=True)
            d_real = D(ctx_flat, tgt)
            d_fake = D(ctx_flat, fake.detach())
            d_loss = 0.5 * (gan_loss(d_real, True) + gan_loss(d_fake, False))
            d_loss.backward()
            opt_D.step()

            # --- generator ---
            if step % N_CRITIC == 0:
                opt_G.zero_grad(set_to_none=True)
                adv = gan_loss(D(ctx_flat, fake), True)
                rec_l1 = l1(fake, tgt)
                rec_ssim = ssim_loss_fn(denorm(fake), denorm(tgt))
                g_loss = adv + LAMBDA_L1 * rec_l1 + LAMBDA_SSIM * rec_ssim
                g_loss.backward()
                opt_G.step()
                g_run += g_loss.item()
            d_run += d_loss.item()

        # --- validation (reconstruction only) ---
        G.eval()
        v_run = 0.0
        with torch.no_grad():
            for ctx, ctx_t, t_norm, tgt in va_loader:
                ctx = ctx.to(device); tgt = tgt.to(device)
                t_norm = t_norm.to(device)
                ctx_flat = ctx.view(ctx.size(0), -1, IMG_SIZE, IMG_SIZE)
                pred = G(ctx_flat, t_norm)
                v_run += (l1(pred, tgt).item()
                          + LAMBDA_SSIM * ssim_loss_fn(denorm(pred),
                                                       denorm(tgt)).item())
        v_loss = v_run / max(len(va_loader), 1)

        history['G_total'].append(g_run / max(len(tr_loader), 1))
        history['D'].append(d_run / max(len(tr_loader), 1))
        history['val_recon'].append(v_loss)

        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep:4d}/{epochs} | G={history['G_total'][-1]:.4f} "
                  f"D={history['D'][-1]:.4f} val={v_loss:.4f} "
                  f"patience={stopper.counter}/{pc.PATIENCE}")

        if stopper.step(v_loss, G):
            print(f"Early stop at epoch {ep} (best val {stopper.best:.4f})")
            break

    G = stopper.restore(G)
    torch.save(G.state_dict(), Path(out_dir) / 'generator_best.pt')
    save_training_curves(history, Path(out_dir) / 'training_curves.png')

    # --- inference on the held-out tumour ---
    loaded = load_test_tumor(test_tumor, device)
    if loaded is None:
        print(f"  [warn] test tumour {test_tumor} not loadable")
        return None
    ctx, ctx_t, ctx_times_min, tgt_times_min, tgt_norm, gt_arr = loaded
    ctx_flat = ctx.view(1, -1, IMG_SIZE, IMG_SIZE)

    G.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(tgt_times_min)):
            p = G(ctx_flat, tgt_norm[i].unsqueeze(0))
            preds.append(denorm(p[0]).cpu().permute(1, 2, 0).numpy())
    pred_arr = np.array(preds)

    ssim_s, psnr_s, mae_s = compute_metrics(pred_arr, gt_arr)
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
