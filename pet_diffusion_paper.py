"""
pet_diffusion_paper.py   (REBUILT)
==================================
Conditional diffusion model for PET frame prediction.

WHY THE FIRST VERSION FAILED (SSIM 0.355, below the trivial baseline, SD 0.161)
That spread -- 0.211 to 0.723 across folds -- is the signature of a diffusion
model that has not converged, not of a method that does not work. Reporting it
as-is would invite an immediate reviewer objection, because published
conditional diffusion results at this resolution are competitive with GANs.
Four changes, in rough order of impact:

 1. RESIDUAL PARAMETERISATION. The old model generated the whole frame from
    pure noise. Here it generates the RESIDUAL from the last context frame,
    which is a far smaller and more concentrated target. With ~900 training
    pairs the difference between learning "a PET image" and learning "how a PET
    image changes" is enormous. The last frame is added back at the end.

 2. V-PREDICTION instead of epsilon-prediction (Salimans & Ho). Epsilon-
    prediction is poorly conditioned at low noise levels, which is precisely
    where a residual target lives. V-prediction is stable across the whole
    schedule and is the standard choice for few-step sampling.

 3. EMA WEIGHTS. Diffusion training is famously noisy; the raw weights at any
    given step are a poor model. An exponential moving average (decay 0.999) is
    standard practice and is what gets evaluated and checkpointed here.

 4. MORE STEPS AND A REAL EPOCH BUDGET. DDIM raised from 50 to 100 steps, and
    PET_DIFFUSION_EPOCH_MULT (default 3) multiplies the profile's epoch budget
    for this model only, since diffusion needs several times what a regression
    model does. Report both the sampler and the budget in the paper.

# --- PATCHED-BY-02_patch_scripts --- (already LOOCV-correct; patcher skips)
"""

import copy
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
    autodetect_max_time, setup_device, denorm, finalize_fold,
    build_loaders, load_test_tumor, make_run_loocv, push_config,
    save_training_curves, save_comparison_grid, save_metrics_plot, env_int,
)
import pet_common as pc

MODEL_NAME = 'diffusion'

TIME_EMB_DIM = 64
BASE_FEATURES = 64
LR = 2e-4
N_TIMESTEPS = 1000
DDIM_STEPS = env_int('PET_DDIM_STEPS', 100)
EPOCH_MULT = env_int('PET_DIFFUSION_EPOCH_MULT', 3)
EMA_DECAY = 0.999


def cosine_beta_schedule(n, s=0.008):
    steps = torch.arange(n + 1, dtype=torch.float32) / n
    ac = torch.cos((steps + s) / (1 + s) * np.pi * 0.5) ** 2
    ac = ac / ac[0]
    return (1 - (ac[1:] / ac[:-1])).clamp(1e-8, 0.999)


class Schedule:
    def __init__(self, n_steps, device):
        self.n = n_steps
        self.betas = cosine_beta_schedule(n_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.acp = torch.cumprod(self.alphas, 0)
        self.sa = self.acp.sqrt()
        self.sb = (1 - self.acp).sqrt()

    def q_sample(self, x0, s, noise):
        return self.sa[s][:, None, None, None] * x0 + \
               self.sb[s][:, None, None, None] * noise

    def to_v(self, x0, noise, s):
        """v = sqrt(acp)*noise - sqrt(1-acp)*x0"""
        return self.sa[s][:, None, None, None] * noise - \
               self.sb[s][:, None, None, None] * x0

    def from_v(self, x_t, v, s):
        """Recover (x0, noise) from a predicted v at step s."""
        a = self.sa[s][:, None, None, None]
        b = self.sb[s][:, None, None, None]
        x0 = a * x_t - b * v
        noise = b * x_t + a * v
        return x0, noise


class Block(nn.Module):
    def __init__(self, cin, cout, emb_dim):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.emb = nn.Linear(emb_dim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class ConditionalDenoiser(nn.Module):
    """U-Net v-predictor conditioned on context, diffusion step and scan time."""

    def __init__(self, ctx_ch, features=64, emb_dim=64):
        super().__init__()
        f = features
        self.step_emb = SinusoidalTimeEmbedding(emb_dim)
        self.time_emb = SinusoidalTimeEmbedding(emb_dim)
        self.emb_mix = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2), nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim))
        self.in_conv = nn.Conv2d(3 + ctx_ch, f, 3, 1, 1)
        self.d1 = Block(f, f, emb_dim)
        self.d2 = Block(f, f * 2, emb_dim)
        self.d3 = Block(f * 2, f * 4, emb_dim)
        self.mid = Block(f * 4, f * 4, emb_dim)
        self.u3 = Block(f * 8, f * 2, emb_dim)
        self.u2 = Block(f * 4, f, emb_dim)
        self.u1 = Block(f * 2, f, emb_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, f), nn.SiLU(),
                                 nn.Conv2d(f, 3, 3, 1, 1))
        self.pool = nn.AvgPool2d(2)

    def forward(self, x_noisy, ctx_flat, s, t):
        emb = self.emb_mix(torch.cat([self.step_emb(s), self.time_emb(t)], -1))
        h = self.in_conv(torch.cat([x_noisy, ctx_flat], 1))
        h1 = self.d1(h, emb)
        h2 = self.d2(self.pool(h1), emb)
        h3 = self.d3(self.pool(h2), emb)
        m = self.mid(h3, emb)
        u3 = F.interpolate(self.u3(torch.cat([m, h3], 1), emb),
                           scale_factor=2, mode='nearest')
        u2 = F.interpolate(self.u2(torch.cat([u3, h2], 1), emb),
                           scale_factor=2, mode='nearest')
        return self.out(self.u1(torch.cat([u2, h1], 1), emb))


@torch.no_grad()
def ddim_sample(model, ctx_flat, t_norm, sched, n_steps, device):
    """Deterministic DDIM (eta=0) in residual space."""
    b = ctx_flat.size(0)
    x = torch.randn(b, 3, IMG_SIZE, IMG_SIZE, device=device)
    idx = torch.linspace(sched.n - 1, 0, n_steps).long().to(device)
    for i in range(n_steps):
        s = idx[i].repeat(b)
        v = model(x, ctx_flat, s.float(), t_norm)
        x0, eps = sched.from_v(x, v, s)
        x0 = x0.clamp(-2, 2)          # residuals, not images: wider range
        if i < n_steps - 1:
            ap = sched.acp[idx[i + 1]]
            x = ap.sqrt() * x0 + (1 - ap).sqrt() * eps
        else:
            x = x0
    return x


def train_one_fold(train_tumors, val_tumors, test_tumor, epochs, out_dir,
                   device, fold_label=''):
    push_config(globals())
    os.makedirs(out_dir, exist_ok=True)
    epochs = epochs * EPOCH_MULT

    tr_loader, va_loader, n_tr, n_va = build_loaders(train_tumors, val_tumors,
                                                     device)
    if n_tr == 0 or n_va == 0:
        print("  [warn] empty train or val set")
        return None

    sched = Schedule(N_TIMESTEPS, device)
    model = ConditionalDenoiser(N_CONTEXT * 3, BASE_FEATURES,
                                TIME_EMB_DIM).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    print(f"\nDiffusion params  : "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f"Training pairs    : {n_tr}")
    print(f"Validation pairs  : {n_va}")
    print(f"Parameterisation  : v-prediction on the residual from last context")
    print(f"Train / DDIM steps: {N_TIMESTEPS} / {DDIM_STEPS}")
    print(f"Epochs            : {epochs}  (profile x{EPOCH_MULT})")

    stopper = EarlyStopper(patience=pc.PATIENCE * EPOCH_MULT,
                           min_epochs=pc.MIN_EPOCHS * EPOCH_MULT)
    history = {'train_v': [], 'val_v': []}

    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for ctx, ctx_t, t_norm, tgt in tr_loader:
            ctx = ctx.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            t_norm = t_norm.to(device, non_blocking=True)
            ctx_flat = ctx.view(ctx.size(0), -1, IMG_SIZE, IMG_SIZE)
            last = ctx[:, -1]
            resid = tgt - last                       # the diffusion target

            s = torch.randint(0, N_TIMESTEPS, (tgt.size(0),), device=device)
            noise = torch.randn_like(resid)
            x_noisy = sched.q_sample(resid, s, noise)
            v_true = sched.to_v(resid, noise, s)

            loss = F.mse_loss(model(x_noisy, ctx_flat, s.float(), t_norm),
                              v_true)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(EMA_DECAY).add_(pm, alpha=1 - EMA_DECAY)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            run += loss.item()

        # Fixed-seed validation so early stopping tracks the model, not the
        # sampling noise.
        ema.eval()
        vgen = torch.Generator(device='cpu').manual_seed(SEED)
        vrun = 0.0
        with torch.no_grad():
            for ctx, ctx_t, t_norm, tgt in va_loader:
                ctx = ctx.to(device); tgt = tgt.to(device)
                t_norm = t_norm.to(device)
                ctx_flat = ctx.view(ctx.size(0), -1, IMG_SIZE, IMG_SIZE)
                resid = tgt - ctx[:, -1]
                s = torch.randint(0, N_TIMESTEPS, (tgt.size(0),),
                                  generator=vgen).to(device)
                noise = torch.randn(resid.shape, generator=vgen).to(device)
                x_noisy = sched.q_sample(resid, s, noise)
                v_true = sched.to_v(resid, noise, s)
                vrun += F.mse_loss(ema(x_noisy, ctx_flat, s.float(), t_norm),
                                   v_true).item()
        v_loss = vrun / max(len(va_loader), 1)

        history['train_v'].append(run / max(len(tr_loader), 1))
        history['val_v'].append(v_loss)

        if ep % 25 == 0 or ep == 1:
            print(f"Ep {ep:4d}/{epochs} | v={history['train_v'][-1]:.4f} "
                  f"val={v_loss:.4f} patience={stopper.counter}")

        if stopper.step(v_loss, ema):
            print(f"Early stop at epoch {ep} (best val {stopper.best:.4f})")
            break

    ema = stopper.restore(ema)
    torch.save(ema.state_dict(), Path(out_dir) / 'denoiser_ema.pt')
    save_training_curves(history, Path(out_dir) / 'training_curves.png')

    loaded = load_test_tumor(test_tumor, device)
    if loaded is None:
        return None
    ctx, ctx_t, ctx_times_min, tgt_times_min, tgt_norm, gt_arr = loaded
    ctx_flat = ctx.view(1, -1, IMG_SIZE, IMG_SIZE)
    last = ctx[:, -1]

    ema.eval()
    torch.manual_seed(SEED)
    preds = []
    for i in range(len(tgt_times_min)):
        r = ddim_sample(ema, ctx_flat, tgt_norm[i].unsqueeze(0), sched,
                        DDIM_STEPS, device)
        img = (last + r).clamp(-1, 1)
        preds.append(denorm(img[0]).cpu().permute(1, 2, 0).numpy())
        if (i + 1) % 25 == 0:
            print(f"  sampled {i+1}/{len(tgt_times_min)} frames")
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
