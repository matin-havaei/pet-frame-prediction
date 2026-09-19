"""
pet_simvp_paper.py
==================
SimVP-style spatiotemporal predictor, adapted for continuous-time conditioning.

WHY THIS MODEL IS IN THE PAPER
Your architecture list is drawn almost entirely from image-to-image translation
(U-Net, GAN variants) plus one recurrent model. SimVP (Gao et al., CVPR 2022)
is the standard modern reference point for VIDEO PREDICTION specifically -- a
purely convolutional model that outperformed the RNN family (ConvLSTM,
PredRNN) on the usual benchmarks while being far cheaper to train. Including it
means your comparison covers the video-prediction literature and not only the
image-translation literature, which is the more defensible framing given your
task is next-frame prediction.

ADAPTATION
Vanilla SimVP maps T input frames to T output frames on a fixed grid. Your task
predicts a single frame at an ARBITRARY query time, so the temporal translator
is conditioned on a sinusoidal embedding of the normalised acquisition time and
the decoder emits one frame. This is a deliberate departure from the published
architecture and must be described as such in the paper -- call it
"SimVP-style" rather than SimVP.

Encoder and decoder follow the original: strided convolutions with GroupNorm
and SiLU. The translator uses the Inception-style multi-kernel bottleneck that
gives SimVP its receptive-field range at low parameter cost.

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
    autodetect_max_time, setup_device, denorm, compute_metrics, ssim_loss_fn,
    build_loaders, load_test_tumor, make_run_loocv, push_config,
    save_training_curves, save_comparison_grid, save_metrics_plot, env_int,
)
import pet_common as pc

MODEL_NAME = 'simvp'

HID_S = 64          # spatial hidden channels
HID_T = 256         # temporal translator channels
N_S = 4             # encoder/decoder depth
N_T = 6             # translator depth
TIME_EMB_DIM = 64
LR = 1e-3
LAMBDA_SSIM = 0.3


def conv_gn(cin, cout, stride=1, transpose=False):
    if transpose:
        conv = nn.ConvTranspose2d(cin, cout, 3, stride, 1,
                                  output_padding=stride - 1)
    else:
        conv = nn.Conv2d(cin, cout, 3, stride, 1)
    return nn.Sequential(conv, nn.GroupNorm(min(8, cout), cout), nn.SiLU())


class Encoder(nn.Module):
    """Downsamples on every other layer, as in the reference implementation."""

    def __init__(self, cin, hid, n_layers):
        super().__init__()
        strides = [2 if i % 2 == 0 else 1 for i in range(n_layers)]
        layers = [conv_gn(cin, hid, strides[0])]
        for i in range(1, n_layers):
            layers.append(conv_gn(hid, hid, strides[i]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        skip = None
        for i, l in enumerate(self.layers):
            x = l(x)
            if i == 0:
                skip = x
        return x, skip


class Decoder(nn.Module):
    def __init__(self, hid, cout, n_layers):
        super().__init__()
        strides = [2 if i % 2 == 0 else 1 for i in range(n_layers)][::-1]
        layers = []
        for i in range(n_layers - 1):
            layers.append(conv_gn(hid, hid, strides[i], transpose=True))
        self.layers = nn.ModuleList(layers)
        self.final = conv_gn(hid * 2, hid, strides[-1], transpose=True)
        self.out = nn.Sequential(nn.Conv2d(hid, cout, 3, 1, 1), nn.Tanh())

    def forward(self, x, skip):
        for l in self.layers:
            x = l(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
        x = self.final(torch.cat([x, skip], dim=1))
        return self.out(x)


class InceptionBlock(nn.Module):
    """Parallel kernels of different size, concatenated -- SimVP's translator."""

    def __init__(self, cin, cout, kernels=(3, 5, 7, 11)):
        super().__init__()
        self.reduce = nn.Conv2d(cin, cout // 2, 1)
        per = cout // len(kernels)
        self.branches = nn.ModuleList([
            nn.Conv2d(cout // 2, per, k, 1, k // 2) for k in kernels
        ])
        self.norm = nn.GroupNorm(8, per * len(kernels))
        self.proj = nn.Conv2d(per * len(kernels), cout, 1)

    def forward(self, x):
        h = self.reduce(x)
        h = torch.cat([b(h) for b in self.branches], dim=1)
        return F.silu(self.proj(F.silu(self.norm(h))))


class Translator(nn.Module):
    """Mixes the flattened spatial features, conditioned on query time."""

    def __init__(self, cin, hid, n_layers, emb_dim):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(emb_dim)
        self.time_proj = nn.Linear(emb_dim, hid)
        self.enc = nn.ModuleList(
            [InceptionBlock(cin, hid)] +
            [InceptionBlock(hid, hid) for _ in range(n_layers // 2 - 1)])
        self.dec = nn.ModuleList(
            [InceptionBlock(hid, hid) for _ in range(n_layers // 2 - 1)] +
            [InceptionBlock(hid, cin)])

    def forward(self, x, t):
        bias = self.time_proj(self.time_emb(t))[:, :, None, None]
        skips = []
        for i, l in enumerate(self.enc):
            x = l(x)
            if i == 0:
                x = x + bias
            skips.append(x)
        for i, l in enumerate(self.dec):
            if i > 0 and skips:
                x = x + skips[-i]
            x = l(x)
        return x


class SimVP(nn.Module):
    def __init__(self, n_context, hid_s=64, hid_t=256, n_s=4, n_t=6,
                 emb_dim=64):
        super().__init__()
        self.n_context = n_context
        self.enc = Encoder(3, hid_s, n_s)
        self.trans = Translator(hid_s * n_context, hid_t, n_t, emb_dim)
        self.dec = Decoder(hid_s, 3, n_s)

    def forward(self, ctx, t):
        # ctx: (B, T, C, H, W) -> encode each frame independently
        b, T, c, h, w = ctx.shape
        flat = ctx.reshape(b * T, c, h, w)
        feat, skip = self.enc(flat)
        _, fc, fh, fw = feat.shape
        # Stack the per-frame features along channels for the translator.
        z = feat.reshape(b, T * fc, fh, fw)
        z = self.trans(z, t)
        # Take the slice corresponding to the last context frame as the seed.
        z = z.reshape(b, T, fc, fh, fw)[:, -1]
        skip = skip.reshape(b, T, *skip.shape[1:])[:, -1]
        return self.dec(z, skip)


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

    model = SimVP(N_CONTEXT, HID_S, HID_T, N_S, N_T, TIME_EMB_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=max(epochs * len(tr_loader), 1),
        pct_start=0.1)
    l1 = nn.L1Loss()

    print(f"\nSimVP params      : "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f"Training pairs    : {n_tr}")
    print(f"Validation pairs  : {n_va}")
    print(f"Epochs            : {epochs} "
          f"(patience={pc.PATIENCE}, min={pc.MIN_EPOCHS})")

    stopper = EarlyStopper(patience=pc.PATIENCE, min_epochs=pc.MIN_EPOCHS)
    history = {'train': [], 'val': []}

    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for ctx, ctx_t, t_norm, tgt in tr_loader:
            ctx = ctx.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            t_norm = t_norm.to(device, non_blocking=True)

            pred = model(ctx, t_norm)
            loss = l1(pred, tgt) + LAMBDA_SSIM * ssim_loss_fn(denorm(pred),
                                                              denorm(tgt))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            try:
                sched.step()
            except ValueError:
                pass          # OneCycle exhausted after an early-stop restart
            run += loss.item()

        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for ctx, ctx_t, t_norm, tgt in va_loader:
                ctx = ctx.to(device); tgt = tgt.to(device)
                t_norm = t_norm.to(device)
                pred = model(ctx, t_norm)
                vrun += (l1(pred, tgt).item()
                         + LAMBDA_SSIM * ssim_loss_fn(denorm(pred),
                                                      denorm(tgt)).item())
        v_loss = vrun / max(len(va_loader), 1)

        history['train'].append(run / max(len(tr_loader), 1))
        history['val'].append(v_loss)

        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep:4d}/{epochs} | loss={history['train'][-1]:.4f} "
                  f"val={v_loss:.4f} patience={stopper.counter}/{pc.PATIENCE}")

        if stopper.step(v_loss, model):
            print(f"Early stop at epoch {ep} (best val {stopper.best:.4f})")
            break

    model = stopper.restore(model)
    torch.save(model.state_dict(), Path(out_dir) / 'simvp_best.pt')
    save_training_curves(history, Path(out_dir) / 'training_curves.png')

    loaded = load_test_tumor(test_tumor, device)
    if loaded is None:
        return None
    ctx, ctx_t, ctx_times_min, tgt_times_min, tgt_norm, gt_arr = loaded

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(tgt_times_min)):
            p = model(ctx, tgt_norm[i].unsqueeze(0))
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
