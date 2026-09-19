"""
pet_swin_paper.py
=================
Swin-style hierarchical transformer with shifted-window attention, in a U-Net
layout with skip connections.

WHY THIS AND NOT ANOTHER PLAIN ViT
Your existing ViT scored 0.711 -- the weakest of the seven working deep models.
That is the expected result and worth explaining rather than burying: a plain
ViT flattens the image into non-overlapping patches and applies global
attention with no inductive bias toward locality, which needs far more data
than ten tumours provide. It also loses fine spatial detail at the patch
boundaries, which SSIM punishes.

Swin (Liu et al., ICCV 2021) fixes exactly this. Attention is computed inside
local windows, so cost is linear rather than quadratic in the number of tokens,
and the windows SHIFT between consecutive blocks so information still crosses
window boundaries. Combined with a hierarchical encoder-decoder and skip
connections, it retains the locality bias of a CNN while keeping attention.
Swin-UNet variants are now the standard transformer backbone in medical imaging,
so this is the fair transformer comparison; the plain ViT becomes an ablation
showing what the inductive bias is worth.

Conditioning on the query acquisition time is injected as an additive bias on
the bottleneck tokens, matching how the other models here receive it.

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

MODEL_NAME = 'swin'

EMBED_DIM = 96
DEPTHS = (2, 2, 4)
HEADS = (3, 6, 12)
WINDOW = 4
TIME_EMB_DIM = 64
LR = 3e-4
LAMBDA_SSIM = 0.4


def window_partition(x, w):
    """(B,H,W,C) -> (B*nW, w*w, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // w, w, W // w, w, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, C)


def window_reverse(win, w, H, W, B):
    """(B*nW, w*w, C) -> (B,H,W,C)"""
    C = win.shape[-1]
    x = win.view(B, H // w, W // w, w, w, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        # Learned relative position bias within a window -- the component that
        # gives Swin its spatial awareness without absolute embeddings.
        self.bias = nn.Parameter(torch.zeros(heads, window * window,
                                             window * window))
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.bias.unsqueeze(0)
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window, shift):
        super().__init__()
        self.window = window
        self.shift = shift
        self.n1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, window)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, x, H, W):
        B, L, C = x.shape
        h = self.n1(x).view(B, H, W, C)
        if self.shift:
            h = torch.roll(h, (-self.shift, -self.shift), dims=(1, 2))
        win = window_partition(h, self.window)
        win = self.attn(win)
        h = window_reverse(win, self.window, H, W, B)
        if self.shift:
            h = torch.roll(h, (self.shift, self.shift), dims=(1, 2))
        x = x + h.view(B, L, C)
        return x + self.mlp(self.n2(x))


class Stage(nn.Module):
    def __init__(self, dim, depth, heads, window):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, heads, window, 0 if i % 2 == 0 else window // 2)
            for i in range(depth)])

    def forward(self, x, H, W):
        for b in self.blocks:
            x = b(x, H, W)
        return x


class SwinUNet(nn.Module):
    def __init__(self, ctx_ch, dim=96, depths=(2, 2, 4), heads=(3, 6, 12),
                 window=4, emb_dim=64):
        super().__init__()
        self.window = window
        self.time_emb = SinusoidalTimeEmbedding(emb_dim)

        self.patch = nn.Conv2d(ctx_ch, dim, 2, 2)          # 64 -> 32
        self.s1 = Stage(dim, depths[0], heads[0], window)
        self.down1 = nn.Conv2d(dim, dim * 2, 2, 2)         # 32 -> 16
        self.s2 = Stage(dim * 2, depths[1], heads[1], window)
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 2, 2)     # 16 -> 8
        self.s3 = Stage(dim * 4, depths[2], heads[2], window)

        self.time_proj = nn.Linear(emb_dim, dim * 4)

        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, 2)
        self.f2 = nn.Conv2d(dim * 4, dim * 2, 3, 1, 1)
        self.s2d = Stage(dim * 2, depths[1], heads[1], window)
        self.up1 = nn.ConvTranspose2d(dim * 2, dim, 2, 2)
        self.f1 = nn.Conv2d(dim * 2, dim, 3, 1, 1)
        self.s1d = Stage(dim, depths[0], heads[0], window)

        self.out = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, 2, 2), nn.GELU(),
            nn.Conv2d(dim // 2, 3, 3, 1, 1), nn.Tanh())

    @staticmethod
    def _flat(x):
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), H, W

    @staticmethod
    def _unflat(x, H, W):
        B, L, C = x.shape
        return x.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, ctx_flat, t):
        x = self.patch(ctx_flat)
        f, H, W = self._flat(x)
        f = self.s1(f, H, W)
        e1 = self._unflat(f, H, W)

        x = self.down1(e1)
        f, H, W = self._flat(x)
        f = self.s2(f, H, W)
        e2 = self._unflat(f, H, W)

        x = self.down2(e2)
        f, H, W = self._flat(x)
        f = f + self.time_proj(self.time_emb(t)).unsqueeze(1)
        f = self.s3(f, H, W)
        b = self._unflat(f, H, W)

        d2 = self.f2(torch.cat([self.up2(b), e2], 1))
        f, H, W = self._flat(d2)
        d2 = self._unflat(self.s2d(f, H, W), H, W)

        d1 = self.f1(torch.cat([self.up1(d2), e1], 1))
        f, H, W = self._flat(d1)
        d1 = self._unflat(self.s1d(f, H, W), H, W)
        return self.out(d1)


def train_one_fold(train_tumors, val_tumors, test_tumor, epochs, out_dir,
                   device, fold_label=''):
    push_config(globals())
    os.makedirs(out_dir, exist_ok=True)

    tr_loader, va_loader, n_tr, n_va = build_loaders(train_tumors, val_tumors,
                                                     device)
    if n_tr == 0 or n_va == 0:
        print("  [warn] empty train or val set")
        return None

    model = SwinUNet(N_CONTEXT * 3, EMBED_DIM, DEPTHS, HEADS, WINDOW,
                     TIME_EMB_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    l1 = nn.L1Loss()

    print(f"\nSwin-UNet params  : "
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
            ctx_flat = ctx.view(ctx.size(0), -1, IMG_SIZE, IMG_SIZE)

            pred = model(ctx_flat, t_norm)
            loss = l1(pred, tgt) + LAMBDA_SSIM * ssim_loss_fn(denorm(pred),
                                                              denorm(tgt))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()
        sched.step()

        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for ctx, ctx_t, t_norm, tgt in va_loader:
                ctx = ctx.to(device); tgt = tgt.to(device)
                t_norm = t_norm.to(device)
                ctx_flat = ctx.view(ctx.size(0), -1, IMG_SIZE, IMG_SIZE)
                pred = model(ctx_flat, t_norm)
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
    torch.save(model.state_dict(), Path(out_dir) / 'swin_best.pt')
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
            p = model(ctx_flat, tgt_norm[i].unsqueeze(0))
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
