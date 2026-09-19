"""
pet_attention_unet_paper.py
===========================
Attention U-Net Baseline for PET Tumor Simulation Completion
— Paper-ready version — matches cGAN style exactly —

Architecture  : U-Net + Additive Attention Gates (Oktay et al. 2018)
                Attention gates suppress irrelevant activations in skip connections.
Time encoding : Sinusoidal embedding injected at bottleneck
Loss          : L1 + SSIM
Split         : 7 train / 2 val / 1 test

Reference: Oktay O, et al. Attention U-Net: Learning Where to Look for the Pancreas.
           MIDL 2018. arXiv:1804.03999

Setup:
  pip install torch torchvision numpy pillow matplotlib tqdm scikit-image scipy

Run:
  python pet_attention_unet_paper.py
  python pet_attention_unet_paper.py --loocv
"""

import os
import re
import csv
import math
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT     = r"C:\PET_DATA\IMAGES"
OUTPUT_DIR    = r"C:\PET_DATA\RESULTS"
IMG_SUBFOLDER = r"cleaned\resized"

ALL_TUMORS = [
    't1-r=0.1', 'T1-R=0.2', 'T1-R=0.3', 'T1-R=0.4',
    'T2-R=0.1', 'T2-R=0.2', 't2-r=0.3', 'T2-R=0.4',
    'T3-R=0.2', 'T3-R=0.1',
]
TEST_TUMOR   = 'T3-R=0.1'
VAL_TUMORS   = ['T3-R=0.2', 'T2-R=0.4']
TRAIN_TUMORS = [t for t in ALL_TUMORS
                if t != TEST_TUMOR and t not in VAL_TUMORS]

N_CONTEXT    = 10
IMG_SIZE     = 64
MAX_TIME_MIN = 0

# Model
TIME_EMB_DIM = 64
FEATURES     = 64

# Training
BATCH_SIZE   = 16
EPOCHS       = 1000
LR           = 2e-4
WEIGHT_DECAY = 1e-4
LAMBDA_L1    = 1.0
LAMBDA_SSIM  = 0.5
PATIENCE     = 150
MIN_EPOCHS   = 400

# LOOCV
LOOCV_EPOCHS = 800

SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def extract_timepoint(filename):
    nums = re.findall(r'\d+', Path(filename).stem)
    return int(nums[0]) if nums else None


def get_files(tumor_folder, subfolder):
    for sub in [subfolder, 'resized', 'cleaned', '']:
        d = Path(tumor_folder) / sub if sub else Path(tumor_folder)
        if d.exists():
            files = sorted(
                [str(f) for f in d.iterdir()
                 if f.suffix.lower() in IMG_EXTS
                 and not f.name.startswith('_')
                 and extract_timepoint(f.name) is not None],
                key=lambda p: extract_timepoint(Path(p).name)
            )
            if files:
                return files
    return []


def autodetect_max_time(tumor_names, data_root, subfolder):
    max_t = 0
    for tumor in tumor_names:
        files = get_files(Path(data_root) / tumor, subfolder)
        for f in files:
            t = extract_timepoint(Path(f).name)
            if t and t > max_t:
                max_t = t
    assert max_t > 0, "Could not detect any timepoints — check folder/filenames"
    print(f"Auto-detected MAX_TIME_MIN: {max_t} min")
    return float(max_t)


def load_frame(path):
    img = Image.open(path).convert('RGB')
    return transforms.ToTensor()(img) * 2.0 - 1.0


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


def setup_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        dev = torch.device('cpu')
        print("No GPU found — running on CPU")
    return dev


# ═════════════════════════════════════════════════════════════════════════════
#  DATASET
# ═════════════════════════════════════════════════════════════════════════════

class PETPairDataset(Dataset):
    def __init__(self, tumor_names, data_root, subfolder,
                 n_context, max_time_min, augment=False, label=''):
        self.n_context = n_context
        self.max_time  = max_time_min
        self.augment   = augment
        self.samples   = []

        tag = f' [{label}]' if label else ''
        print(f"\n{'Tumor':<20} {'Total':>7} {'Pairs':>8}{tag}")
        print("─" * 38)
        total_pairs = 0

        for tumor in tumor_names:
            files = get_files(Path(data_root) / tumor, subfolder)
            if not files:
                print(f"  {tumor:<18} ❌ not found"); continue
            if len(files) <= n_context:
                print(f"  {tumor:<18} ❌ too few ({len(files)})"); continue

            frames, times = [], []
            for path in files:
                t_min = extract_timepoint(Path(path).name)
                frames.append(load_frame(path))
                times.append(t_min / max_time_min)

            frames = torch.stack(frames)
            times  = torch.tensor(times, dtype=torch.float32)
            ctx_frames = frames[:n_context]
            ctx_times  = times[:n_context]

            n_pairs = 0
            for i in range(n_context, len(files)):
                self.samples.append({
                    'ctx': ctx_frames, 'ctx_t': ctx_times,
                    't_norm': times[i], 'target': frames[i], 'tumor': tumor,
                })
                n_pairs += 1
            total_pairs += n_pairs
            print(f"  {tumor:<18} {len(files):>7} {n_pairs:>8}")

        print("─" * 38)
        print(f"  {'TOTAL':<18} {'':>7} {total_pairs:>8} pairs\n")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ctx = s['ctx'].clone(); tgt = s['target'].clone()
        if self.augment:
            if random.random() > 0.5:
                ctx = torch.flip(ctx, dims=[3]); tgt = torch.flip(tgt, dims=[2])
            if random.random() > 0.5:
                ctx = torch.flip(ctx, dims=[2]); tgt = torch.flip(tgt, dims=[1])
        return ctx, s['ctx_t'], s['t_norm'], tgt


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL — Attention U-Net
# ═════════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

    def forward(self, t):
        half  = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=-1))


def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AttentionGate(nn.Module):
    """
    Additive attention gate (Oktay et al. 2018).
    g  = gating signal (from decoder, lower resolution)
    x  = skip connection (from encoder, higher resolution)
    """
    def __init__(self, f_g, f_x, f_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, 1, bias=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_x, f_int, 1, bias=False),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # align spatial size of g1 to x1 (g is at lower res)
        g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """U-Net with additive attention gates on skip connections."""
    def __init__(self, in_ch, out_ch=3, features=64, time_emb_dim=64):
        super().__init__()
        f = features
        self.time_emb  = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, f * 8)

        # Encoder
        self.enc1 = double_conv(in_ch, f)
        self.enc2 = double_conv(f,    f*2)
        self.enc3 = double_conv(f*2,  f*4)
        self.enc4 = double_conv(f*4,  f*8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = double_conv(f*8, f*8)

        # Attention gates
        self.ag4 = AttentionGate(f_g=f*8, f_x=f*8, f_int=f*4)
        self.ag3 = AttentionGate(f_g=f*8, f_x=f*4, f_int=f*2)
        self.ag2 = AttentionGate(f_g=f*4, f_x=f*2, f_int=f)
        self.ag1 = AttentionGate(f_g=f*2, f_x=f,   f_int=f//2)

        # Decoder
        self.up4  = nn.ConvTranspose2d(f*8,  f*8, 2, stride=2)
        self.dec4 = double_conv(f*16, f*8)
        self.up3  = nn.ConvTranspose2d(f*8,  f*4, 2, stride=2)
        self.dec3 = double_conv(f*8,  f*4)
        self.up2  = nn.ConvTranspose2d(f*4,  f*2, 2, stride=2)
        self.dec2 = double_conv(f*4,  f*2)
        self.up1  = nn.ConvTranspose2d(f*2,  f,   2, stride=2)
        self.dec1 = double_conv(f*2,  f)

        self.out_layer = nn.Sequential(
            nn.Conv2d(f, out_ch, 1), nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias  is not None: nn.init.zeros_(m.bias)

    def forward(self, ctx, t):
        t_bias = self.time_proj(self.time_emb(t))

        e1 = self.enc1(ctx)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b  = self.bottleneck(self.pool(e4))
        b  = b + t_bias[:, :, None, None]

        # Decoder with attention gates on skip connections
        e4a = self.ag4(b,  e4)
        d4  = self.dec4(torch.cat([self.up4(b),  e4a], dim=1))

        e3a = self.ag3(d4, e3)
        d3  = self.dec3(torch.cat([self.up3(d4), e3a], dim=1))

        e2a = self.ag2(d3, e2)
        d2  = self.dec2(torch.cat([self.up2(d3), e2a], dim=1))

        e1a = self.ag1(d2, e1)
        d1  = self.dec1(torch.cat([self.up1(d2), e1a], dim=1))

        return self.out_layer(d1)


# ═════════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _gaussian_kernel(kernel_size=11, sigma=1.5, channels=3):
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2*sigma**2)); g /= g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)
    return kernel.expand(channels, 1, -1, -1)


def ssim_loss_fn(pred, target, kernel_size=11, sigma=1.5,
                 C1=0.01**2, C2=0.03**2):
    C = pred.shape[1]
    kernel = _gaussian_kernel(kernel_size, sigma, C).to(pred.device)
    pad = kernel_size // 2
    mu_p  = F.conv2d(pred,   kernel, padding=pad, groups=C)
    mu_t  = F.conv2d(target, kernel, padding=pad, groups=C)
    mu_p2, mu_t2, mu_pt = mu_p*mu_p, mu_t*mu_t, mu_p*mu_t
    sp2 = F.conv2d(pred*pred,     kernel, padding=pad, groups=C) - mu_p2
    st2 = F.conv2d(target*target, kernel, padding=pad, groups=C) - mu_t2
    spt = F.conv2d(pred*target,   kernel, padding=pad, groups=C) - mu_pt
    ssim_map = ((2*mu_pt+C1)*(2*spt+C2)) / ((mu_p2+mu_t2+C1)*(sp2+st2+C2))
    return 1.0 - ssim_map.mean()


def recon_loss(pred, target, lambda_l1=1.0, lambda_ssim=0.5):
    l1   = F.l1_loss(pred, target)
    ssim = ssim_loss_fn(pred, target)
    return lambda_l1 * l1 + lambda_ssim * ssim, l1.item(), ssim.item()


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred_arr, gt_arr):
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    ssim_s, psnr_s, mae_s = [], [], []
    for p, g in zip(pred_arr, gt_arr):
        ssim_s.append(sk_ssim(g, p, data_range=1.0, channel_axis=2))
        psnr_s.append(sk_psnr(g, p, data_range=1.0))
        mae_s.append(float(np.abs(g - p).mean()))
    return np.array(ssim_s), np.array(psnr_s), np.array(mae_s)


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def save_training_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['total'],    color='steelblue', lw=2, label='total (train)')
    axes[0].plot(history['val_loss'], color='tomato',    lw=2,
                 linestyle='--', label='total (val)')
    axes[0].set_title('Total Loss — Train vs Validation')
    axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['l1'],   color='seagreen', lw=2, label='L1 (train)')
    axes[1].plot(history['ssim'], color='purple',   lw=1.5,
                 linestyle=':', label='SSIM-loss (train)')
    axes[1].set_title('Loss Components')
    axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def save_comparison_grid(ctx_frames, pred_frames, gt_frames,
                         ctx_times_min, tgt_times_min, tumor_name, path, n_show=10):
    T       = len(pred_frames)
    indices = np.linspace(0, T-1, min(n_show, T), dtype=int)
    n_cols  = len(indices)
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 2.2, 7))
    if n_cols == 1: axes = axes[:, None]
    for col, idx in enumerate(indices):
        t_min = int(tgt_times_min[idx])
        if col < len(ctx_times_min):
            cimg = (ctx_frames[col] * 0.5 + 0.5).clip(0, 1)
            axes[0, col].imshow(cimg.transpose(1, 2, 0))
            axes[0, col].set_title(f"ctx\n{int(ctx_times_min[col])}min", fontsize=7)
        else:
            axes[0, col].axis('off')
        axes[1, col].imshow(pred_frames[idx])
        axes[1, col].set_title(f"pred\n{t_min}min", fontsize=7)
        axes[2, col].imshow(gt_frames[idx])
        axes[2, col].set_title(f"GT\n{t_min}min", fontsize=7)
    for ax in axes.flat: ax.axis('off')
    axes[0, 0].set_ylabel("Context",      fontsize=9)
    axes[1, 0].set_ylabel("Predicted",    fontsize=9)
    axes[2, 0].set_ylabel("Ground Truth", fontsize=9)
    plt.suptitle(
        f"Attention U-Net — {tumor_name}  |  First {N_CONTEXT} frames given → rest predicted",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")


def save_tac(ctx_mean, ctx_t, pred_mean, gt_mean, tgt_t, tumor_name, path):
    plt.figure(figsize=(11, 4))
    plt.plot(ctx_t,  ctx_mean,  'o-', color='gray',      lw=2, ms=5,
             label=f'Context ({N_CONTEXT} frames given)', zorder=5)
    plt.plot(tgt_t,  gt_mean,   '-',  color='steelblue', lw=2.5,
             label='Ground Truth', zorder=3)
    plt.plot(tgt_t,  pred_mean, '--', color='tomato',    lw=2.5,
             label='Predicted (Attention U-Net)', zorder=4)
    plt.axvline(ctx_t[-1], color='black', lw=1.2, linestyle=':',
                alpha=0.7, label='Prediction boundary')
    plt.xlabel('Time (minutes)', fontsize=11)
    plt.ylabel('Mean Pixel Intensity', fontsize=11)
    plt.title(f'Time-Activity Curve — {tumor_name}', fontsize=12)
    plt.legend(fontsize=10); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def save_metrics_plot(ssim_s, psnr_s, mae_s, times_min, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, scores, lbl, col in zip(
            axes,
            [ssim_s, psnr_s, mae_s],
            ['SSIM ↑', 'PSNR dB ↑', 'MAE ↓'],
            ['steelblue', 'seagreen', 'tomato']):
        ax.plot(times_min, scores, color=col, lw=2)
        ax.set_xlabel('Time (min)'); ax.set_title(lbl); ax.grid(alpha=0.3)
        ax.axhline(np.mean(scores), color='black', linestyle='--',
                   alpha=0.5, label=f'mean={np.mean(scores):.3f}')
        ax.legend(fontsize=9)
    plt.suptitle('Prediction Quality Over Time', fontsize=12)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPER
# ═════════════════════════════════════════════════════════════════════════════

def run_inference(model, device, test_tumor):
    test_files = get_files(Path(DATA_ROOT) / test_tumor, IMG_SUBFOLDER)
    if not test_files:
        return None, None, None, None
    test_frames = torch.stack([load_frame(p) for p in test_files])
    test_times  = torch.tensor(
        [extract_timepoint(Path(p).name) / MAX_TIME_MIN for p in test_files],
        dtype=torch.float32)
    ctx_frames    = test_frames[:N_CONTEXT].unsqueeze(0).to(device)
    ctx_flat      = ctx_frames.view(1, -1, IMG_SIZE, IMG_SIZE)
    ctx_times_min = [int(extract_timepoint(Path(p).name))
                     for p in test_files[:N_CONTEXT]]
    tgt_times_min = [int(extract_timepoint(Path(p).name))
                     for p in test_files[N_CONTEXT:]]
    model.eval()
    pred_list, gt_list = [], []
    with torch.no_grad():
        for i in range(len(test_files[N_CONTEXT:])):
            t_norm = test_times[N_CONTEXT + i].unsqueeze(0).to(device)
            pred   = model(ctx_flat, t_norm)
            pred_list.append(denorm(pred[0]).cpu().permute(1,2,0).numpy())
            gt_list.append(denorm(test_frames[N_CONTEXT+i]).permute(1,2,0).numpy())
    return (np.array(pred_list), np.array(gt_list),
            ctx_times_min, tgt_times_min)


# ═════════════════════════════════════════════════════════════════════════════
#  TRAIN ONE FOLD
# ═════════════════════════════════════════════════════════════════════════════

def train_one_fold(train_tumors, val_tumors, test_tumor,
                   epochs, out_dir, device, fold_label=''):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nBuilding TRAINING dataset{' — ' + fold_label if fold_label else ''}:")
    train_ds = PETPairDataset(train_tumors, DATA_ROOT, IMG_SUBFOLDER,
                              N_CONTEXT, MAX_TIME_MIN, augment=True,  label='train')
    print(f"\nBuilding VALIDATION dataset{' — ' + fold_label if fold_label else ''}:")
    val_ds   = PETPairDataset(val_tumors,   DATA_ROOT, IMG_SUBFOLDER,
                              N_CONTEXT, MAX_TIME_MIN, augment=False, label='val')
    if len(train_ds) == 0:
        print("ERROR: No training samples."); return None

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(device.type == 'cuda'))

    in_ch = N_CONTEXT * 3
    model = AttentionUNet(in_ch, 3, FEATURES, TIME_EMB_DIM).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nAttention U-Net params: {n_params:,}")
    print(f"Training pairs        : {len(train_ds)}")
    print(f"Validation pairs      : {len(val_ds)}")
    print(f"Train tumors          : {train_tumors}")
    print(f"Val tumors            : {val_tumors}")
    print(f"Epochs                : {epochs}  (patience={PATIENCE}, min={MIN_EPOCHS})\n")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < epochs // 2: return 1.0
        return 1.0 - (epoch - epochs // 2) / (epochs // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history = {'total': [], 'l1': [], 'ssim': [], 'val_loss': []}
    best_val      = float('inf')
    patience_count = 0
    best_ckpt     = os.path.join(out_dir, 'best_model.pth')

    print("─" * 65)
    for epoch in range(1, epochs + 1):
        model.train()
        ep = {'total': [], 'l1': [], 'ssim': []}

        for ctx_f, _, t_norm, target in train_loader:
            B        = ctx_f.size(0)
            ctx_f    = ctx_f.to(device)
            t_norm   = t_norm.to(device)
            target   = target.to(device)
            ctx_flat = ctx_f.view(B, -1, IMG_SIZE, IMG_SIZE)

            pred = model(ctx_flat, t_norm)
            loss, l1, ssim = recon_loss(pred, target, LAMBDA_L1, LAMBDA_SSIM)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep['total'].append(loss.item())
            ep['l1'].append(l1)
            ep['ssim'].append(ssim)

        sched.step()
        for k in ep:
            history[k].append(np.mean(ep[k]))

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for ctx_f, _, t_norm, target in val_loader:
                B        = ctx_f.size(0)
                ctx_f    = ctx_f.to(device)
                t_norm   = t_norm.to(device)
                target   = target.to(device)
                ctx_flat = ctx_f.view(B, -1, IMG_SIZE, IMG_SIZE)
                pred = model(ctx_flat, t_norm)
                loss, _, _ = recon_loss(pred, target, LAMBDA_L1, LAMBDA_SSIM)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        history['val_loss'].append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            patience_count = 0
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'opt': opt.state_dict(), 'val_loss': avg_val,
                'history': history,
            }, best_ckpt)
        else:
            patience_count += 1

        if epoch % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f'model_ep{epoch}.pth'))

        if epoch % 50 == 0 or epoch == 1:
            print(f"Ep {epoch:4d}/{epochs} | "
                  f"loss={history['total'][-1]:.4f}  val={avg_val:.4f}  "
                  f"patience={patience_count}/{PATIENCE}")

        if epoch >= MIN_EPOCHS and patience_count >= PATIENCE:
            print(f"\n⚡ Early stopping at epoch {epoch}")
            break

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    save_training_curves(history, os.path.join(out_dir, 'training_curves.png'))

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"\nLoaded best model: epoch {ckpt['epoch']}  "
          f"val_loss={ckpt['val_loss']:.4f}")

    # Overfitting gap
    print("\n── Overfitting gap check ──────────────────────────────────")
    train_ssims = []
    for tumor in train_tumors:
        p_arr, g_arr, _, _ = run_inference(model, device, tumor)
        if p_arr is not None:
            s, _, _ = compute_metrics(p_arr, g_arr)
            train_ssims.append(s.mean())
            print(f"  Train tumor {tumor:<18}  SSIM={s.mean():.4f}")
    avg_train_ssim = np.mean(train_ssims)

    pred_arr, gt_arr, ctx_times_min, tgt_times_min = run_inference(
        model, device, test_tumor)
    ssim_s, psnr_s, mae_s = compute_metrics(pred_arr, gt_arr)
    test_ssim = ssim_s.mean()
    gap = avg_train_ssim - test_ssim
    print(f"\n  Avg train SSIM : {avg_train_ssim:.4f}")
    print(f"  Test  SSIM     : {test_ssim:.4f}")
    print(f"  Gap            : {gap:.4f}  ", end='')
    if gap < 0.05:   print("✓ No overfitting detected")
    elif gap < 0.15: print("~ Mild generalisation gap (acceptable for N=10)")
    else:            print("⚠ Large gap — possible overfitting")

    # Early / late split
    print("\n── Early / Late prediction analysis ───────────────────────")
    SPLIT_MIN  = 2000
    early_mask = np.array(tgt_times_min) < SPLIT_MIN
    late_mask  = ~early_mask
    if early_mask.sum() > 0 and late_mask.sum() > 0:
        e_ssim = ssim_s[early_mask]; l_ssim = ssim_s[late_mask]
        print(f"  Early (<{SPLIT_MIN}min, n={early_mask.sum()}):  "
              f"SSIM={e_ssim.mean():.4f} ± {e_ssim.std():.4f}")
        print(f"  Late  (≥{SPLIT_MIN}min, n={late_mask.sum()}):   "
              f"SSIM={l_ssim.mean():.4f} ± {l_ssim.std():.4f}")
        stat, p_val = stats.mannwhitneyu(e_ssim, l_ssim, alternative='greater')
        print(f"  Mann-Whitney U test (early > late): stat={stat:.1f}  p={p_val:.4f}  "
              f"{'✓ significant' if p_val < 0.05 else '✗ not significant'} at α=0.05")

    print(f"\n── Results on {test_tumor} ──────────────────────")
    print(f"  Mean SSIM : {ssim_s.mean():.4f} ± {ssim_s.std():.4f}")
    print(f"  Mean PSNR : {psnr_s.mean():.2f} ± {psnr_s.std():.2f} dB")
    print(f"  Mean MAE  : {mae_s.mean():.4f} ± {mae_s.std():.4f}")

    if not fold_label:
        print("\nSaving results...")
        save_comparison_grid(
            torch.stack([load_frame(p) for p in
                         get_files(Path(DATA_ROOT)/test_tumor,
                                    IMG_SUBFOLDER)[:N_CONTEXT]]).numpy(),
            pred_arr, gt_arr, ctx_times_min, tgt_times_min,
            test_tumor, os.path.join(out_dir, 'prediction_vs_GT.png')
        )
        test_files_all = get_files(Path(DATA_ROOT) / test_tumor, IMG_SUBFOLDER)
        ctx_frames_cpu = torch.stack([load_frame(p)
                                      for p in test_files_all[:N_CONTEXT]])
        ctx_mean  = denorm(ctx_frames_cpu).mean(dim=[1, 2, 3]).numpy()
        save_tac(ctx_mean, ctx_times_min,
                 pred_arr.mean(axis=(1,2,3)), gt_arr.mean(axis=(1,2,3)),
                 tgt_times_min, test_tumor,
                 os.path.join(out_dir, 'TAC_comparison.png'))
        save_metrics_plot(ssim_s, psnr_s, mae_s, tgt_times_min,
                          os.path.join(out_dir, 'metrics_over_time.png'))

    return ssim_s, psnr_s, mae_s


# ═════════════════════════════════════════════════════════════════════════════
#  LOOCV
# ═════════════════════════════════════════════════════════════════════════════

def run_loocv(device):
    print("\n" + "═"*65)
    print("  LEAVE-ONE-OUT CROSS-VALIDATION — Attention U-Net")
    print("═"*65)
    results = []
    for test_t in ALL_TUMORS:
        remaining = [t for t in ALL_TUMORS if t != test_t]
        val_t   = remaining[:2]
        train_t = remaining[2:]
        fold_dir = os.path.join(OUTPUT_DIR, 'loocv', test_t.replace('=',''))
        print(f"\n{'─'*55}")
        print(f"  Fold: test={test_t}  val={val_t}")
        print(f"{'─'*55}")
        out = train_one_fold(
            train_t, val_t, test_t,
            LOOCV_EPOCHS, fold_dir, device, fold_label=f'LOOCV:{test_t}'
        )
        if out is not None:
            ssim_s, psnr_s, mae_s = out
            results.append({
                'test_tumor': test_t,
                'ssim_mean': ssim_s.mean(), 'ssim_std': ssim_s.std(),
                'psnr_mean': psnr_s.mean(), 'psnr_std': psnr_s.std(),
                'mae_mean':  mae_s.mean(),  'mae_std':  mae_s.std(),
            })

    print("\n" + "═"*65)
    print("  LOOCV SUMMARY — Attention U-Net")
    print("═"*65)
    ssims = [r['ssim_mean'] for r in results]
    psnrs = [r['psnr_mean'] for r in results]
    maes  = [r['mae_mean']  for r in results]
    print(f"  {'Tumor':<20} {'SSIM':>8} {'PSNR':>10} {'MAE':>10}")
    print("  " + "─"*52)
    for r in results:
        print(f"  {r['test_tumor']:<20} {r['ssim_mean']:.4f}   "
              f"{r['psnr_mean']:>6.2f} dB   {r['mae_mean']:.4f}")
    print("  " + "─"*52)
    print(f"  {'MEAN ± STD':<20} "
          f"{np.mean(ssims):.4f}±{np.std(ssims):.4f}   "
          f"{np.mean(psnrs):>5.2f}±{np.std(psnrs):.2f} dB   "
          f"{np.mean(maes):.4f}±{np.std(maes):.4f}")

    paper_line = (
        f"In LOOCV across all {len(results)} tumors, Attention U-Net achieved "
        f"SSIM {np.mean(ssims):.3f}±{np.std(ssims):.3f}, "
        f"PSNR {np.mean(psnrs):.2f}±{np.std(psnrs):.2f} dB, "
        f"MAE {np.mean(maes):.4f}±{np.std(maes):.4f}."
    )
    print(f"\n  Paper sentence:\n  \"{paper_line}\"")

    csv_path = os.path.join(OUTPUT_DIR, 'loocv_results_attention_unet.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print(f"\n  LOOCV CSV saved: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Attention U-Net — PET Tumor Prediction')
    parser.add_argument('--loocv', action='store_true',
                        help='Run Leave-One-Out Cross-Validation instead of single run')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = setup_device()

    global MAX_TIME_MIN
    if MAX_TIME_MIN == 0:
        MAX_TIME_MIN = autodetect_max_time(ALL_TUMORS, DATA_ROOT, IMG_SUBFOLDER)

    if args.loocv:
        run_loocv(device)
    else:
        train_one_fold(
            TRAIN_TUMORS, VAL_TUMORS, TEST_TUMOR,
            EPOCHS, OUTPUT_DIR, device
        )
        print(f"\n{'═'*55}")
        print(f"  Results saved to: {OUTPUT_DIR}")
        for f in ['best_model.pth', 'training_curves.png',
                  'prediction_vs_GT.png', 'TAC_comparison.png',
                  'metrics_over_time.png']:
            p   = os.path.join(OUTPUT_DIR, f)
            chk = "✓" if os.path.exists(p) else "✗"
            print(f"    {chk}  {f}")
        print(f"{'═'*55}\n")


if __name__ == '__main__':
    main()
