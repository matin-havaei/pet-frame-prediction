"""
pet_pix2pix_paper.py
====================
Pix2Pix (cGAN WITHOUT time conditioning) for PET Tumor Simulation Completion
— Paper-ready version — matches cGAN style exactly —

Role in paper : ABLATION BASELINE — identical to cGAN but the target
                timepoint is NOT given to the generator. This isolates the
                contribution of temporal conditioning: if cGAN >> Pix2Pix,
                the time embedding is the key driver.

Architecture  : U-Net generator + PatchGAN discriminator (Isola et al. 2017)
                Context frames stacked → generator → predicted frame
                NO time embedding anywhere (neither G nor D)
Loss          : LSGAN (MSE) + L1 + SSIM
Split         : 7 train / 2 val / 1 test

Reference: Isola P, et al. Image-to-Image Translation with Conditional
           Adversarial Networks. CVPR 2017. arXiv:1611.07004

Setup:
  pip install torch torchvision numpy pillow matplotlib tqdm scikit-image scipy

Run:
  python pet_pix2pix_paper.py
  python pet_pix2pix_paper.py --loocv
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

# Training
BATCH_SIZE   = 16
EPOCHS       = 800
LR_G         = 2e-4
LR_D         = 1e-4
BETA1        = 0.5
LAMBDA_L1    = 100.0
LAMBDA_SSIM  = 0.3
PATIENCE     = 150
MIN_EPOCHS   = 300

# LOOCV
LOOCV_EPOCHS = 600

SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITIES  (identical across all models)
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
            if files: return files
    return []

def autodetect_max_time(tumor_names, data_root, subfolder):
    max_t = 0
    for tumor in tumor_names:
        for f in get_files(Path(data_root) / tumor, subfolder):
            t = extract_timepoint(Path(f).name)
            if t and t > max_t: max_t = t
    assert max_t > 0, "Could not detect timepoints"
    print(f"Auto-detected MAX_TIME_MIN: {max_t} min")
    return float(max_t)

def load_frame(path):
    return transforms.ToTensor()(Image.open(path).convert('RGB')) * 2.0 - 1.0

def denorm(t): return (t * 0.5 + 0.5).clamp(0, 1)

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
        self.augment = augment
        self.samples = []
        tag = f' [{label}]' if label else ''
        print(f"\n{'Tumor':<20} {'Total':>7} {'Pairs':>8}{tag}")
        print("─" * 38)
        total_pairs = 0
        for tumor in tumor_names:
            files = get_files(Path(data_root) / tumor, subfolder)
            if not files: print(f"  {tumor:<18} ❌ not found"); continue
            if len(files) <= n_context: print(f"  {tumor:<18} ❌ too few"); continue
            frames = torch.stack([load_frame(p) for p in files])
            times  = torch.tensor(
                [extract_timepoint(Path(p).name)/max_time_min for p in files],
                dtype=torch.float32)
            ctx_f = frames[:n_context]; ctx_t = times[:n_context]
            n_pairs = 0
            for i in range(n_context, len(files)):
                self.samples.append({
                    'ctx': ctx_f, 'ctx_t': ctx_t,
                    't_norm': times[i], 'target': frames[i],
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
            if random.random() > 0.5: ctx = torch.flip(ctx, [3]); tgt = torch.flip(tgt, [2])
            if random.random() > 0.5: ctx = torch.flip(ctx, [2]); tgt = torch.flip(tgt, [1])
        return ctx, s['ctx_t'], s['t_norm'], tgt


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL — Pix2Pix (NO time conditioning)
# ═════════════════════════════════════════════════════════════════════════════

def down_block(in_ch, out_ch, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
    if norm: layers.append(nn.InstanceNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def up_block(in_ch, out_ch, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout: layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """
    Standard Pix2Pix U-Net generator.
    Input : stacked context frames  (B, N*3, H, W)
    Output: predicted frame          (B, 3,   H, W)
    NO time embedding — this is the key difference from cGAN.
    """
    def __init__(self, in_ch, out_ch=3, features=64):
        super().__init__()
        f = features
        # Encoder
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, f, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.e2 = down_block(f,    f*2)
        self.e3 = down_block(f*2,  f*4)
        self.e4 = down_block(f*4,  f*8)
        self.e5 = down_block(f*8,  f*8, norm=False)  # bottleneck
        # Decoder
        self.d5 = up_block(f*8,  f*8,  dropout=True)
        self.d4 = up_block(f*16, f*4)
        self.d3 = up_block(f*8,  f*2)
        self.d2 = up_block(f*4,  f)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(f*2, out_ch, 4, 2, 1), nn.Tanh()
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, ctx):
        e1 = self.e1(ctx)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        d5 = self.d5(e5)
        d4 = self.d4(torch.cat([d5, e4], 1))
        d3 = self.d3(torch.cat([d4, e3], 1))
        d2 = self.d2(torch.cat([d3, e2], 1))
        return self.out(torch.cat([d2, e1], 1))


class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator — identical to cGAN version."""
    def __init__(self, in_ch):
        super().__init__()
        f = 64
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, f,    4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(f,    f*2,  4, 2, 1, bias=False), nn.InstanceNorm2d(f*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(f*2,  f*4,  4, 2, 1, bias=False), nn.InstanceNorm2d(f*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(f*4,  f*8,  4, 1, 1, bias=False), nn.InstanceNorm2d(f*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(f*8,  1,    4, 1, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x): return self.model(x)


# ═════════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _gaussian_kernel(kernel_size=11, sigma=1.5, channels=3):
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords**2)/(2*sigma**2)); g /= g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)

def ssim_loss_fn(pred, target, C1=0.01**2, C2=0.03**2):
    C = pred.shape[1]; kernel = _gaussian_kernel(channels=C).to(pred.device); pad = 5
    mu_p = F.conv2d(pred,   kernel, padding=pad, groups=C)
    mu_t = F.conv2d(target, kernel, padding=pad, groups=C)
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p*mu_t
    sp2 = F.conv2d(pred**2,    kernel, padding=pad, groups=C) - mu_p2
    st2 = F.conv2d(target**2,  kernel, padding=pad, groups=C) - mu_t2
    spt = F.conv2d(pred*target, kernel, padding=pad, groups=C) - mu_pt
    return 1.0 - (((2*mu_pt+C1)*(2*spt+C2))/((mu_p2+mu_t2+C1)*(sp2+st2+C2))).mean()

criterion_GAN = nn.MSELoss()   # LSGAN


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
        mae_s.append(float(np.abs(g-p).mean()))
    return np.array(ssim_s), np.array(psnr_s), np.array(mae_s)


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def save_training_curves(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['G_total'], color='steelblue', lw=2, label='G total (train)')
    axes[0].plot(history['val_l1'],  color='tomato', lw=2, linestyle='--', label='L1 (val)')
    axes[0].set_title('Generator Loss vs Val L1'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['G_adv'], color='seagreen', lw=2, label='G adversarial')
    axes[1].plot(history['D'],     color='orange',   lw=2, label='Discriminator')
    axes[1].set_title('Adversarial Losses'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(history['G_l1'],   color='purple', lw=2,   label='L1')
    axes[2].plot(history['G_ssim'], color='teal',   lw=1.5, linestyle=':', label='SSIM-loss')
    axes[2].set_title('Reconstruction Components'); axes[2].legend(); axes[2].grid(alpha=0.3)
    for ax in axes: ax.set_xlabel('Epoch')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")

def save_comparison_grid(ctx_frames, pred_frames, gt_frames,
                         ctx_times_min, tgt_times_min, tumor_name, path, n_show=10):
    T = len(pred_frames); indices = np.linspace(0, T-1, min(n_show,T), dtype=int)
    n_cols = len(indices)
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols*2.2, 7))
    if n_cols == 1: axes = axes[:, None]
    for col, idx in enumerate(indices):
        t_min = int(tgt_times_min[idx])
        if col < len(ctx_times_min):
            cimg = (ctx_frames[col]*0.5+0.5).clip(0,1)
            axes[0,col].imshow(cimg.transpose(1,2,0))
            axes[0,col].set_title(f"ctx\n{int(ctx_times_min[col])}min", fontsize=7)
        else: axes[0,col].axis('off')
        axes[1,col].imshow(pred_frames[idx]); axes[1,col].set_title(f"pred\n{t_min}min", fontsize=7)
        axes[2,col].imshow(gt_frames[idx]);   axes[2,col].set_title(f"GT\n{t_min}min",   fontsize=7)
    for ax in axes.flat: ax.axis('off')
    axes[0,0].set_ylabel("Context", fontsize=9)
    axes[1,0].set_ylabel("Predicted", fontsize=9)
    axes[2,0].set_ylabel("Ground Truth", fontsize=9)
    plt.suptitle(f"Pix2Pix — {tumor_name}  |  First {N_CONTEXT} frames → rest predicted", fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

def save_tac(ctx_mean, ctx_t, pred_mean, gt_mean, tgt_t, tumor_name, path):
    plt.figure(figsize=(11,4))
    plt.plot(ctx_t, ctx_mean,  'o-', color='gray',      lw=2, ms=5, label=f'Context ({N_CONTEXT} frames)', zorder=5)
    plt.plot(tgt_t, gt_mean,   '-',  color='steelblue', lw=2.5, label='Ground Truth', zorder=3)
    plt.plot(tgt_t, pred_mean, '--', color='tomato',    lw=2.5, label='Predicted (Pix2Pix)', zorder=4)
    plt.axvline(ctx_t[-1], color='black', lw=1.2, linestyle=':', alpha=0.7, label='Prediction boundary')
    plt.xlabel('Time (minutes)', fontsize=11); plt.ylabel('Mean Pixel Intensity', fontsize=11)
    plt.title(f'Time-Activity Curve — {tumor_name}', fontsize=12)
    plt.legend(fontsize=10); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close(); print(f"  Saved: {path}")

def save_metrics_plot(ssim_s, psnr_s, mae_s, times_min, path):
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for ax, scores, lbl, col in zip(axes, [ssim_s,psnr_s,mae_s],
                                    ['SSIM ↑','PSNR dB ↑','MAE ↓'],
                                    ['steelblue','seagreen','tomato']):
        ax.plot(times_min, scores, color=col, lw=2)
        ax.set_xlabel('Time (min)'); ax.set_title(lbl); ax.grid(alpha=0.3)
        ax.axhline(np.mean(scores), color='black', linestyle='--', alpha=0.5,
                   label=f'mean={np.mean(scores):.3f}'); ax.legend(fontsize=9)
    plt.suptitle('Prediction Quality Over Time', fontsize=12)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPER
# ═════════════════════════════════════════════════════════════════════════════

def run_inference(G, device, test_tumor):
    test_files = get_files(Path(DATA_ROOT)/test_tumor, IMG_SUBFOLDER)
    if not test_files: return None, None, None, None
    test_frames   = torch.stack([load_frame(p) for p in test_files])
    ctx_flat      = test_frames[:N_CONTEXT].unsqueeze(0).view(1,-1,IMG_SIZE,IMG_SIZE).to(device)
    ctx_times_min = [int(extract_timepoint(Path(p).name)) for p in test_files[:N_CONTEXT]]
    tgt_times_min = [int(extract_timepoint(Path(p).name)) for p in test_files[N_CONTEXT:]]
    G.eval()
    pred_list, gt_list = [], []
    with torch.no_grad():
        pred = G(ctx_flat)   # single forward pass — no time conditioning
        # replicate for all target frames (model ignores timepoint)
        for i in range(len(test_files)-N_CONTEXT):
            pred_list.append(denorm(pred[0]).cpu().permute(1,2,0).numpy())
            gt_list.append(denorm(test_frames[N_CONTEXT+i]).permute(1,2,0).numpy())
    return np.array(pred_list), np.array(gt_list), ctx_times_min, tgt_times_min


# ═════════════════════════════════════════════════════════════════════════════
#  TRAIN ONE FOLD
# ═════════════════════════════════════════════════════════════════════════════

def train_one_fold(train_tumors, val_tumors, test_tumor,
                   epochs, out_dir, device, fold_label=''):
    os.makedirs(out_dir, exist_ok=True)
    label = f' — {fold_label}' if fold_label else ''

    print(f"\nBuilding TRAINING dataset{label}:")
    train_ds = PETPairDataset(train_tumors, DATA_ROOT, IMG_SUBFOLDER,
                              N_CONTEXT, MAX_TIME_MIN, augment=True,  label='train')
    print(f"\nBuilding VALIDATION dataset{label}:")
    val_ds   = PETPairDataset(val_tumors,   DATA_ROOT, IMG_SUBFOLDER,
                              N_CONTEXT, MAX_TIME_MIN, augment=False, label='val')
    if len(train_ds) == 0: print("ERROR: No training samples."); return None

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type=='cuda'))
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(device.type=='cuda'))

    in_ch = N_CONTEXT * 3
    G = UNetGenerator(in_ch, 3).to(device)
    D = PatchDiscriminator(in_ch + 3).to(device)    # ctx + real/fake target

    n_G = sum(p.numel() for p in G.parameters() if p.requires_grad)
    n_D = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"\nGenerator params    : {n_G:,}")
    print(f"Discriminator params: {n_D:,}")
    print(f"Training pairs      : {len(train_ds)}")
    print(f"NOTE: No time conditioning — ablation of cGAN")
    print(f"Epochs              : {epochs}  (patience={PATIENCE}, min={MIN_EPOCHS})\n")

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, 0.999))

    history = {'G_total':[], 'G_adv':[], 'G_l1':[], 'G_ssim':[], 'D':[], 'val_l1':[]}
    best_val = float('inf'); patience_count = 0
    best_ckpt = os.path.join(out_dir, 'best_model.pth')

    print("─" * 65)
    for epoch in range(1, epochs+1):
        G.train(); D.train()
        ep = {k: [] for k in ['G_total','G_adv','G_l1','G_ssim','D']}

        for ctx_f, _, _, target in train_loader:
            B       = ctx_f.size(0)
            ctx_f   = ctx_f.to(device)
            target  = target.to(device)
            ctx_flat = ctx_f.view(B, -1, IMG_SIZE, IMG_SIZE)

            fake = G(ctx_flat)

            # ── Discriminator ──────────────────────────────────────────
            opt_D.zero_grad()
            real_pair = torch.cat([ctx_flat, target], 1)
            fake_pair = torch.cat([ctx_flat, fake.detach()], 1)
            real_lbl  = torch.ones(B,1,*D(real_pair).shape[2:], device=device)
            fake_lbl  = torch.zeros_like(real_lbl)
            loss_D = 0.5*(criterion_GAN(D(real_pair), real_lbl) +
                          criterion_GAN(D(fake_pair), fake_lbl))
            loss_D.backward(); opt_D.step()

            # ── Generator ──────────────────────────────────────────────
            opt_G.zero_grad()
            fake_pair2 = torch.cat([ctx_flat, fake], 1)
            loss_adv  = criterion_GAN(D(fake_pair2), real_lbl)
            loss_l1   = F.l1_loss(fake, target)
            loss_ssim = ssim_loss_fn(fake, target)
            loss_G    = loss_adv + LAMBDA_L1*loss_l1 + LAMBDA_SSIM*loss_ssim
            loss_G.backward(); opt_G.step()

            ep['G_total'].append(loss_G.item()); ep['G_adv'].append(loss_adv.item())
            ep['G_l1'].append(loss_l1.item());   ep['G_ssim'].append(loss_ssim.item())
            ep['D'].append(loss_D.item())

        for k in ep: history[k].append(np.mean(ep[k]))

        # Validation
        G.eval()
        val_l1s = []
        with torch.no_grad():
            for ctx_f, _, _, target in val_loader:
                B       = ctx_f.size(0)
                ctx_flat = ctx_f.view(B,-1,IMG_SIZE,IMG_SIZE).to(device)
                target   = target.to(device)
                fake = G(ctx_flat)
                val_l1s.append(F.l1_loss(fake, target).item())
        avg_val = np.mean(val_l1s)
        history['val_l1'].append(avg_val)

        if avg_val < best_val:
            best_val = avg_val; patience_count = 0
            torch.save({'epoch': epoch, 'G': G.state_dict(), 'D': D.state_dict(),
                        'val_l1': avg_val, 'history': history}, best_ckpt)
        else:
            patience_count += 1

        if epoch % 100 == 0:
            torch.save(G.state_dict(), os.path.join(out_dir, f'G_ep{epoch}.pth'))

        if epoch % 50 == 0 or epoch == 1:
            print(f"Ep {epoch:4d}/{epochs} | "
                  f"G={history['G_total'][-1]:.4f}  D={history['D'][-1]:.4f}  "
                  f"val_L1={avg_val:.4f}  patience={patience_count}/{PATIENCE}")

        if epoch >= MIN_EPOCHS and patience_count >= PATIENCE:
            print(f"\n⚡ Early stopping at epoch {epoch}"); break

    print(f"\nTraining complete. Best val L1: {best_val:.4f}")
    save_training_curves(history, os.path.join(out_dir, 'training_curves.png'))

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    G.load_state_dict(ckpt['G'])
    print(f"\nLoaded best model: epoch {ckpt['epoch']}  val_L1={ckpt['val_l1']:.4f}")

    # Overfitting gap
    print("\n── Overfitting gap check ──────────────────────────────────")
    train_ssims = []
    for tumor in train_tumors:
        p_arr, g_arr, _, _ = run_inference(G, device, tumor)
        if p_arr is not None:
            s, _, _ = compute_metrics(p_arr, g_arr)
            train_ssims.append(s.mean())
            print(f"  Train tumor {tumor:<18}  SSIM={s.mean():.4f}")
    avg_train_ssim = np.mean(train_ssims)

    pred_arr, gt_arr, ctx_times_min, tgt_times_min = run_inference(G, device, test_tumor)
    ssim_s, psnr_s, mae_s = compute_metrics(pred_arr, gt_arr)
    gap = avg_train_ssim - ssim_s.mean()
    print(f"\n  Avg train SSIM : {avg_train_ssim:.4f}")
    print(f"  Test  SSIM     : {ssim_s.mean():.4f}")
    print(f"  Gap            : {gap:.4f}  ", end='')
    if gap < 0.05:   print("✓ No overfitting")
    elif gap < 0.15: print("~ Mild gap (acceptable for N=10)")
    else:            print("⚠ Large gap")

    # Early / late
    print("\n── Early / Late prediction analysis ───────────────────────")
    SPLIT_MIN = 2000
    early_mask = np.array(tgt_times_min) < SPLIT_MIN; late_mask = ~early_mask
    if early_mask.sum() > 0 and late_mask.sum() > 0:
        e_ssim = ssim_s[early_mask]; l_ssim = ssim_s[late_mask]
        print(f"  Early (<{SPLIT_MIN}min, n={early_mask.sum()}):  SSIM={e_ssim.mean():.4f} ± {e_ssim.std():.4f}")
        print(f"  Late  (≥{SPLIT_MIN}min, n={late_mask.sum()}):   SSIM={l_ssim.mean():.4f} ± {l_ssim.std():.4f}")
        stat, p_val = stats.mannwhitneyu(e_ssim, l_ssim, alternative='greater')
        print(f"  Mann-Whitney U: stat={stat:.1f}  p={p_val:.4f}  "
              f"{'✓ significant' if p_val<0.05 else '✗ not significant'} at α=0.05")

    print(f"\n── Results on {test_tumor} ──────────────────────")
    print(f"  Mean SSIM : {ssim_s.mean():.4f} ± {ssim_s.std():.4f}")
    print(f"  Mean PSNR : {psnr_s.mean():.2f} ± {psnr_s.std():.2f} dB")
    print(f"  Mean MAE  : {mae_s.mean():.4f} ± {mae_s.std():.4f}")

    if not fold_label:
        save_comparison_grid(
            torch.stack([load_frame(p) for p in
                         get_files(Path(DATA_ROOT)/test_tumor, IMG_SUBFOLDER)[:N_CONTEXT]]).numpy(),
            pred_arr, gt_arr, ctx_times_min, tgt_times_min,
            test_tumor, os.path.join(out_dir, 'prediction_vs_GT.png'))
        ctx_frames_cpu = torch.stack([load_frame(p) for p in
                         get_files(Path(DATA_ROOT)/test_tumor, IMG_SUBFOLDER)[:N_CONTEXT]])
        save_tac(denorm(ctx_frames_cpu).mean(dim=[1,2,3]).numpy(), ctx_times_min,
                 pred_arr.mean(axis=(1,2,3)), gt_arr.mean(axis=(1,2,3)),
                 tgt_times_min, test_tumor, os.path.join(out_dir, 'TAC_comparison.png'))
        save_metrics_plot(ssim_s, psnr_s, mae_s, tgt_times_min,
                          os.path.join(out_dir, 'metrics_over_time.png'))

    return ssim_s, psnr_s, mae_s


# ═════════════════════════════════════════════════════════════════════════════
#  LOOCV
# ═════════════════════════════════════════════════════════════════════════════

def run_loocv(device):
    print("\n" + "═"*65)
    print("  LEAVE-ONE-OUT CROSS-VALIDATION — Pix2Pix")
    print("═"*65)
    results = []
    for test_t in ALL_TUMORS:
        remaining = [t for t in ALL_TUMORS if t != test_t]
        val_t = remaining[:2]; train_t = remaining[2:]
        fold_dir = os.path.join(OUTPUT_DIR, 'loocv', test_t.replace('=',''))
        print(f"\n{'─'*55}\n  Fold: test={test_t}  val={val_t}\n{'─'*55}")
        out = train_one_fold(train_t, val_t, test_t, LOOCV_EPOCHS,
                             fold_dir, device, fold_label=f'LOOCV:{test_t}')
        if out is not None:
            ssim_s, psnr_s, mae_s = out
            results.append({'test_tumor': test_t,
                             'ssim_mean': ssim_s.mean(), 'ssim_std': ssim_s.std(),
                             'psnr_mean': psnr_s.mean(), 'psnr_std': psnr_s.std(),
                             'mae_mean':  mae_s.mean(),  'mae_std':  mae_s.std()})

    print("\n" + "═"*65 + "\n  LOOCV SUMMARY — Pix2Pix\n" + "═"*65)
    ssims = [r['ssim_mean'] for r in results]
    psnrs = [r['psnr_mean'] for r in results]
    maes  = [r['mae_mean']  for r in results]
    print(f"  {'Tumor':<20} {'SSIM':>8} {'PSNR':>10} {'MAE':>10}\n  {'─'*52}")
    for r in results:
        print(f"  {r['test_tumor']:<20} {r['ssim_mean']:.4f}   {r['psnr_mean']:>6.2f} dB   {r['mae_mean']:.4f}")
    print(f"  {'─'*52}\n  {'MEAN ± STD':<20} "
          f"{np.mean(ssims):.4f}±{np.std(ssims):.4f}   "
          f"{np.mean(psnrs):>5.2f}±{np.std(psnrs):.2f} dB   "
          f"{np.mean(maes):.4f}±{np.std(maes):.4f}")
    paper_line = (f"In LOOCV across all {len(results)} tumors, Pix2Pix (no time conditioning) achieved "
                  f"SSIM {np.mean(ssims):.3f}±{np.std(ssims):.3f}, "
                  f"PSNR {np.mean(psnrs):.2f}±{np.std(psnrs):.2f} dB, "
                  f"MAE {np.mean(maes):.4f}±{np.std(maes):.4f}.")
    print(f"\n  Paper sentence:\n  \"{paper_line}\"")
    csv_path = os.path.join(OUTPUT_DIR, 'loocv_results_pix2pix.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    print(f"\n  LOOCV CSV saved: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Pix2Pix — PET Tumor Prediction (no time conditioning)')
    parser.add_argument('--loocv', action='store_true')
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = setup_device()
    global MAX_TIME_MIN
    if MAX_TIME_MIN == 0:
        MAX_TIME_MIN = autodetect_max_time(ALL_TUMORS, DATA_ROOT, IMG_SUBFOLDER)
    if args.loocv:
        run_loocv(device)
    else:
        train_one_fold(TRAIN_TUMORS, VAL_TUMORS, TEST_TUMOR, EPOCHS, OUTPUT_DIR, device)
        print(f"\n{'═'*55}\n  Results saved to: {OUTPUT_DIR}")
        for f in ['best_model.pth','training_curves.png','prediction_vs_GT.png',
                  'TAC_comparison.png','metrics_over_time.png']:
            p = os.path.join(OUTPUT_DIR, f)
            print(f"    {'✓' if os.path.exists(p) else '✗'}  {f}")
        print(f"{'═'*55}\n")

if __name__ == '__main__':
    main()
