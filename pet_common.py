"""
pet_common.py
=============
Shared framework for the additional PET frame-prediction models.

The nine original pet_*_paper.py scripts each carry their own copy of the data
loading, metric and plotting code. Duplicating that another four times would be
unmaintainable, so the new models import it from here instead. Everything a
model file needs is re-exported, and the LOOCV driver is built once.

Contract with _pet_worker.py -- every model module must expose:
    ALL_TUMORS, DATA_ROOT, IMG_SUBFOLDER, OUTPUT_DIR, MAX_TIME_MIN
    autodetect_max_time(names, root, subfolder)
    setup_device()
    train_one_fold(train_tumors, val_tumors, test_tumor, epochs, out_dir,
                   device, fold_label='') -> (ssim_s, psnr_s, mae_s) | None
    run_loocv(device)

IMPORTANT -- config propagation. The worker sets attributes on the MODEL
module (mod.OUTPUT_DIR, mod.MAX_TIME_MIN). Because model files do
`from pet_common import *`, they hold copies, not references. Every model's
run_loocv() therefore calls push_config(globals()) first, which copies the
model module's values back into this module so the shared helpers see them.
"""

import csv
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

DATA_ROOT     = os.environ.get('PET_DATA_ROOT',
                               r"C:\PET_DATA\IMAGES")
IMG_SUBFOLDER = os.environ.get('PET_IMG_SUBFOLDER', r"cleaned\resized")
OUTPUT_DIR    = os.environ.get('PET_OUTPUT_DIR',
                               r"C:\PET_DATA\RESULTS")

ALL_TUMORS = [
    't1-r=0.1', 'T1-R=0.2', 'T1-R=0.3', 'T1-R=0.4',
    'T2-R=0.1', 'T2-R=0.2', 't2-r=0.3', 'T2-R=0.4',
    'T3-R=0.2', 'T3-R=0.1',
]

N_CONTEXT    = 10
IMG_SIZE     = 64
MAX_TIME_MIN = 0.0

BATCH_SIZE   = 16
LOOCV_EPOCHS = 400
EPOCHS       = 800
MIN_EPOCHS   = 80
PATIENCE     = 60
N_VAL        = 2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

_CFG_KEYS = ('DATA_ROOT', 'IMG_SUBFOLDER', 'OUTPUT_DIR', 'ALL_TUMORS',
             'N_CONTEXT', 'IMG_SIZE', 'MAX_TIME_MIN', 'BATCH_SIZE',
             'LOOCV_EPOCHS', 'EPOCHS', 'MIN_EPOCHS', 'PATIENCE', 'N_VAL', 'SEED')


def env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Runtime overrides supplied by 03_run_all.py.
LOOCV_EPOCHS = env_int('PET_LOOCV_EPOCHS', LOOCV_EPOCHS)
EPOCHS       = env_int('PET_EPOCHS',       EPOCHS)
MIN_EPOCHS   = env_int('PET_MIN_EPOCHS',   MIN_EPOCHS)
PATIENCE     = env_int('PET_PATIENCE',     PATIENCE)
BATCH_SIZE   = env_int('PET_BATCH_SIZE',   BATCH_SIZE)
N_VAL        = env_int('PET_N_VAL',        N_VAL)


def push_config(model_globals):
    """Copy a model module's config values into this module.

    Called at the start of every run_loocv/train_one_fold so that values the
    worker set on the model module (OUTPUT_DIR, MAX_TIME_MIN) are visible to
    the shared helpers below.
    """
    g = globals()
    for k in _CFG_KEYS:
        if k in model_globals:
            g[k] = model_globals[k]


# ═════════════════════════════════════════════════════════════════════════════
#  DATA UTILITIES  (behaviour identical to the original nine scripts)
# ═════════════════════════════════════════════════════════════════════════════

def extract_timepoint(filename):
    nums = re.findall(r'\d+', Path(filename).stem)
    return int(nums[0]) if nums else None


def get_files(tumor_folder, subfolder):
    # Accept both separators: the config uses a Windows backslash, but the
    # same path must resolve if the code is ever run on Linux.
    cands = [subfolder, str(subfolder).replace('\\', '/'),
             'cleaned/resized', 'resized', 'cleaned', '']
    for sub in cands:
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
    """Largest timepoint across all tumours, with a diagnostic on failure.

    A bare AssertionError here is useless: the caller cannot tell whether the
    root is missing, the tumour folders are named differently, the images sit
    in another subfolder, or the filenames contain no digits. So on failure
    this reports exactly which of those it is.
    """
    max_t = 0
    found_any = 0
    for tumor in tumor_names:
        files = get_files(Path(data_root) / tumor, subfolder)
        if files:
            found_any += 1
        for f in files:
            t = extract_timepoint(Path(f).name)
            if t and t > max_t:
                max_t = t

    if max_t <= 0:
        root = Path(data_root)
        print("\n" + "!" * 70)
        print("  COULD NOT FIND ANY IMAGES")
        print("!" * 70)
        print(f"  DATA_ROOT     : {data_root}")
        print(f"  exists?       : {root.is_dir()}")
        print(f"  IMG_SUBFOLDER : {subfolder}")
        if root.is_dir():
            subs = sorted(d.name for d in root.iterdir() if d.is_dir())
            print(f"  folders here  : {subs[:15] if subs else '(none)'}")
            print(f"  expected      : {list(tumor_names)[:4]} ...")
            missing = [t for t in tumor_names
                       if not (root / t).is_dir()
                       and not any(d.lower() == t.lower() for d in subs)]
            if missing:
                print(f"  MISSING       : {missing}")
            else:
                probe = root / tumor_names[0]
                inner = sorted(d.name for d in probe.iterdir()) if probe.is_dir() else []
                print(f"  tumour folders exist, but no usable images inside.")
                print(f"  contents of {tumor_names[0]}: {inner[:10]}")
                print("  -> either IMG_SUBFOLDER is wrong, or the filenames")
                print("     contain no digits (timepoints are parsed from them).")
        else:
            print("  The DATA_ROOT folder does not exist. If you moved your")
            print("  images, re-point the scripts at the new location:")
            print('      python 00_set_paths.py --data "C:\\new\\path\\to\\IMAGES"')
        print("!" * 70 + "\n")
        raise FileNotFoundError(
            f"No timepoints found under {data_root} (subfolder={subfolder!r}). "
            f"{found_any}/{len(tumor_names)} tumour folders yielded images.")

    print(f"Auto-detected MAX_TIME_MIN: {max_t} min")
    return float(max_t)


def load_frame(path):
    img = Image.open(path).convert('RGB')
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
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
        print("No GPU found -- running on CPU")
    return dev


class PETPairDataset(Dataset):
    """(context stack, context times, target time, target frame) samples.

    All frames are decoded once at construction and held in RAM: ten tumours
    of ~140 64x64 RGB frames is about 70 MB, so re-decoding every epoch would
    be pure waste.
    """

    def __init__(self, tumor_names, data_root, subfolder,
                 n_context, max_time_min, augment=False, label=''):
        self.n_context = n_context
        self.max_time = max_time_min
        self.augment = augment
        self.samples = []

        tag = f' [{label}]' if label else ''
        print(f"\n{'Tumor':<20} {'Total':>7} {'Pairs':>8}{tag}")
        print("-" * 38)
        total_pairs = 0

        for tumor in tumor_names:
            files = get_files(Path(data_root) / tumor, subfolder)
            if not files:
                print(f"  {tumor:<18} NOT FOUND")
                continue
            if len(files) <= n_context:
                print(f"  {tumor:<18} too few ({len(files)})")
                continue

            frames, times = [], []
            for path in files:
                frames.append(load_frame(path))
                times.append(extract_timepoint(Path(path).name) / max_time_min)

            frames = torch.stack(frames)
            times = torch.tensor(times, dtype=torch.float32)
            ctx_frames = frames[:n_context]
            ctx_times = times[:n_context]

            n_pairs = 0
            for i in range(n_context, len(files)):
                self.samples.append({
                    'ctx': ctx_frames, 'ctx_t': ctx_times,
                    't_norm': times[i], 'target': frames[i], 'tumor': tumor,
                })
                n_pairs += 1
            total_pairs += n_pairs
            print(f"  {tumor:<18} {len(files):>7} {n_pairs:>8}")

        print("-" * 38)
        print(f"  {'TOTAL':<18} {'':>7} {total_pairs:>8} pairs\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ctx, tgt = s['ctx'].clone(), s['target'].clone()
        if self.augment:
            if random.random() > 0.5:
                ctx = torch.flip(ctx, dims=[3]); tgt = torch.flip(tgt, dims=[2])
            if random.random() > 0.5:
                ctx = torch.flip(ctx, dims=[2]); tgt = torch.flip(tgt, dims=[1])
        return ctx, s['ctx_t'], s['t_norm'], tgt


# ═════════════════════════════════════════════════════════════════════════════
#  BUILDING BLOCKS
# ═════════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmbedding(nn.Module):
    """Fourier features of a scalar time, then a small MLP."""

    def __init__(self, emb_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

    def forward(self, t):
        half = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=-1))


def _gaussian_kernel(kernel_size=11, sigma=1.5, channels=3):
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)


def ssim_loss_fn(pred, target, kernel_size=11, sigma=1.5):
    """1 - SSIM, computed on [0,1] tensors. Differentiable."""
    c = pred.shape[1]
    kernel = _gaussian_kernel(kernel_size, sigma, c).to(pred.device)
    pad = kernel_size // 2
    mu_p = F.conv2d(pred, kernel, padding=pad, groups=c)
    mu_t = F.conv2d(target, kernel, padding=pad, groups=c)
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
    s_p = F.conv2d(pred * pred, kernel, padding=pad, groups=c) - mu_p2
    s_t = F.conv2d(target * target, kernel, padding=pad, groups=c) - mu_t2
    s_pt = F.conv2d(pred * target, kernel, padding=pad, groups=c) - mu_pt
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_pt + c1) * (2 * s_pt + c2)) / \
           ((mu_p2 + mu_t2 + c1) * (s_p + s_t + c2))
    return 1.0 - ssim.mean()


def compute_metrics(pred_arr, gt_arr):
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    ssim_s, psnr_s, mae_s = [], [], []
    for p, g in zip(pred_arr, gt_arr):
        ssim_s.append(sk_ssim(g, p, data_range=1.0, channel_axis=2))
        psnr_s.append(sk_psnr(g, p, data_range=1.0))
        mae_s.append(float(np.abs(g - p).mean()))
    return np.array(ssim_s), np.array(psnr_s), np.array(mae_s)


# =============================================================================
#  COLORMAP DECODING  --  metrics in activity space
# =============================================================================
#
# The frames are jet-colormapped renderings, not scalar activity maps. Jet is a
# nonlinear, NON-MONOTONIC map from activity to RGB: as activity rises the blue
# channel falls, then green rises and falls, then red rises. Two consequences:
#
#   1. Per-channel SSIM/PSNR/MAE on RGB do not measure image quality in any
#      physically meaningful sense. A prediction that is structurally correct
#      but slightly off in intensity can land in a completely different hue and
#      be scored as catastrophically wrong. This is exactly what happened to
#      the cVAE: visually sound output, SSIM 0.29.
#   2. Per-voxel kinetic fitting in RGB space is invalid outright -- tracer
#      washout is monotonic in activity but NOT monotonic in R, G or B.
#
# The fix is to invert the colormap: match each RGB pixel to the nearest entry
# of the colormap lookup table and recover the scalar it encodes. Inversion is
# approximate (jet is not perfectly injective, and compression blurs it) but
# far closer to the truth than treating RGB as three intensity channels.
#
# PET_METRIC_SPACE selects what train_one_fold returns:
#     'rgb'      -- as before; directly comparable with the completed run
#     'activity' -- decoded scalar; the methodologically correct choice
# Both are ALWAYS computed and saved, so switching never requires retraining.

COLORMAP_NAME = os.environ.get('PET_COLORMAP', 'jet')
METRIC_SPACE = os.environ.get('PET_METRIC_SPACE', 'rgb').lower()

_LUT_CACHE = {}


def colormap_lut(name=None, n=512):
    """(n,3) float32 lookup table for the named matplotlib colormap."""
    name = name or COLORMAP_NAME
    key = (name, n)
    if key not in _LUT_CACHE:
        try:
            cm = matplotlib.colormaps[name]
        except Exception:
            cm = plt.get_cmap(name)
        _LUT_CACHE[key] = np.asarray(
            [cm(i / (n - 1))[:3] for i in range(n)], dtype=np.float32)
    return _LUT_CACHE[key]


def rgb_to_activity(arr, name=None, n=512):
    """Invert a colormap. (..., H, W, 3) in [0,1] -> (..., H, W) in [0,1].

    Nearest-neighbour match against the LUT, in chunks: the naive broadcast
    (P x n x 3) would allocate gigabytes for a whole tumour.
    """
    lut = colormap_lut(name, n)
    shape = arr.shape[:-1]
    flat = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
    out = np.empty(flat.shape[0], dtype=np.float32)
    chunk = 32768
    for i in range(0, flat.shape[0], chunk):
        blk = flat[i:i + chunk]
        d = ((blk[:, None, :] - lut[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = d.argmin(1).astype(np.float32) / (n - 1)
    return out.reshape(shape)


def activity_to_rgb(act, name=None, n=512):
    """Forward direction: (..., H, W) scalar in [0,1] -> (..., H, W, 3) RGB.

    Needed by the kinetic baseline, which fits in activity space and must
    re-encode its predictions so RGB metrics stay comparable with the other
    models.
    """
    lut = colormap_lut(name, n)
    idx = np.clip((np.asarray(act, dtype=np.float32) * (n - 1)).round(), 0,
                  n - 1).astype(np.int32)
    return lut[idx]


def compute_metrics_activity(pred_arr, gt_arr):
    """SSIM / PSNR / MAE on decoded scalar activity rather than RGB."""
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    p_act = rgb_to_activity(np.asarray(pred_arr))
    g_act = rgb_to_activity(np.asarray(gt_arr))
    ssim_s, psnr_s, mae_s = [], [], []
    for p, g in zip(p_act, g_act):
        ssim_s.append(sk_ssim(g, p, data_range=1.0))
        psnr_s.append(sk_psnr(g, p, data_range=1.0))
        mae_s.append(float(np.abs(g - p).mean()))
    return np.array(ssim_s), np.array(psnr_s), np.array(mae_s)


def finalize_fold(pred_arr, gt_arr, tgt_times_min, out_dir, model_name=''):
    """Compute both metric spaces, save everything, return the selected one.

    Predictions are written to disk so metrics can be recomputed later WITHOUT
    retraining -- the single most useful artefact to have when a reviewer asks
    for a different metric.
    """
    rgb = compute_metrics(pred_arr, gt_arr)
    try:
        act = compute_metrics_activity(pred_arr, gt_arr)
    except Exception as exc:
        print(f"  [warn] activity-space metrics failed: {exc}")
        act = rgb

    try:
        np.savez_compressed(
            Path(out_dir) / 'fold_outputs.npz',
            pred=np.asarray(pred_arr, dtype=np.float16),
            gt=np.asarray(gt_arr, dtype=np.float16),
            times_min=np.asarray(tgt_times_min),
            ssim_rgb=rgb[0], psnr_rgb=rgb[1], mae_rgb=rgb[2],
            ssim_act=act[0], psnr_act=act[1], mae_act=act[2],
        )
    except Exception as exc:
        print(f"  [warn] could not write fold_outputs.npz: {exc}")
    print(f"  metrics RGB      : SSIM {rgb[0].mean():.4f} | "
          f"PSNR {rgb[1].mean():.2f} | MAE {rgb[2].mean():.4f}")
    print(f"  metrics ACTIVITY : SSIM {act[0].mean():.4f} | "
          f"PSNR {act[1].mean():.2f} | MAE {act[2].mean():.4f}")
    return act if METRIC_SPACE == 'activity' else rgb



def _safe_save_fig(fig, path, dpi=130):
    """Save a figure without ever letting a filesystem hiccup kill a run.

    Observed failure: OSError [Errno 22] Invalid argument when writing a PNG
    into a OneDrive-synced Desktop folder. The sync client intercepts file
    creation and can reject the handle. Losing an entire trained fold because
    a decorative plot could not be written is unacceptable, so every figure
    save is best-effort: it retries once via a temporary file, then gives up
    with a warning and lets training continue.
    """
    import shutil, tempfile
    try:
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return True
    except Exception as exc:
        try:
            tmp = Path(tempfile.gettempdir()) / f"_pet_{os.getpid()}_{Path(path).name}"
            fig.savefig(tmp, dpi=dpi)
            shutil.move(str(tmp), str(path))
            plt.close(fig)
            return True
        except Exception:
            plt.close(fig)
            print(f"  [warn] could not write {Path(path).name}: "
                  f"{type(exc).__name__}: {exc}")
            print(f"  [warn] continuing anyway -- metrics are unaffected")
            return False



# ═════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════


def _safe_plot(fn):
    """Never let a cosmetic figure kill a fold.

    A diffusion fold that trained for 683 epochs was destroyed by an
    OSError raised inside savefig. Metrics and predictions are already
    persisted by finalize_fold before any plotting happens, so a failed
    figure costs nothing -- but an uncaught exception costs hours. Every
    plotting helper is wrapped.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as exc:
            print(f"  [warn] {fn.__name__} failed ({type(exc).__name__}: {exc})"
                  f" -- metrics are unaffected, continuing")
            try:
                plt.close('all')
            except Exception:
                pass
            return None
    return wrapper


def _finite(arr, fill=0.0):
    """Replace inf/NaN so matplotlib cannot compute an invalid canvas size.

    A single infinite PSNR (which happens when a predicted frame matches the
    ground truth exactly) makes the axis limits infinite; tight_layout then
    produces a NaN figure size and savefig fails with a confusing Errno 22.
    """
    a = np.asarray(arr, dtype=np.float64).copy()
    bad = ~np.isfinite(a)
    if bad.any():
        good = a[~bad]
        a[bad] = good.max() if good.size else fill
    return a


@_safe_plot
def save_training_curves(history, path):
    """Plot the curves AND write the raw numbers next to them.

    A PNG cannot be aggregated. Dumping the same history to CSV lets
    05_overfitting_report.py quantify the train/validation gap across every
    fold and model instead of eyeballing 130 plots.
    """
    try:
        import csv as _csv
        keys_all = [k for k in history if history[k]]
        n = max((len(history[k]) for k in keys_all), default=0)
        if n:
            csv_path = Path(path).with_name('training_history.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
                w = _csv.writer(fh)
                w.writerow(['epoch'] + keys_all)
                for i in range(n):
                    w.writerow([i + 1] + [
                        history[k][i] if i < len(history[k]) else ''
                        for k in keys_all])
    except Exception as exc:
        print(f"  [warn] could not write training_history.csv: {exc}")

    keys = [k for k in history if len(history[k]) > 1]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in keys:
        ax.plot(_finite(history[k]), lw=1.8, label=k)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); _safe_save_fig(fig, path)


@_safe_plot
def save_comparison_grid(pred_arr, gt_arr, tgt_times, tumor_name, path,
                         n_show=6):
    n = min(n_show, len(pred_arr))
    if n == 0:
        return
    idx = np.linspace(0, len(pred_arr) - 1, n).astype(int)
    fig, axes = plt.subplots(3, n, figsize=(2.1 * n, 6.6))
    if n == 1:
        axes = axes.reshape(3, 1)
    for j, i in enumerate(idx):
        axes[0, j].imshow(gt_arr[i]); axes[0, j].set_title(
            f"t={tgt_times[i]}min", fontsize=8)
        axes[1, j].imshow(pred_arr[i])
        axes[2, j].imshow(np.abs(gt_arr[i] - pred_arr[i]), cmap='hot',
                          vmin=0, vmax=0.5)
        for r in range(3):
            axes[r, j].axis('off')
    axes[0, 0].set_ylabel('GT'); axes[1, 0].set_ylabel('Pred')
    fig.suptitle(f"{tumor_name}  (top: ground truth, mid: prediction, "
                 f"bottom: |error|)", fontsize=10)
    fig.tight_layout(); _safe_save_fig(fig, path)


@_safe_plot
def save_metrics_plot(ssim_s, psnr_s, mae_s, times_min, path):
    ssim_s, psnr_s, mae_s = _finite(ssim_s), _finite(psnr_s), _finite(mae_s)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, vals, name in zip(axes, (ssim_s, psnr_s, mae_s),
                              ('SSIM', 'PSNR (dB)', 'MAE')):
        ax.plot(times_min[:len(vals)], vals, 'o-', ms=3, lw=1.4)
        ax.set_xlabel('Time (min)'); ax.set_ylabel(name); ax.grid(alpha=0.3)
    fig.tight_layout(); _safe_save_fig(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
#  LOOCV DRIVER
# ═════════════════════════════════════════════════════════════════════════════

def fold_split(all_tumors, test_t, n_val=2, seed=SEED):
    """Deterministic LOOCV split with a rotating validation set.

    A fixed seeded permutation is formed once; the validation tumours for the
    fold testing on `test_t` are the next n_val entries cyclically after it.
    Every tumour therefore serves as validation in exactly n_val folds, which
    a naive `remaining[:2]` does not achieve.
    """
    order = list(all_tumors)
    random.Random(seed).shuffle(order)
    n = len(order)
    i = order.index(test_t)
    rest = [order[(i + k) % n] for k in range(1, n)]
    return rest[:n_val], rest[n_val:]


def build_loaders(train_tumors, val_tumors, device, augment=True):
    tr = PETPairDataset(train_tumors, DATA_ROOT, IMG_SUBFOLDER, N_CONTEXT,
                        MAX_TIME_MIN, augment=augment, label='train')
    va = PETPairDataset(val_tumors, DATA_ROOT, IMG_SUBFOLDER, N_CONTEXT,
                        MAX_TIME_MIN, augment=False, label='val')
    pin = device.type == 'cuda'
    return (DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, pin_memory=pin, drop_last=False),
            DataLoader(va, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=0, pin_memory=pin),
            len(tr), len(va))


def load_test_tumor(test_tumor, device):
    """Return (ctx[1,T,C,H,W], ctx_times_min, tgt_times_min, gt_arr) or None."""
    files = get_files(Path(DATA_ROOT) / test_tumor, IMG_SUBFOLDER)
    if not files or len(files) <= N_CONTEXT:
        return None
    frames = torch.stack([load_frame(p) for p in files])
    times = torch.tensor(
        [extract_timepoint(Path(p).name) / MAX_TIME_MIN for p in files],
        dtype=torch.float32)
    ctx = frames[:N_CONTEXT].unsqueeze(0).to(device)
    ctx_t = times[:N_CONTEXT].unsqueeze(0).to(device)
    ctx_times_min = [int(extract_timepoint(Path(p).name))
                     for p in files[:N_CONTEXT]]
    tgt_times_min = [int(extract_timepoint(Path(p).name))
                     for p in files[N_CONTEXT:]]
    tgt_norm = times[N_CONTEXT:].to(device)
    gt_arr = np.array([denorm(f).permute(1, 2, 0).numpy()
                       for f in frames[N_CONTEXT:]])
    return ctx, ctx_t, ctx_times_min, tgt_times_min, tgt_norm, gt_arr


def make_run_loocv(model_name, train_one_fold_fn, model_globals):
    """Return a run_loocv(device) closure for a model module."""

    def run_loocv(device=None):
        push_config(model_globals)
        if device is None:
            device = setup_device()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        loocv_dir = Path(OUTPUT_DIR) / 'loocv'
        loocv_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print(f"  LEAVE-ONE-OUT CROSS-VALIDATION -- {model_name}")
        print("=" * 70)

        results = []
        for k, test_t in enumerate(ALL_TUMORS, 1):
            val_t, train_t = fold_split(ALL_TUMORS, test_t, n_val=N_VAL,
                                        seed=SEED)
            print("\n" + "=" * 70)
            print(f"  Fold {k}/{len(ALL_TUMORS)}: test={test_t}  val={val_t}")
            print("=" * 70)

            fold_dir = loocv_dir / test_t.replace('=', '').replace('.', 'p')
            fold_dir.mkdir(parents=True, exist_ok=True)

            # Resolve the function from the model module's globals at CALL
            # time, not at closure-creation time. _pet_worker.py wraps
            # mod.train_one_fold to capture per-frame metrics; binding the
            # original function object here would silently bypass that wrapper
            # and the per-frame CSV would come out empty.
            fn = model_globals.get('train_one_fold', train_one_fold_fn)
            out = fn(train_t, val_t, test_t, LOOCV_EPOCHS, str(fold_dir),
                     device, fold_label=f"LOOCV:{test_t}")
            if out is None:
                print(f"  [warn] fold {test_t} produced no result")
                continue
            ssim_s, psnr_s, mae_s = out
            results.append({
                'test_tumor': test_t,
                'n_frames': len(ssim_s),
                'ssim_mean': float(np.mean(ssim_s)),
                'ssim_std': float(np.std(ssim_s)),
                'psnr_mean': float(np.mean(psnr_s)),
                'psnr_std': float(np.std(psnr_s)),
                'mae_mean': float(np.mean(mae_s)),
                'mae_std': float(np.std(mae_s)),
            })
            print(f"  -> SSIM {np.mean(ssim_s):.4f} | "
                  f"PSNR {np.mean(psnr_s):.2f} dB | MAE {np.mean(mae_s):.4f}")

        if not results:
            print("\n[FAIL] no folds completed")
            return

        csv_path = Path(OUTPUT_DIR) / f'loocv_results_{model_name}.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)

        ss = np.array([r['ssim_mean'] for r in results])
        pp = np.array([r['psnr_mean'] for r in results])
        mm = np.array([r['mae_mean'] for r in results])

        print("\n" + "=" * 70)
        print(f"  LOOCV SUMMARY -- {model_name}  ({len(results)} folds)")
        print("=" * 70)
        print(f"  {'Tumor':<20}{'SSIM':>10}{'PSNR':>10}{'MAE':>10}")
        print("  " + "-" * 50)
        for r in results:
            print(f"  {r['test_tumor']:<20}{r['ssim_mean']:>10.4f}"
                  f"{r['psnr_mean']:>10.2f}{r['mae_mean']:>10.4f}")
        print("  " + "-" * 50)
        print(f"  {'MEAN +/- SD':<20}{ss.mean():>10.4f}{pp.mean():>10.2f}"
              f"{mm.mean():>10.4f}")
        print(f"  {'':<20}{ss.std():>10.4f}{pp.std():>10.2f}{mm.std():>10.4f}")
        print(f"\n  results -> {csv_path}")

    return run_loocv


class EarlyStopper:
    """Min-epochs floor plus patience, tracking the best validation loss."""

    def __init__(self, patience=PATIENCE, min_epochs=MIN_EPOCHS):
        self.patience = patience
        self.min_epochs = min_epochs
        self.best = float('inf')
        self.best_state = None
        self.counter = 0
        self.epoch = 0

    def step(self, val_loss, model):
        self.epoch += 1
        if val_loss < self.best - 1e-6:
            self.best = val_loss
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
        return (self.epoch >= self.min_epochs
                and self.counter >= self.patience)

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model
