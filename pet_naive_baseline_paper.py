"""
pet_naive_baseline_paper.py
===========================
Naive Baselines for PET Tumor Simulation Completion
— Paper-ready version — matches cGAN style exactly —

Role in paper : FLOOR BASELINES — trivial predictors that require no learning.
                All learned models must beat these. Grounds the comparison table.

Two baselines computed:
  1. Last-frame repeat  : predict the last context frame for every future timepoint
  2. Mean-frame         : predict the pixel-wise mean of all context frames

No training required — runs inference only.

Split         : same 7/2/1 as all other models (test on T3-R=0.1)
                LOOCV also supported for consistent reporting.

Setup:
  pip install numpy pillow matplotlib tqdm scikit-image scipy

Run:
  python pet_naive_baseline_paper.py
  python pet_naive_baseline_paper.py --loocv
"""

import os
import re
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy import stats
from torchvision import transforms
import torch

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

SEED = 42
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
# ─────────────────────────────────────────────────────────────────────────────


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

def load_frame_np(path):
    """Load as H×W×3 float32 in [0,1]."""
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0


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
#  INFERENCE — both baselines at once
# ═════════════════════════════════════════════════════════════════════════════

def run_naive(test_tumor):
    """
    Returns (last_frame_results, mean_frame_results, ctx_times_min, tgt_times_min)
    Each result is (ssim_arr, psnr_arr, mae_arr).
    """
    test_files = get_files(Path(DATA_ROOT)/test_tumor, IMG_SUBFOLDER)
    if not test_files: return None

    ctx_frames = [load_frame_np(p) for p in test_files[:N_CONTEXT]]
    gt_frames  = [load_frame_np(p) for p in test_files[N_CONTEXT:]]

    ctx_times_min = [int(extract_timepoint(Path(p).name)) for p in test_files[:N_CONTEXT]]
    tgt_times_min = [int(extract_timepoint(Path(p).name)) for p in test_files[N_CONTEXT:]]

    last_frame  = ctx_frames[-1]                              # H×W×3
    mean_frame  = np.mean(ctx_frames, axis=0).clip(0, 1)     # H×W×3

    T = len(gt_frames)
    last_preds = np.stack([last_frame] * T)   # (T, H, W, 3)
    mean_preds = np.stack([mean_frame] * T)   # (T, H, W, 3)
    gt_arr     = np.stack(gt_frames)          # (T, H, W, 3)

    return (compute_metrics(last_preds, gt_arr),
            compute_metrics(mean_preds, gt_arr),
            ctx_frames, ctx_times_min, tgt_times_min,
            last_preds, mean_preds, gt_arr)


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def save_comparison_grid(ctx_frames, pred_frames, gt_frames,
                         ctx_times_min, tgt_times_min,
                         tumor_name, method_name, path, n_show=10):
    T = len(pred_frames); indices = np.linspace(0, T-1, min(n_show,T), dtype=int)
    n_cols = len(indices)
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols*2.2, 7))
    if n_cols == 1: axes = axes[:, None]
    for col, idx in enumerate(indices):
        t_min = int(tgt_times_min[idx])
        if col < len(ctx_times_min):
            axes[0,col].imshow(ctx_frames[col].clip(0,1))
            axes[0,col].set_title(f"ctx\n{int(ctx_times_min[col])}min", fontsize=7)
        else: axes[0,col].axis('off')
        axes[1,col].imshow(pred_frames[idx].clip(0,1))
        axes[1,col].set_title(f"pred\n{t_min}min", fontsize=7)
        axes[2,col].imshow(gt_frames[idx].clip(0,1))
        axes[2,col].set_title(f"GT\n{t_min}min", fontsize=7)
    for ax in axes.flat: ax.axis('off')
    axes[0,0].set_ylabel("Context", fontsize=9)
    axes[1,0].set_ylabel(method_name, fontsize=9)
    axes[2,0].set_ylabel("Ground Truth", fontsize=9)
    plt.suptitle(f"{method_name} — {tumor_name}  |  First {N_CONTEXT} frames given", fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

def save_metrics_plot(ssim_s, psnr_s, mae_s, times_min, method_name, path):
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for ax, scores, lbl, col in zip(axes, [ssim_s,psnr_s,mae_s],
                                    ['SSIM ↑','PSNR dB ↑','MAE ↓'],
                                    ['steelblue','seagreen','tomato']):
        ax.plot(times_min, scores, color=col, lw=2)
        ax.set_xlabel('Time (min)'); ax.set_title(f"{method_name} — {lbl}"); ax.grid(alpha=0.3)
        ax.axhline(np.mean(scores), color='black', linestyle='--', alpha=0.5,
                   label=f'mean={np.mean(scores):.3f}'); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); print(f"  Saved: {path}")

def save_tac(ctx_frames, ctx_t, last_mean, mean_mean, gt_mean, tgt_t, tumor_name, path):
    ctx_mean = [f.mean() for f in ctx_frames]
    plt.figure(figsize=(11,4))
    plt.plot(ctx_t, ctx_mean,  'o-', color='gray',      lw=2, ms=5, label=f'Context ({N_CONTEXT} frames)', zorder=5)
    plt.plot(tgt_t, gt_mean,   '-',  color='steelblue', lw=2.5, label='Ground Truth', zorder=3)
    plt.plot(tgt_t, last_mean, '--', color='tomato',    lw=2,   label='Last-frame repeat', zorder=4)
    plt.plot(tgt_t, mean_mean, ':',  color='seagreen',  lw=2,   label='Mean-frame', zorder=4)
    plt.axvline(ctx_t[-1], color='black', lw=1.2, linestyle=':', alpha=0.7, label='Prediction boundary')
    plt.xlabel('Time (minutes)', fontsize=11); plt.ylabel('Mean Pixel Intensity', fontsize=11)
    plt.title(f'Time-Activity Curve — {tumor_name}', fontsize=12)
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close(); print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  EVALUATE ONE TUMOR
# ═════════════════════════════════════════════════════════════════════════════

def _print_results(name, ssim_s, psnr_s, mae_s, tgt_times_min):
    SPLIT_MIN = 2000
    early_mask = np.array(tgt_times_min) < SPLIT_MIN
    late_mask  = ~early_mask
    print(f"\n  ── {name} ──────────────────────────────────────")
    print(f"  Mean SSIM : {ssim_s.mean():.4f} ± {ssim_s.std():.4f}")
    print(f"  Mean PSNR : {psnr_s.mean():.2f} ± {psnr_s.std():.2f} dB")
    print(f"  Mean MAE  : {mae_s.mean():.4f} ± {mae_s.std():.4f}")
    if early_mask.sum() > 0 and late_mask.sum() > 0:
        e_ssim = ssim_s[early_mask]; l_ssim = ssim_s[late_mask]
        print(f"  Early (<{SPLIT_MIN}min, n={early_mask.sum()}):  SSIM={e_ssim.mean():.4f} ± {e_ssim.std():.4f}")
        print(f"  Late  (≥{SPLIT_MIN}min, n={late_mask.sum()}):   SSIM={l_ssim.mean():.4f} ± {l_ssim.std():.4f}")
        stat, p_val = stats.mannwhitneyu(e_ssim, l_ssim, alternative='greater')
        print(f"  Mann-Whitney U: p={p_val:.4f}  "
              f"{'✓ significant' if p_val<0.05 else '✗ not significant'}")


def evaluate_tumor(tumor, out_dir, save_plots=True):
    result = run_naive(tumor)
    if result is None:
        print(f"  {tumor}: ❌ not found"); return None, None

    (last_ssim, last_psnr, last_mae), \
    (mean_ssim, mean_psnr, mean_mae), \
    ctx_frames, ctx_times_min, tgt_times_min, \
    last_preds, mean_preds, gt_arr = result

    _print_results("Last-frame repeat", last_ssim, last_psnr, last_mae, tgt_times_min)
    _print_results("Mean-frame",        mean_ssim, mean_psnr, mean_mae, tgt_times_min)

    if save_plots:
        os.makedirs(out_dir, exist_ok=True)
        save_comparison_grid(ctx_frames, last_preds, gt_arr, ctx_times_min, tgt_times_min,
                             tumor, "Last-frame repeat",
                             os.path.join(out_dir, 'last_frame_vs_GT.png'))
        save_comparison_grid(ctx_frames, mean_preds, gt_arr, ctx_times_min, tgt_times_min,
                             tumor, "Mean-frame",
                             os.path.join(out_dir, 'mean_frame_vs_GT.png'))
        save_metrics_plot(last_ssim, last_psnr, last_mae, tgt_times_min,
                          "Last-frame", os.path.join(out_dir, 'last_frame_metrics.png'))
        save_metrics_plot(mean_ssim, mean_psnr, mean_mae, tgt_times_min,
                          "Mean-frame", os.path.join(out_dir, 'mean_frame_metrics.png'))
        save_tac(ctx_frames, ctx_times_min,
                 last_preds.mean(axis=(1,2,3)),
                 mean_preds.mean(axis=(1,2,3)),
                 gt_arr.mean(axis=(1,2,3)),
                 tgt_times_min, tumor,
                 os.path.join(out_dir, 'TAC_comparison.png'))

    last_res = {'ssim_mean': last_ssim.mean(), 'ssim_std': last_ssim.std(),
                'psnr_mean': last_psnr.mean(), 'psnr_std': last_psnr.std(),
                'mae_mean':  last_mae.mean(),  'mae_std':  last_mae.std()}
    mean_res = {'ssim_mean': mean_ssim.mean(), 'ssim_std': mean_ssim.std(),
                'psnr_mean': mean_psnr.mean(), 'psnr_std': mean_psnr.std(),
                'mae_mean':  mean_mae.mean(),  'mae_std':  mean_mae.std()}
    return last_res, mean_res


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN RUN  (single test tumor)
# ═════════════════════════════════════════════════════════════════════════════

def run_single(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'═'*65}")
    print(f"  Naive Baselines — Test tumor: {TEST_TUMOR}")
    print(f"{'═'*65}")
    last_res, mean_res = evaluate_tumor(TEST_TUMOR, out_dir, save_plots=True)
    if last_res is None: return

    print(f"\n{'═'*65}")
    print(f"  SUMMARY")
    print(f"{'═'*65}")
    print(f"  {'Method':<25} {'SSIM':>8} {'PSNR':>10} {'MAE':>10}")
    print(f"  {'─'*55}")
    for name, res in [("Last-frame repeat", last_res), ("Mean-frame", mean_res)]:
        print(f"  {name:<25} {res['ssim_mean']:.4f}   "
              f"{res['psnr_mean']:>6.2f} dB   {res['mae_mean']:.4f}")

    csv_path = os.path.join(out_dir, 'naive_baseline_results.csv')
    rows = [{'method': 'last_frame', 'test_tumor': TEST_TUMOR, **last_res},
            {'method': 'mean_frame', 'test_tumor': TEST_TUMOR, **mean_res}]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"\n  Results saved: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  LOOCV
# ═════════════════════════════════════════════════════════════════════════════

def run_loocv():
    print("\n" + "═"*65)
    print("  LEAVE-ONE-OUT CROSS-VALIDATION — Naive Baselines")
    print("═"*65)
    last_results, mean_results = [], []

    for test_t in ALL_TUMORS:
        fold_dir = os.path.join(OUTPUT_DIR, 'loocv', test_t.replace('=',''))
        print(f"\n{'─'*55}\n  Fold: test={test_t}\n{'─'*55}")
        last_res, mean_res = evaluate_tumor(test_t, fold_dir, save_plots=False)
        if last_res is not None:
            last_results.append({'test_tumor': test_t, **last_res})
            mean_results.append({'test_tumor': test_t, **mean_res})

    def _print_loocv(name, results):
        print(f"\n{'═'*65}\n  LOOCV — {name}\n{'═'*65}")
        ssims = [r['ssim_mean'] for r in results]
        psnrs = [r['psnr_mean'] for r in results]
        maes  = [r['mae_mean']  for r in results]
        print(f"  {'Tumor':<20} {'SSIM':>8} {'PSNR':>10} {'MAE':>10}\n  {'─'*52}")
        for r in results:
            print(f"  {r['test_tumor']:<20} {r['ssim_mean']:.4f}   "
                  f"{r['psnr_mean']:>6.2f} dB   {r['mae_mean']:.4f}")
        print(f"  {'─'*52}\n  {'MEAN ± STD':<20} "
              f"{np.mean(ssims):.4f}±{np.std(ssims):.4f}   "
              f"{np.mean(psnrs):>5.2f}±{np.std(psnrs):.2f} dB   "
              f"{np.mean(maes):.4f}±{np.std(maes):.4f}")
        tag = name.lower().replace(' ','_').replace('-','')
        paper_line = (f"In LOOCV across all {len(results)} tumors, {name} achieved "
                      f"SSIM {np.mean(ssims):.3f}±{np.std(ssims):.3f}, "
                      f"PSNR {np.mean(psnrs):.2f}±{np.std(psnrs):.2f} dB, "
                      f"MAE {np.mean(maes):.4f}±{np.std(maes):.4f}.")
        print(f"\n  Paper sentence:\n  \"{paper_line}\"")
        csv_path = os.path.join(OUTPUT_DIR, f'loocv_results_{tag}.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
        print(f"  LOOCV CSV saved: {csv_path}")

    _print_loocv("Last-frame repeat", last_results)
    _print_loocv("Mean-frame",        mean_results)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Naive Baselines — PET Tumor Prediction')
    parser.add_argument('--loocv', action='store_true',
                        help='Run Leave-One-Out Cross-Validation')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    global MAX_TIME_MIN
    if MAX_TIME_MIN == 0:
        MAX_TIME_MIN = autodetect_max_time(ALL_TUMORS, DATA_ROOT, IMG_SUBFOLDER)

    if args.loocv:
        run_loocv()
    else:
        run_single(OUTPUT_DIR)

if __name__ == '__main__':
    main()
