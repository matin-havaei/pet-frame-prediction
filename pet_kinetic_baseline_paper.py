"""
pet_kinetic_baseline_paper.py   (CORRECTED)
===========================================
Classical per-voxel kinetic curve fitting. No training.

WHAT WAS WRONG WITH THE FIRST VERSION
It scored SSIM 0.237, losing to last-frame-repeat in 10/10 folds. A physically
motivated fit should not lose to doing nothing. Two independent bugs:

 1. FITTING IN RGB SPACE. The frames are jet-colormapped. Tracer washout is
    monotonic in ACTIVITY but not in R, G or B -- as activity falls, blue rises
    while red falls. Fitting y = a*exp(-k*t) to the red channel and to the blue
    channel yields contradictory decay constants, and the recombined RGB triple
    need not lie on the colormap at all. This version decodes the colormap to
    scalar activity, fits there, and re-encodes for reporting.

 2. UNCONSTRAINED 13-FOLD EXTRAPOLATION. The context is the first 10 frames of
    ~130, so the fit is estimated over roughly the first 8% of the timeline and
    then extrapolated to 100%. Small errors in k are amplified enormously:
    exp(-k) with k off by 1 changes the prediction by a factor of e. This
    version clamps k to a plausible band and shrinks each voxel's estimate
    toward the image-wide median decay in proportion to how poorly that voxel
    fits (James-Stein style shrinkage). Noisy voxels inherit the global decay
    rather than extrapolating their own noise.

This is now a fair baseline. If the deep models still beat it, that comparison
carries real weight; if they do not, that is the paper's finding.

# --- PATCHED-BY-02_patch_scripts --- (already LOOCV-correct; patcher skips)
"""

import os
from pathlib import Path

import numpy as np

from pet_common import (
    ALL_TUMORS, DATA_ROOT, IMG_SUBFOLDER, OUTPUT_DIR, MAX_TIME_MIN,
    N_CONTEXT, IMG_SIZE, SEED, N_VAL,
    autodetect_max_time, setup_device, denorm,
    rgb_to_activity, activity_to_rgb, finalize_fold,
    load_test_tumor, make_run_loocv, push_config,
    save_comparison_grid, save_metrics_plot,
)
import pet_common as pc

MODEL_NAME = 'kinetic'

EPS = 1e-6
K_MIN, K_MAX = -0.5, 6.0     # plausible decay band on normalised time
SHRINK_STRENGTH = 1.0        # higher = pull harder toward the global decay


def weighted_loglinear(y, t):
    """Amplitude-weighted least squares of log y = log a - k t, per voxel.

    y : (T, P) activity in [0,1];  t : (T,) normalised time.
    Weighting by y matters: unweighted log-space regression is dominated by
    near-zero background voxels, whose log values are huge and noisy.
    """
    yc = np.clip(y, EPS, None)
    logy = np.log(yc)
    w = yc
    tt = t[:, None]

    sw = w.sum(0)
    swt = (w * tt).sum(0)
    swtt = (w * tt * tt).sum(0)
    swy = (w * logy).sum(0)
    swty = (w * tt * logy).sum(0)

    den = sw * swtt - swt ** 2
    ok = np.abs(den) > 1e-12
    slope = np.zeros_like(sw)
    inter = np.zeros_like(sw)
    slope[ok] = (sw[ok] * swty[ok] - swt[ok] * swy[ok]) / den[ok]
    inter[ok] = (swy[ok] - slope[ok] * swt[ok]) / sw[ok]

    pred = inter[None, :] + slope[None, :] * tt
    resid = np.sqrt((w * (logy - pred) ** 2).sum(0) / np.maximum(sw, EPS))
    return np.exp(np.clip(inter, -20, 5)), -slope, resid, ok


def fit_activity(ctx_act, ctx_times):
    """Per-voxel (a, k) with clamping and shrinkage applied."""
    T = ctx_act.shape[0]
    y = ctx_act.reshape(T, -1)
    a, k, resid, ok = weighted_loglinear(y, ctx_times)

    # Global decay from bright voxels only -- background is mostly noise.
    mean_y = y.mean(0)
    bright = mean_y > np.percentile(mean_y, 70)
    sel = bright & ok
    k_global = float(np.median(k[sel])) if sel.any() else 0.0
    k_global = float(np.clip(k_global, K_MIN, K_MAX))

    # Shrink toward k_global by how badly each voxel fits: a voxel with the
    # median residual gets weight 0.5, a clean fit keeps its own k, a noisy one
    # is pulled almost entirely to the global value.
    med_r = float(np.median(resid[ok])) if ok.any() else 1.0
    lam = SHRINK_STRENGTH * resid / max(med_r, EPS)
    wgt = 1.0 / (1.0 + lam)
    k = wgt * k + (1.0 - wgt) * k_global

    k = np.clip(k, K_MIN, K_MAX)
    a = np.clip(a, 0.0, 2.0)
    a = np.where(ok, a, y[-1])            # degenerate fit -> last observation
    k = np.where(ok, k, k_global)
    return a, k, k_global


def train_one_fold(train_tumors, val_tumors, test_tumor, epochs, out_dir,
                   device, fold_label=''):
    """No training occurs; train/val tumours are accepted and ignored.

    The predictor uses only the held-out tumour's own context frames, so it
    cannot leak information across folds. It sits inside the LOOCV loop purely
    so its numbers land in the same table as everything else.
    """
    push_config(globals())
    os.makedirs(out_dir, exist_ok=True)

    loaded = load_test_tumor(test_tumor, device)
    if loaded is None:
        print(f"  [warn] test tumour {test_tumor} not loadable")
        return None
    ctx, ctx_t, ctx_times_min, tgt_times_min, tgt_norm, gt_arr = loaded

    ctx_rgb = denorm(ctx[0]).cpu().permute(0, 2, 3, 1).numpy()   # (T,H,W,3)
    ctx_act = rgb_to_activity(ctx_rgb)                            # (T,H,W)
    ctx_times = ctx_t[0].cpu().numpy()
    tgt_times = tgt_norm.cpu().numpy()

    span = float(ctx_times.max() - ctx_times.min())
    horizon = float(tgt_times.max() - ctx_times.min())
    print(f"  context spans t={ctx_times.min():.3f}-{ctx_times.max():.3f} "
          f"(width {span:.3f}); predicting to t={tgt_times.max():.3f}")
    print(f"  extrapolation factor: {horizon / max(span, 1e-6):.1f}x beyond "
          f"the fitted window")

    a, k, k_global = fit_activity(ctx_act, ctx_times)
    print(f"  global decay constant k = {k_global:.3f}")

    H, W = ctx_act.shape[1], ctx_act.shape[2]
    preds = []
    for tt in tgt_times:
        act = np.clip(a * np.exp(-k * tt), 0.0, 1.0).reshape(H, W)
        preds.append(activity_to_rgb(act))
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
