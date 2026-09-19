#!/usr/bin/env python
"""
05_overfitting_report.py
========================
Quantifies overfitting for every model and fold, so the paper can state it
rather than gesture at it.

Three independent signals, none of which requires retraining:

  1. TRAIN / VALIDATION GAP. From training_history.csv where available (the
     pet_common.py models) and otherwise parsed from the run logs, which every
     model prints. Reported as the ratio of final validation loss to final
     training loss. A ratio near 1 means the model generalises to the held-out
     validation tumours; a large ratio means it has memorised the training
     tumours.

  2. EARLY-STOP BEHAVIOUR. Whether each fold stopped on patience or ran to the
     epoch cap. Stopping early is overfitting caught and prevented. Hitting the
     cap in most folds means the budget, not the data, was the binding
     constraint -- an UNDER-training signal, and something a reviewer will ask
     about if a model looks weak.

  3. VALIDATION-TO-TEST CONSISTENCY. Correlation across folds between the
     validation loss a model achieved and the test SSIM it went on to score.
     This is the one that matters at n=10. Strong correlation means validation
     performance predicts held-out performance, so model selection was sound.
     Weak or negative correlation means the model is fitting tumour identity
     rather than kinetics, and the early-stopping checkpoint was chosen on a
     signal unrelated to generalisation.

Usage:
    python 05_overfitting_report.py --results RESULTS_fast
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Matches the per-epoch lines every model prints, e.g.
#   "Ep  120/400 | loss=0.0413 val=0.0587 patience=12/60"
#   "Ep    1/400 | G=28.59 D=0.52 val=0.3708 patience=0/60"
EPOCH_RE = re.compile(
    r"Ep\s+(\d+)\s*/\s*(\d+).*?(?:loss|G|mse|recon|v)\s*=\s*([\d.eE+-]+).*?"
    r"val\s*=\s*([\d.eE+-]+)")
# The nine original scripts print "Early stopping at epoch N" (with a unicode
# bolt prefix); the pet_common.py models print "Early stop at epoch N".
STOP_RE = re.compile(r"Early stop(?:ping)? at epoch (\d+)")
FOLD_RE = re.compile(r"\[fold\]\s*test=(\S+)")


def parse_log(path):
    """Yield one record per fold found in a model's log."""
    text = path.read_text(encoding='utf-8', errors='replace')
    folds, cur = [], None
    for line in text.splitlines():
        f = FOLD_RE.search(line)
        if f:
            if cur:
                folds.append(cur)
            cur = {'test_tumor': f.group(1), 'train': [], 'val': [],
                   'stopped_at': None, 'cap': None}
            continue
        if cur is None:
            continue
        m = EPOCH_RE.search(line)
        if m:
            cur['cap'] = int(m.group(2))
            cur['train'].append(float(m.group(3)))
            cur['val'].append(float(m.group(4)))
        s = STOP_RE.search(line)
        if s:
            cur['stopped_at'] = int(s.group(1))
    if cur:
        folds.append(cur)
    # A log may contain more than one attempt at the same fold (a crashed run
    # followed by a successful one). Keep only the last attempt per tumour.
    dedup = {}
    for f in folds:
        dedup[f['test_tumor']] = f
    return list(dedup.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    root = Path(args.results)
    logs = sorted((root / 'logs').glob('*.log'))
    if not logs:
        raise SystemExit(f"No logs under {root/'logs'}")

    # Test SSIM per (model, fold), for signal 3.
    per_tumor = {}
    pt_path = root / 'analysis' / 'per_tumor_means.csv'
    if pt_path.exists():
        d = pd.read_csv(pt_path)
        for _, r in d.iterrows():
            per_tumor[(r['model'], r['test_tumor'])] = r['ssim']

    rows, fold_rows = [], []
    for log in logs:
        model = log.stem
        folds = parse_log(log)
        if not folds:
            continue
        gaps, stops, caps, vals, tests = [], [], [], [], []
        degrades, bests = [], []
        for f in folds:
            if not f['val']:
                continue
            tr_end = float(np.mean(f['train'][-3:]))
            va_end = float(np.mean(f['val'][-3:]))
            # NOTE: val/train ratios are NOT comparable across models, because
            # each model's training loss is a different quantity on a different
            # scale (the cGAN's includes 100 x L1, the diffusion model's is a
            # v-MSE, and so on). Kept for within-model reference only.
            gap = va_end / tr_end if tr_end > 1e-9 else np.nan
            # Scale-free and comparable: how far validation loss climbed back
            # up from its own minimum. 0 = never degraded; 0.5 = ended 50%
            # worse than its best. This is overfitting measured directly.
            v_arr = np.asarray(f['val'], dtype=float)
            v_best = float(v_arr.min())
            degrade = ((va_end - v_best) / v_best) if v_best > 1e-9 else np.nan
            best_ep = int(v_arr.argmin()) + 1
            early = f['stopped_at'] is not None
            gaps.append(gap); stops.append(early); caps.append(f['cap'] or 0)
            degrades.append(degrade); bests.append(best_ep)
            vals.append(va_end)
            tests.append(per_tumor.get((model, f['test_tumor']), np.nan))
            fold_rows.append({
                'model': model, 'test_tumor': f['test_tumor'],
                'final_train': round(tr_end, 5), 'final_val': round(va_end, 5),
                'val_over_train': round(gap, 3),
                'val_degradation': round(degrade, 3),
                'stopped_epoch': f['stopped_at'] or f['cap'],
                'early_stopped': early,
            })

        if not gaps:
            continue
        v, t = np.array(vals), np.array(tests)
        ok = ~np.isnan(t)
        # Lower validation loss should mean higher test SSIM -> negative r is
        # the healthy sign. Reported flipped so positive = good.
        corr = (-np.corrcoef(v[ok], t[ok])[0, 1]
                if ok.sum() >= 3 and np.std(v[ok]) > 0 else np.nan)
        rows.append({
            'model': model,
            'n_folds': len(gaps),
            'val_degradation': round(float(np.nanmean(degrades)), 3),
            'val_over_train': round(float(np.mean(gaps)), 3),
            'pct_early_stopped': round(100 * float(np.mean(stops)), 1),
            'mean_stop_epoch': int(np.mean([r['stopped_epoch']
                                            for r in fold_rows
                                            if r['model'] == model])),
            'epoch_cap': int(np.median(caps)),
            'val_predicts_test_r': round(float(corr), 3)
            if not np.isnan(corr) else np.nan,
        })

    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    summary = pd.DataFrame(rows).sort_values('val_degradation')
    summary.to_csv(out / 'overfitting_summary.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out / 'overfitting_per_fold.csv', index=False)

    print("=" * 84)
    print("  OVERFITTING REPORT")
    print("=" * 84)
    print(summary.to_string(index=False))
    print()
    print("How to read this:")
    print("  val_degradation     PRIMARY overfitting measure, and the only one")
    print("                      comparable ACROSS models: how far validation")
    print("                      loss climbed back from its own minimum.")
    print("                      0.00 = never degraded. 0.50 = ended 50% worse")
    print("                      than its best epoch.")
    print("  val_over_train      within-model reference ONLY. Not comparable")
    print("                      across models: each has a different training")
    print("                      loss on a different scale (the cGAN's carries")
    print("                      a 100x L1 term, so its ratio looks tiny).")
    print("  pct_early_stopped   high  = patience is doing its job.")
    print("                      0%    = hit the epoch cap, likely UNDER-trained.")
    print("  val_predicts_test_r positive and large = validation loss predicted")
    print("                      held-out performance, so model selection was")
    print("                      sound. Near zero or negative = the early-stop")
    print("                      checkpoint was chosen on a signal unrelated to")
    print("                      generalisation; report this as a limitation.")
    print(f"\nWritten to {out}")


if __name__ == '__main__':
    main()
