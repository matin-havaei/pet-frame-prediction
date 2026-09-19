#!/usr/bin/env python
"""
_pet_worker.py
==============
Runs the LOOCV loop for ONE model inside its own process. Launched by
03_run_all.py; not normally invoked by hand.

Why a separate process per model rather than one long script:
  * a CUDA OOM or a crash in one architecture cannot take down the whole run;
  * all VRAM is returned to the driver between models, which matters a lot on
    a 4 GB card under Windows WDDM where the display also holds memory;
  * each model gets its own clean log file.

The worker monkeypatches `train_one_fold` on the imported module so that the
per-FRAME metric arrays it already returns are captured and written to disk.
Nothing inside the model files needs to change for this: run_loocv() looks up
train_one_fold as a module global, so rebinding the module attribute is enough.
Per-frame values are required later for the cluster bootstrap in 04_compare.

Usage:
    python _pet_worker.py --module pet_unet_paper --name unet --results DIR
"""

import argparse
import csv
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Unbuffered stdout so the orchestrator's tee sees progress live.
# Unbuffered so the orchestrator sees progress live, and UTF-8 so the box
# drawing characters in the model scripts' tables cannot raise
# UnicodeEncodeError on a legacy Windows code page.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace",
                            line_buffering=True)
    except Exception:
        try:
            _stream.reconfigure(errors="replace", line_buffering=True)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="module name, no .py")
    ap.add_argument("--name", required=True, help="short label for outputs")
    ap.add_argument("--results", required=True, help="results root directory")
    args = ap.parse_args()

    results_root = Path(args.results)
    model_dir = results_root / args.name
    model_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 70)
    print(f"  MODEL: {args.name}   (module {args.module})")
    print("=" * 70)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    mod = importlib.import_module(args.module)

    # Redirect all of the module's own outputs into the unified results tree.
    mod.OUTPUT_DIR = str(model_dir)
    os.makedirs(mod.OUTPUT_DIR, exist_ok=True)

    # run_loocv() normally relies on main() having resolved this first.
    if getattr(mod, "MAX_TIME_MIN", 0) == 0:
        mod.MAX_TIME_MIN = mod.autodetect_max_time(
            mod.ALL_TUMORS, mod.DATA_ROOT, mod.IMG_SUBFOLDER
        )
    print(f"[cfg] MAX_TIME_MIN = {mod.MAX_TIME_MIN}")
    print(f"[cfg] tumours      = {len(mod.ALL_TUMORS)} -> {mod.ALL_TUMORS}")

    per_frame_rows = []
    fold_timings = []
    fold_splits = []

    is_naive = args.name == "naive"

    if is_naive:
        # ------------------------------------------------------------------
        # Parameter-free baselines: no training, so just evaluate every tumour
        # directly and harvest the per-frame arrays run_naive already returns.
        # ------------------------------------------------------------------
        for test_t in mod.ALL_TUMORS:
            t0 = time.time()
            out = mod.run_naive(test_t)
            if out is None:
                print(f"  [warn] {test_t}: not found, skipped")
                continue
            (last_metrics, mean_metrics, _ctx, _ctx_t, tgt_times, *_rest) = out
            for method, (ssim_s, psnr_s, mae_s) in (
                ("last_frame", last_metrics),
                ("mean_frame", mean_metrics),
            ):
                for k in range(len(ssim_s)):
                    per_frame_rows.append({
                        "model": method,
                        "test_tumor": test_t,
                        "frame_idx": k,
                        "t_min": tgt_times[k] if k < len(tgt_times) else "",
                        "ssim": float(ssim_s[k]),
                        "psnr": float(psnr_s[k]),
                        "mae": float(mae_s[k]),
                    })
            dt = time.time() - t0
            fold_timings.append({"test_tumor": test_t, "seconds": round(dt, 1)})
            print(f"  fold {test_t:<16} done in {dt:6.1f}s")

        # Also produce the script's own standard LOOCV CSVs/console summary.
        mod.run_loocv()

    else:
        # ------------------------------------------------------------------
        # Learned models: wrap train_one_fold to capture per-frame metrics and
        # wall-clock timing, then hand control to the script's own run_loocv.
        # ------------------------------------------------------------------
        original_train_one_fold = mod.train_one_fold

        def wrapped(train_tumors, val_tumors, test_tumor, *a, **kw):
            t0 = time.time()
            fold_splits.append({
                "test_tumor": test_tumor,
                "val_tumors": list(val_tumors),
                "n_train": len(train_tumors),
                "train_tumors": list(train_tumors),
            })
            print(f"\n[fold] test={test_tumor}  val={list(val_tumors)}  "
                  f"train={len(train_tumors)} tumours")
            out = original_train_one_fold(train_tumors, val_tumors, test_tumor,
                                          *a, **kw)
            dt = time.time() - t0
            fold_timings.append({"test_tumor": test_tumor, "seconds": round(dt, 1)})

            if out is not None:
                ssim_s, psnr_s, mae_s = out
                for k in range(len(ssim_s)):
                    per_frame_rows.append({
                        "model": args.name,
                        "test_tumor": test_tumor,
                        "frame_idx": k,
                        "t_min": "",
                        "ssim": float(ssim_s[k]),
                        "psnr": float(psnr_s[k]),
                        "mae": float(mae_s[k]),
                    })
            elapsed = time.time() - t_start
            done = len(fold_timings)
            total = len(mod.ALL_TUMORS)
            if done < total:
                eta = elapsed / done * (total - done)
                print(f"[time] fold {done}/{total} took {dt/60:.1f} min | "
                      f"model ETA {eta/60:.1f} min")
            return out

        mod.train_one_fold = wrapped

        device = mod.setup_device()
        mod.run_loocv(device)

    # ----------------------------------------------------------------------
    # Persist the artefacts 04_compare_models.py needs.
    # ----------------------------------------------------------------------
    pf_path = model_dir / f"per_frame_{args.name}.csv"
    if per_frame_rows:
        with open(pf_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_frame_rows[0].keys()))
            w.writeheader()
            w.writerows(per_frame_rows)
        print(f"\n[out] per-frame metrics ({len(per_frame_rows)} rows) -> {pf_path}")
    else:
        print("\n[warn] no per-frame metrics captured")

    manifest = {
        "model": args.name,
        "module": args.module,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_seconds": round(time.time() - t_start, 1),
        "n_folds": len(fold_timings),
        "fold_timings": fold_timings,
        "fold_splits": fold_splits,
        "config": {
            k: getattr(mod, k, None)
            for k in ("LOOCV_EPOCHS", "MIN_EPOCHS", "PATIENCE", "BATCH_SIZE",
                      "N_CONTEXT", "IMG_SIZE", "SEED", "MAX_TIME_MIN")
        },
    }
    with open(model_dir / f"manifest_{args.name}.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    mins = (time.time() - t_start) / 60
    print(f"[done] {args.name} finished in {mins:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
