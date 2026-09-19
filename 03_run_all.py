#!/usr/bin/env python
"""
03_run_all.py
=============
Runs LOOCV for all nine models, one after another, unattended.

Designed to be started and walked away from:
  * one subprocess per model, so a crash or CUDA OOM in one architecture
    cannot abort the remaining eight;
  * automatic retry at half batch size when a CUDA out-of-memory is detected,
    which is the realistic failure mode on a 4 GB GTX 1650 under WDDM;
  * resumable -- finished models are detected on restart and skipped, so if
    Windows reboots you just run the same command again;
  * live console output AND a per-model log file under the results tree;
  * running ETA for the whole sweep.

Time profiles (chosen with --profile):
    full    LOOCV_EPOCHS=800, MIN_EPOCHS=400, PATIENCE=150
            The settings in your scripts as written. On a GTX 1650 expect
            roughly 60-100 hours for the whole sweep. Early stopping cannot
            fire before epoch 400, so most folds run close to the full budget.

    fast    LOOCV_EPOCHS=400, MIN_EPOCHS=80, PATIENCE=60
            Lets early stopping actually do its job. Typically 8-16 hours
            total. Recommended for an overnight run.

    smoke   LOOCV_EPOCHS=3, MIN_EPOCHS=1, PATIENCE=2
            About 10-20 minutes. Proves the whole pipeline end to end -- data
            loading, all nine architectures, CSV output, comparison stats --
            without producing meaningful numbers. ALWAYS RUN THIS FIRST.

Usage:
    python 03_run_all.py --profile smoke
    python 03_run_all.py --profile fast
    python 03_run_all.py --profile fast --only unet vit
    python 03_run_all.py --profile fast --restart
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Windows consoles default to a legacy code page (cp1256, cp1252, cp437...)
# that cannot represent the box-drawing characters the pet_*.py scripts print
# in their summary tables. Without this, the run dies with UnicodeEncodeError
# on a cosmetic separator line. Force UTF-8 and fall back to replacing any
# character the terminal still cannot draw.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

HERE = Path(__file__).resolve().parent

# Ordered cheapest-first: the baseline and the small models finish early, so
# even a run that gets cut short yields a usable partial comparison table.
MODELS = [
    ("naive",           "pet_naive_baseline_paper"),
    ("unet",            "pet_unet_paper"),
    ("attention_unet",  "pet_attention_unet_paper"),
    ("attention_resnet", "pet_attention_resnet_paper"),
    ("cvae",            "pet_cvae_paper"),
    ("pix2pix",         "pet_pix2pix_paper"),
    ("temporal_unet",   "pet_temporal_unet_paper"),
    ("vit",             "pet_vit_paper"),
    ("convlstm",        "pet_convlstm_paper"),
    # --- additional models (share pet_common.py) ---
    ("kinetic",         "pet_kinetic_baseline_paper"),
    ("simvp",           "pet_simvp_paper"),
    ("cgan",            "pet_cgan_paper"),
    ("diffusion",       "pet_diffusion_paper"),
    ("cvae2",           "pet_cvae2_paper"),
    ("swin",            "pet_swin_paper"),
]

PROFILES = {
    "full":  {"PET_LOOCV_EPOCHS": "800", "PET_MIN_EPOCHS": "400", "PET_PATIENCE": "150"},
    "fast":  {"PET_LOOCV_EPOCHS": "400", "PET_MIN_EPOCHS": "80",  "PET_PATIENCE": "60"},
    "smoke": {"PET_LOOCV_EPOCHS": "3",   "PET_MIN_EPOCHS": "1",   "PET_PATIENCE": "2"},
}

# ConvLSTM holds whole sequences in memory and is the tightest fit on 4 GB.
# Sized for a 6 GB card (GTX 1660). On 4 GB halve these; on 12 GB+ double them.
BATCH_OVERRIDE = {"convlstm": "4", "vit": "16", "diffusion": "16",
                  "simvp": "16", "swin": "16", "cvae2": "16"}

OOM_SIGNS = ("out of memory", "CUDA error", "CUBLAS_STATUS_ALLOC_FAILED")


def inherit_data_paths(env):
    """Copy DATA_ROOT / IMG_SUBFOLDER out of pet_unet_paper.py into the env.

    The new models (pet_common.py) read PET_DATA_ROOT and PET_IMG_SUBFOLDER
    from the environment, so this keeps them in sync with whatever
    00_set_paths.py wrote into the nine original scripts. Without it you would
    have to set the path in two places and they would silently drift apart.
    """
    src_path = HERE / "pet_unet_paper.py"
    if not src_path.exists():
        return
    src = src_path.read_text(encoding="utf-8", errors="replace")
    for var, envname in (("DATA_ROOT", "PET_DATA_ROOT"),
                         ("IMG_SUBFOLDER", "PET_IMG_SUBFOLDER")):
        m = re.search(var + r'\s*=\s*r?["\']([^"\']*)["\']', src)
        if m:
            env[envname] = m.group(1)
            print(f"  {envname} = {m.group(1)}")


def human(seconds):
    return str(timedelta(seconds=int(seconds)))


def _has_torch(exe: Path) -> bool:
    """True if `exe` can import torch."""
    try:
        r = subprocess.run([str(exe), "-c", "import torch"],
                           capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


def resolve_python():
    """Pick the interpreter that actually has torch installed.

    Running `python 03_run_all.py` with the system interpreter is an easy
    mistake: the script itself needs nothing but the standard library, so it
    starts happily and then every single worker dies with
    ModuleNotFoundError: No module named 'torch'. Rather than depend on the
    user typing `.venv\\Scripts\\python` every time, look for the venv next to
    this script and verify torch before launching anything.
    """
    candidates = []
    for rel in ("Scripts/python.exe", "bin/python", "bin/python3"):
        p = HERE / ".venv" / rel
        if p.exists():
            candidates.append(p)
    candidates.append(Path(sys.executable))

    for exe in candidates:
        if _has_torch(exe):
            return exe

    print("=" * 70)
    print("  [FAIL] Could not find a Python with PyTorch installed.")
    print("=" * 70)
    print("  Checked:")
    for exe in candidates:
        print(f"    {exe}")
    print("\n  Most likely you have not run the setup script yet. From this")
    print("  folder, run:")
    print("      01_setup_env.bat")
    print("\n  If setup already succeeded, the .venv folder may be in a")
    print("  different folder from this script. Move .venv next to")
    print("  03_run_all.py, or run this script with the venv interpreter:")
    print("      .venv\\Scripts\\python 03_run_all.py --profile smoke")
    sys.exit(1)


def preflight(exe: Path):
    """Report interpreter, torch version and GPU before committing hours."""
    code = (
        "import torch;"
        "print('  python  :', __import__('sys').executable);"
        "print('  torch   :', torch.__version__);"
        "print('  cuda    :', torch.cuda.is_available());"
        "print('  gpu     :', torch.cuda.get_device_name(0) "
        "if torch.cuda.is_available() else 'CPU ONLY - will be very slow')"
    )
    r = subprocess.run([str(exe), "-c", code], capture_output=True, text=True)
    print(r.stdout.rstrip() or r.stderr.rstrip())
    if "cuda    : False" in r.stdout:
        print("\n  [warn] CUDA not available. Training on CPU will take days,")
        print("         not hours. Fix the install before running --profile fast.")


def is_done(results_root: Path, name: str) -> bool:
    return (results_root / name / f"manifest_{name}.json").exists()


def run_model(name, module, results_root, env_base, log_path, batch=None,
              python_exe=None):
    """Launch one worker. Returns (ok: bool, oom: bool)."""
    env = dict(env_base)
    if batch is not None:
        env["PET_BATCH_SIZE"] = str(batch)
    elif name in BATCH_OVERRIDE:
        env["PET_BATCH_SIZE"] = BATCH_OVERRIDE[name]

    cmd = [str(python_exe or sys.executable), str(HERE / "_pet_worker.py"),
           "--module", module, "--name", name, "--results", str(results_root)]

    tail = []
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n{'='*70}\nSTART {name} {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        if batch is not None:
            log.write(f"(retry at batch size {batch})\n")
        log.write(f"{'='*70}\n")
        log.flush()

        proc = subprocess.Popen(cmd, cwd=str(HERE), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            tail.append(line)
            if len(tail) > 80:
                tail.pop(0)
        proc.wait()

    ok = proc.returncode == 0
    blob = "".join(tail)
    oom = any(sign.lower() in blob.lower() for sign in OOM_SIGNS)
    if "No module named" in blob:
        # An environment problem, not a model problem. Every remaining model
        # will fail identically, so surface it once and stop rather than
        # printing the same traceback nine times.
        missing = blob.split("No module named")[-1].split("\n")[0].strip()
        print("\n" + "=" * 70)
        print(f"  [FAIL] Missing package {missing} in the worker interpreter.")
        print("=" * 70)
        print("  Install it into the same environment, e.g.:")
        print("      .venv\\Scripts\\python -m pip install scikit-image scipy pandas tqdm")
        print("  then re-run. Aborting so this does not repeat for every model.")
        sys.exit(1)
    return ok, oom


def main():
    ap = argparse.ArgumentParser(description="Run LOOCV for all models")
    ap.add_argument("--profile", choices=list(PROFILES), default="fast")
    ap.add_argument("--results", default=None, help="results root directory")
    ap.add_argument("--only", nargs="*", default=None, help="subset of model names")
    ap.add_argument("--restart", action="store_true",
                    help="re-run models that already finished")
    args = ap.parse_args()

    results_root = Path(args.results) if args.results else HERE / f"RESULTS_{args.profile}"
    results_root.mkdir(parents=True, exist_ok=True)
    logs_dir = results_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    todo = MODELS if not args.only else [m for m in MODELS if m[0] in args.only]
    if not todo:
        print(f"[FAIL] no models matched --only {args.only}")
        print("Valid names:", ", ".join(n for n, _ in MODELS))
        sys.exit(1)

    env_base = dict(os.environ)
    env_base.update(PROFILES[args.profile])
    env_base["PYTHONUNBUFFERED"] = "1"
    # Make every worker emit UTF-8 regardless of the console code page.
    env_base["PYTHONIOENCODING"] = "utf-8:replace"
    env_base["PYTHONUTF8"] = "1"
    # Reduces fragmentation on a small card; harmless if unsupported.
    env_base.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("=" * 70)
    print("  DATA PATHS (inherited by the pet_common.py models)")
    print("=" * 70)
    inherit_data_paths(env_base)
    print()

    python_exe = resolve_python()

    t0 = time.time()
    print("=" * 70)
    print("  ENVIRONMENT")
    print("=" * 70)
    preflight(python_exe)
    print()
    print("=" * 70)
    print(f"  PET LOOCV SWEEP -- profile '{args.profile}'")
    print(f"  started  : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  results  : {results_root}")
    print(f"  models   : {len(todo)}")
    print(f"  settings : {PROFILES[args.profile]}")
    if args.profile == "smoke":
        print("  NOTE: smoke profile produces meaningless metrics by design.")
    print("=" * 70)

    summary = []
    for idx, (name, module) in enumerate(todo, 1):
        if is_done(results_root, name) and not args.restart:
            print(f"\n[{idx}/{len(todo)}] {name}: already complete, skipping "
                  f"(use --restart to force)")
            summary.append((name, "skipped", 0.0))
            continue

        elapsed = time.time() - t0
        print(f"\n{'#'*70}")
        print(f"# [{idx}/{len(todo)}] {name}   elapsed {human(elapsed)}")
        if idx > 1 and elapsed > 0:
            eta = elapsed / max(idx - 1, 1) * (len(todo) - idx + 1)
            print(f"# rough sweep ETA: {human(eta)} remaining")
        print(f"{'#'*70}")

        log_path = logs_dir / f"{name}.log"
        m0 = time.time()
        ok, oom = run_model(name, module, results_root, env_base, log_path,
                            python_exe=python_exe)

        if not ok and oom:
            print(f"\n[retry] {name} hit CUDA OOM -- retrying at batch size 4")
            ok, oom = run_model(name, module, results_root, env_base, log_path,
                                batch=4, python_exe=python_exe)
            if not ok and oom:
                print(f"[retry] still OOM -- retrying at batch size 2")
                ok, _ = run_model(name, module, results_root, env_base, log_path,
                              batch=2, python_exe=python_exe)

        dt = time.time() - m0
        status = "ok" if ok else "FAILED"
        summary.append((name, status, dt))
        print(f"\n[{status}] {name} in {human(dt)}  (log: {log_path})")
        if not ok:
            print(f"[note] continuing with the remaining models")

    # ----------------------------------------------------------------------
    total = time.time() - t0
    print("\n" + "=" * 70)
    print("  SWEEP SUMMARY")
    print("=" * 70)
    print(f"  {'model':<20} {'status':<10} {'time':>12}")
    print("  " + "-" * 44)
    for name, status, dt in summary:
        print(f"  {name:<20} {status:<10} {human(dt):>12}")
    print("  " + "-" * 44)
    print(f"  {'TOTAL':<20} {'':<10} {human(total):>12}")
    print(f"  finished: {datetime.now():%Y-%m-%d %H:%M:%S}")

    with open(results_root / "sweep_summary.json", "w", encoding="utf-8") as fh:
        json.dump({
            "profile": args.profile,
            "settings": PROFILES[args.profile],
            "total_seconds": round(total, 1),
            "models": [{"name": n, "status": s, "seconds": round(d, 1)}
                       for n, s, d in summary],
        }, fh, indent=2)

    failed = [n for n, s, _ in summary if s == "FAILED"]
    if failed:
        print(f"\n[warn] failed models: {', '.join(failed)}")
        print(f"       inspect {logs_dir} for tracebacks")
    print(f"\nNext:  python 04_compare_models.py --results {results_root}")


if __name__ == "__main__":
    main()
