@echo off
REM ============================================================================
REM  RUN_ALL_FINAL.bat
REM  The definitive run: every model, one environment, one sweep.
REM
REM  WHY A FULL RE-RUN RATHER THAN PATCHING IN THE MISSING PIECES
REM  cuDNN is nondeterministic, so re-running any model shifts its numbers
REM  slightly even with a fixed seed. A results table assembled from several
REM  different sweeps is therefore not internally consistent, and "these eight
REM  came from one run and those four from another" is not a sentence you want
REM  in a Methods section. One sweep, one table.
REM
REM  What this run adds that the earlier ones could not:
REM    * activity-space metrics for EVERY model, not just the newest four
REM      (the colormap is decoded before SSIM/PSNR/MAE are computed)
REM    * saved predictions per fold, so any future metric question can be
REM      answered without retraining anything
REM    * training histories as CSV, which feeds 05_overfitting_report.py
REM
REM  Results go to RESULTS_final so your existing RESULTS_fast is untouched.
REM
REM  EXPECT 12-20 HOURS on a GTX 1660. Start it and leave it. If the machine
REM  restarts, run this file again -- finished models are skipped.
REM ============================================================================

setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1

set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo [STOP] No .venv folder here. Run START_HERE.bat first.
    pause
    exit /b 1
)

REM Metrics are computed on decoded activity, not on the jet-colormapped RGB.
REM Both spaces are always saved to fold_outputs.npz either way; this selects
REM which one lands in the headline table.
set PET_METRIC_SPACE=activity

echo.
echo ############################################################
echo #   PET LOOCV  --  FINAL FULL SWEEP
echo #
echo #   All models, 10 folds each, metrics in ACTIVITY space.
echo #   12-20 hours. Safe to interrupt and resume.
echo ############################################################
echo.

"%VPY%" 03_run_all.py --profile fast --results RESULTS_final

echo.
echo ============================================================
echo   Statistics
echo ============================================================
"%VPY%" 04_compare_models.py --results RESULTS_final

echo.
echo ============================================================
echo   Overfitting diagnostics
echo ============================================================
"%VPY%" 05_overfitting_report.py --results RESULTS_final

echo.
echo ############################################################
echo #   DONE -- send Claude these from RESULTS_final\analysis\ :
echo #     comparison_table.csv
echo #     vs_baseline.csv
echo #     overfitting_summary.csv
echo ############################################################
echo.
pause
endlocal
