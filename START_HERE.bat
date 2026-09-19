@echo off
REM ============================================================================
REM  START_HERE.bat
REM  Double-click this. It does every setup step in order and stops with a
REM  clear message if anything goes wrong.
REM
REM  Steps:  1. build the Python environment (installs PyTorch + CUDA)
REM          2. point the scripts at your images and check the data
REM          3. apply the LOOCV fixes to the model scripts
REM          4. run a 15-minute smoke test across all 13 models
REM
REM  The real run is a separate file: RUN_REAL.bat
REM ============================================================================

setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1

echo.
echo ############################################################
echo #   PET LOOCV  --  SETUP
echo #   This will take 10-30 minutes. Leave the window open.
echo ############################################################
echo.

REM --- sanity check: are the model scripts actually here? --------------------
if not exist "pet_unet_paper.py" (
    echo [STOP] pet_unet_paper.py is not in this folder.
    echo.
    echo        This file must sit in the SAME folder as your nine
    echo        pet_*_paper.py scripts. Right now it is in:
    echo            %CD%
    echo.
    echo        Move everything into one folder and try again.
    pause
    exit /b 1
)

REM ============================ STEP 1 ======================================
echo.
echo ============================================================
echo   STEP 1 of 4 -- building the Python environment
echo ============================================================
call 01_setup_env.bat
if errorlevel 1 (
    echo.
    echo [STOP] Setup failed. Read the message above and send it to Claude.
    pause
    exit /b 1
)
set "VPY=.venv\Scripts\python.exe"

REM ============================ STEP 2 ======================================
echo.
echo ============================================================
echo   STEP 2 of 4 -- finding your images
echo ============================================================
"%VPY%" 00_set_paths.py
if errorlevel 1 (
    echo.
    echo [STOP] Your image folder could not be read.
    echo        If your images are somewhere other than
    echo            C:\PET_DATA\IMAGES
    echo        then run this instead, with your real path:
    echo            .venv\Scripts\python 00_set_paths.py --data "C:\your\path"
    pause
    exit /b 1
)

echo.
echo   ^>^> Check the table above: every tumour should say "ok".
echo   ^>^> If any say MISSING, stop now and tell Claude.
echo.
pause

REM ============================ STEP 3 ======================================
echo.
echo ============================================================
echo   STEP 3 of 4 -- applying the LOOCV fixes
echo ============================================================
"%VPY%" 02_patch_scripts.py
if errorlevel 1 (
    echo.
    echo [STOP] Patching failed. Nothing was changed. Send the message to Claude.
    pause
    exit /b 1
)

REM ============================ STEP 4 ======================================
echo.
echo ============================================================
echo   STEP 4 of 4 -- smoke test (about 15 minutes)
echo.
echo   This trains every model for 3 epochs. The NUMBERS IT PRINTS
echo   ARE MEANINGLESS ON PURPOSE. The only thing that matters is
echo   that all 13 models say "ok" in the summary table.
echo ============================================================
echo.
"%VPY%" 03_run_all.py --profile smoke

echo.
echo ############################################################
echo #   SETUP FINISHED
echo ############################################################
echo.
echo   Look at the SWEEP SUMMARY table above.
echo.
echo   All 13 models say "ok"      -^>  double-click RUN_REAL.bat
echo   Any model says "FAILED"     -^>  send Claude a screenshot
echo.
pause
endlocal
