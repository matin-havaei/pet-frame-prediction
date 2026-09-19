@echo off
REM ============================================================================
REM  01_setup_env.bat  --  Environment setup for PET LOOCV benchmark
REM  Target system: NVIDIA GeForce GTX 1650 (Turing, sm_75, 4 GB), Windows WDDM
REM  Driver 595.97 -> supports CUDA runtime up to 13.2 (backward compatible)
REM
REM  Run from an ordinary Command Prompt in the folder holding your pet_*.py
REM  files:      01_setup_env.bat
REM ============================================================================

setlocal enabledelayedexpansion
echo.
echo ============================================================
echo   PET LOOCV -- environment setup
echo ============================================================
echo.

REM ---------------------------------------------------------------------------
REM  1. Locate a suitable Python (3.10 - 3.13; 3.14 has no CUDA wheels yet)
REM ---------------------------------------------------------------------------
set "PY_CMD="
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined PY_CMD (
        py -%%V -c "import sys" >nul 2>&1
        if !errorlevel! equ 0 (
            set "PY_CMD=py -%%V"
            echo [ok] Found Python %%V
        )
    )
)
if not defined PY_CMD (
    python -c "import sys; assert (3,10) <= sys.version_info < (3,14)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=python"
        echo [ok] Using default 'python'
    )
)
if not defined PY_CMD (
    echo [FAIL] No Python 3.10-3.13 found.
    echo        CUDA wheels are not published for Python 3.14 yet.
    echo        Install Python 3.12 from python.org and re-run this script.
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM  2. Create virtual environment
REM ---------------------------------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo [ok] Reusing existing .venv
) else (
    echo [..] Creating virtual environment .venv
    %PY_CMD% -m venv .venv
    if errorlevel 1 ( echo [FAIL] venv creation failed & exit /b 1 )
)
set "VPY=.venv\Scripts\python.exe"

echo [..] Upgrading pip
"%VPY%" -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 ( echo [FAIL] pip upgrade failed & exit /b 1 )

REM ---------------------------------------------------------------------------
REM  3. Install PyTorch with CUDA
REM
REM     IMPORTANT: on Windows, plain "pip install torch" pulls a CPU-ONLY
REM     wheel from PyPI. The --index-url below is what makes it CUDA-enabled.
REM
REM     cu128 chosen deliberately: Turing sm_75 is fully supported, the build
REM     is mature, and driver 595.97 runs any CUDA <= 13.2 runtime.
REM ---------------------------------------------------------------------------
echo.
echo [..] Installing PyTorch, CUDA 12.8 build -- this downloads ~2.5 GB
"%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo.
    echo [warn] cu128 install failed. Retrying with the CUDA 13.0 index...
    "%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    if errorlevel 1 ( echo [FAIL] PyTorch install failed & exit /b 1 )
)

REM ---------------------------------------------------------------------------
REM  4. Scientific / plotting dependencies used by the pet_*.py scripts
REM ---------------------------------------------------------------------------
echo.
echo [..] Installing remaining requirements
"%VPY%" -m pip install "numpy<2.3" pillow matplotlib tqdm scikit-image scipy pandas statsmodels
if errorlevel 1 ( echo [FAIL] dependency install failed & exit /b 1 )

REM ---------------------------------------------------------------------------
REM  5. Verify CUDA really works and that sm_75 kernels are present
REM ---------------------------------------------------------------------------
echo.
echo ============================================================
echo   Verification
echo ============================================================
"%VPY%" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch      :',torch.__version__); print('cuda build :',torch.version.cuda); print('cuda avail :',ok); sys.exit(0 if ok else 3)"
if errorlevel 3 (
    echo.
    echo [FAIL] CUDA is NOT available to PyTorch.
    echo        Almost always this means a CPU-only wheel got installed.
    echo        Fix:  .venv\Scripts\python -m pip uninstall -y torch torchvision
    echo              .venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    exit /b 1
)

"%VPY%" -c "import torch; p=torch.cuda.get_device_properties(0); cc='sm_%%d%%d'%%(p.major,p.minor); archs=torch.cuda.get_arch_list(); print('gpu        :',p.name); print('vram       : %%.2f GB'%%(p.total_memory/1024**3)); print('capability :',cc); print('arch list  :',archs); print('sm_75 ok   :', cc in archs)"

"%VPY%" -c "import torch; a=torch.randn(512,512,device='cuda'); b=a@a; torch.cuda.synchronize(); print('matmul test: OK'); print('scikit-image / scipy:', __import__('skimage').__version__, __import__('scipy').__version__)"
if errorlevel 1 ( echo [FAIL] GPU smoke test failed & exit /b 1 )

echo.
echo ============================================================
echo   Setup complete.
echo   Next:  .venv\Scripts\python 02_patch_scripts.py
echo ============================================================
endlocal
