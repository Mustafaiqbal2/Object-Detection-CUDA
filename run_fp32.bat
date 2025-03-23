@echo off
REM Full Precision Object Detection
REM -----------------------------

title Object Detection - FP32 Mode

echo =====================================
echo Object Detection System - FP32 MODE
echo =====================================
echo Running with full precision (no FP16)
echo.

REM Set environment for stability
set CUDA_LAUNCH_BLOCKING=1
set PYTORCH_NO_CUDA_MEMORY_CACHING=1
set DISABLE_FP16=1

echo Starting application in FP32 mode...
echo.

python -m src.main --model=fasterrcnn --resolution=small --interval=3 --confidence=0.4 --disable_fp16=true

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error occurred! Press any key to exit.
    pause > nul
)