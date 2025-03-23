@echo off
REM Safe Mode Object Detection Runner
REM -------------------------------

title Object Detection - Safe Mode

echo =====================================
echo Object Detection System - SAFE MODE
echo =====================================
echo Running with compatibility settings
echo.

REM Set environment variables for stability
set CUDA_LAUNCH_BLOCKING=1
set OMP_NUM_THREADS=2
set PYTORCH_NO_CUDA_MEMORY_CACHING=1

echo Starting application in safe mode...
echo.

REM Run with safe settings
python -m src.main --model=fasterrcnn --resolution=medium --interval=2 --confidence=0.4

pause