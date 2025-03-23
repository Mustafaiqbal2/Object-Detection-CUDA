@echo off
REM Advanced Object Detection Launcher with Performance Profiles
REM -----------------------------------------------------------

REM Set title
title Advanced Object Detection

REM Check command-line arguments for performance profile
set PERF_PROFILE=balanced
if "%1"=="speed" set PERF_PROFILE=speed
if "%1"=="quality" set PERF_PROFILE=quality
if "%1"=="max" set PERF_PROFILE=max
if "%1"=="safe" set PERF_PROFILE=safe
if "%1"=="benchmark" set PERF_PROFILE=benchmark

REM Display startup banner
echo =====================================
echo Advanced Object Detection System
echo =====================================
echo Performance profile: %PERF_PROFILE%
echo.

REM Apply environment settings based on profile
if "%PERF_PROFILE%"=="speed" (
    REM Speed-optimized settings
    set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    set OMP_NUM_THREADS=1
    set CUDA_LAUNCH_BLOCKING=0
    set MODEL_TYPE=fasterrcnn
    set RESOLUTION=small
    set DETECTION_INTERVAL=5
    set CONFIDENCE=0.35
    
    echo Optimizing for maximum speed:
    echo - Using smaller input resolution (384x384)
    echo - Faster R-CNN with ResNet-50
    echo - Processing every 5th frame
    echo - Lower confidence threshold (0.35)
) else if "%PERF_PROFILE%"=="quality" (
    REM Quality-optimized settings
    set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
    set OMP_NUM_THREADS=2
    set CUDA_LAUNCH_BLOCKING=0
    set MODEL_TYPE=fasterrcnn_resnet101
    set RESOLUTION=medium
    set DETECTION_INTERVAL=2
    set CONFIDENCE=0.5
    
    echo Optimizing for better quality:
    echo - Using medium input resolution (512x512)
    echo - Faster R-CNN with ResNet-101
    echo - Processing every 2nd frame
    echo - Standard confidence threshold (0.50)
) else if "%PERF_PROFILE%"=="max" (
    REM Maximum performance (use with caution)
    set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    set OMP_NUM_THREADS=1
    set CUDA_LAUNCH_BLOCKING=0
    set MODEL_TYPE=fasterrcnn
    set RESOLUTION=tiny
    set DETECTION_INTERVAL=8
    set CONFIDENCE=0.3
    
    echo Optimizing for MAXIMUM PERFORMANCE:
    echo - Using tiny input resolution (256x256)
    echo - Faster R-CNN with ResNet-50
    echo - Processing every 8th frame
    echo - Low confidence threshold (0.30)
    echo - WARNING: Quality will be significantly reduced
) else if "%PERF_PROFILE%"=="safe" (
    REM Safe mode with no optimizations (for compatibility)
    set PYTORCH_CUDA_ALLOC_CONF=
    set OMP_NUM_THREADS=4
    set CUDA_LAUNCH_BLOCKING=1
    set MODEL_TYPE=fasterrcnn
    set RESOLUTION=medium
    set DETECTION_INTERVAL=1
    set CONFIDENCE=0.5
    
    echo Running in SAFE MODE:
    echo - Standard resolution (512x512)
    echo - Faster R-CNN with ResNet-50
    echo - Processing every frame
    echo - No CUDA optimizations
    echo - Full precision only (no FP16)
) else if "%PERF_PROFILE%"=="benchmark" (
    REM Benchmark mode to test performance
    set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
    set OMP_NUM_THREADS=2
    set CUDA_LAUNCH_BLOCKING=1
    set MODEL_TYPE=fasterrcnn
    set RESOLUTION=medium
    set DETECTION_INTERVAL=1
    set CONFIDENCE=0.4
    set BENCHMARK_MODE=1
    
    echo Running in BENCHMARK MODE:
    echo - Testing different configurations
    echo - Full system performance evaluation
    echo - Results will be saved to benchmark_results.txt
) else (
    REM Balanced default settings
    set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    set OMP_NUM_THREADS=2
    set CUDA_LAUNCH_BLOCKING=0
    set MODEL_TYPE=fasterrcnn
    set RESOLUTION=medium
    set DETECTION_INTERVAL=3
    set CONFIDENCE=0.45
    
    echo Using balanced settings:
    echo - Medium input resolution (512x512)
    echo - Faster R-CNN with ResNet-50
    echo - Processing every 3rd frame
    echo - Balanced confidence threshold (0.45)
)

echo.
echo Starting application...
echo.

REM Run the application with the selected profile settings
python -m src.main --model=%MODEL_TYPE% --resolution=%RESOLUTION% --interval=%DETECTION_INTERVAL% --confidence=%CONFIDENCE%

REM Keep the window open on error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error occurred! Press any key to exit.
    pause > nul
)