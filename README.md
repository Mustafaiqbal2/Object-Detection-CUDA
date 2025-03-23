# Object Detection System

A high-performance, real-time object detection and tracking system using PyTorch and CUDA.

## Features

- Real-time object detection using RCNN and YOLO models
- Object tracking with trajectory visualization
- Adaptive performance optimization for higher frame rates
- Low-light enhancement for better detection in dark environments
- Multiple pre-configured performance profiles
- CUDA-accelerated operations for faster inference

## Requirements

- Python 3.9+
- CUDA-capable NVIDIA GPU (RTX series recommended)
- PyTorch 2.0+
- CUDA Toolkit 11.7+
- OpenCV 4.7+

See `requirements.txt` for a complete list of dependencies.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/object-detection.git
   cd object-detection
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Install development requirements:
   ```
   pip install -r requirements-dev.txt
   ```

## Quick Start

The easiest way to run the application is using one of the provided batch files:

- **Balanced Performance (Default)**:
  ```
  run_detector.bat
  ```

- **Maximum Speed**:
  ```
  run_detector.bat speed
  ```

- **High Quality**:
  ```
  run_detector.bat quality
  ```

- **Compatibility Mode** (for troubleshooting):
  ```
  run_detector.bat safe
  ```

## Advanced Usage

You can customize the detection parameters through command-line arguments:

```
python -m src.main --model=fasterrcnn --resolution=medium --interval=3 --confidence=0.45 --disable_fp16=false
```

### Command-line Options

- `--model`: Detection model to use (fasterrcnn, fasterrcnn_resnet101, maskrcnn, yolov5, yolov8)
- `--resolution`: Input size for detection (tiny, small, medium, large)
- `--interval`: Process every N frames for detection (higher values = more FPS but less frequent updates)
- `--confidence`: Detection confidence threshold (0.0-1.0)
- `--disable_fp16`: Disable half-precision operations for better compatibility

## Keyboard Controls

While the application is running, you can use these keyboard commands:

- `q`: Quit the application
- `d`: Cycle through detection interval settings (process every N frames)
- `e`: Toggle image enhancement for low-light conditions
- `t`: Toggle trajectory visualization

## Performance Tuning

For optimal performance:

1. Use a smaller resolution (`--resolution=small` or `--resolution=tiny`)
2. Increase the detection interval (`--interval=5` or higher)
3. Lower the confidence threshold for better detection in difficult scenes (`--confidence=0.35`)
4. If you experience CUDA errors, try disabling FP16 (`--disable_fp16=true`)

## Troubleshooting

If you encounter issues:

1. Run the troubleshooting batch file:
   ```
   troubleshoot.bat
   ```

2. Try the safe mode:
   ```
   run_safe.bat
   ```

3. Make sure your GPU drivers are up to date

4. Check that PyTorch is installed with CUDA support:
   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Architecture

The system consists of several key components:

- **Detector**: Implements various object detection models (RCNN, YOLO)
- **Tracker**: Tracks objects between frames for continuous detection
- **Enhancement**: Adaptive image processing for low-light conditions
- **CUDA Utilities**: Optimized CUDA operations for faster processing
