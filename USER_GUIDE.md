# Object Detection System User Guide

## Overview

This object detection system provides real-time detection and tracking of objects from a webcam feed. It leverages deep learning models and CUDA acceleration to provide high performance on consumer hardware.

## System Requirements

### Hardware Requirements
- **CPU**: Intel Core i5/AMD Ryzen 5 or better
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060 or better recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Webcam**: Any standard USB webcam or built-in camera

### Software Requirements
- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.9 or higher
- **CUDA Toolkit**: 11.7 or higher
- **cuDNN**: Compatible with your CUDA version

## Installation

1. **Clone or download the repository**

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Alternative installation using setup.py**:
   ```
   pip install -e .
   ```
   
   For YOLO support:
   ```
   pip install -e .[yolo]
   ```

## Running the Application

### Using Batch Files (Windows)

The easiest way to run the application is to use one of the provided batch files:

- **Standard mode**: `run_detector.bat`
- **Speed-optimized**: `run_detector.bat speed`
- **Quality-optimized**: `run_detector.bat quality`
- **Maximum performance**: `run_detector.bat max`
- **Safe mode**: `run_detector.bat safe`
- **Benchmark mode**: `run_detector.bat benchmark`

### Using Python Directly

```
python -m src.main [options]
```

Options:
- `--model`: Model type (fasterrcnn, fasterrcnn_resnet101, maskrcnn, yolov5, yolov8)
- `--resolution`: Input resolution (tiny, small, medium, large)
- `--interval`: Detection interval (process every N frames)
- `--confidence`: Detection confidence threshold (0.0-1.0)
- `--disable_fp16`: Disable half-precision operations (true/false)

## Usage Tips

### Performance Optimization

1. **Balance detection interval and quality**:
   - Lower values (1-2) provide more responsive detection but lower FPS
   - Higher values (5-10) provide better FPS but less frequent updates

2. **Choose the right resolution**:
   - Smaller resolutions (tiny, small) provide better performance
   - Larger resolutions (medium, large) provide better detection accuracy

3. **Adjust confidence threshold**:
   - Higher values (0.5-0.7) reduce false positives but may miss objects
   - Lower values (0.3-0.4) detect more objects but may include false positives

### Low-Light Performance

The system includes adaptive low-light enhancement that automatically detects and improves dark scenes. You can:

- Toggle enhancement on/off with the 'e' key
- Adjust lighting in your environment for better detection
- Use a flashlight to highlight objects in very dark environments

## Troubleshooting

### Common Issues

1. **CUDA errors**:
   - Update your NVIDIA drivers
   - Try running in safe mode: `run_detector.bat safe`
   - Disable half-precision: `--disable_fp16=true`

2. **Low FPS**:
   - Reduce resolution: `--resolution=tiny`
   - Increase detection interval: `--interval=8`
   - Close other GPU-intensive applications

3. **Detection quality issues**:
   - Try a different model: `--model=fasterrcnn_resnet101`
   - Adjust lighting conditions
   - Ensure objects are clearly visible in the frame

### Getting Help

If you encounter persistent issues:

1. Run the troubleshooting script: `troubleshoot.bat`
2. Check the console output for error messages
3. Verify your system meets the requirements
4. Check for updates to the application

## Advanced Features

### Custom Models

The system supports multiple model types. To use your own custom model:

1. Place your model in the `models` directory
2. Update `src/models/detector.py` to load your model
3. Run the application with your model name

### Performance Monitoring

To monitor system performance:

1. Run in benchmark mode: `run_detector.bat benchmark`
2. Review the generated performance report
```

## 5. Create a requirements-dev.txt

```
# Development requirements
pytest>=7.3.1
black>=23.3.0
pylint>=2.17.0
isort>=5.12.0
mypy>=1.3.0
coverage>=7.2.5
```

These updated documentation files reflect the current state of your project, including all the optimizations and features we've implemented. They provide clear guidance on how to use the application, troubleshoot issues, and understand the architecture.

The README.md provides a quick overview and getting started guide, while the User Guide offers more detailed instructions. The requirements.txt and setup.py files ensure users can easily install the necessary dependencies.

Similar code found with 5 license types