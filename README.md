# Object Tracking Project

This project implements an AI-powered object tracking system using a webcam. It leverages Python with OpenCV and TensorFlow/PyTorch for object detection and tracking, with optional C++ extensions for performance optimizations using CUDA.

## Project Structure

The project is organized as follows:

```
object-tracking-project
├── src
│   ├── main.py                # Entry point for the application
│   ├── camera
│   │   ├── __init__.py
│   │   └── webcam.py          # Webcam capture and processing
│   ├── models
│   │   ├── __init__.py
│   │   ├── detector.py        # Object detection models
│   │   └── tracker.py         # Object tracking algorithms
│   ├── utils
│   │   ├── __init__.py
│   │   ├── visualization.py   # Visualization tools
│   │   └── performance.py     # Performance monitoring
│   └── cpp_extensions         # C++ CUDA optimizations
│       ├── CMakeLists.txt
│       ├── setup.py           # For building C++ extensions
│       ├── include
│       │   └── cuda_ops.h
│       └── src
│           └── cuda_ops.cpp
├── configs
│   └── default.yaml           # Configuration parameters
├── data
│   ├── models                 # Pre-trained models
│   └── test_videos            # Sample videos for testing
├── notebooks
│   └── model_exploration.ipynb
├── tests
│   ├── test_detector.py
│   ├── test_tracker.py
│   └── test_webcam.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd object-tracking-project
   ```

2. Create a Python virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

4. (Optional) If you want to use C++ extensions for performance optimizations, ensure you have CUDA installed and follow the instructions in the `src/cpp_extensions/README.md` for building the C++ components.

## Usage

1. Start the application:
   ```
   python src/main.py
   ```

2. The application will initialize the webcam, load the object detection model, and start tracking objects in real-time.

## Configuration

Configuration parameters can be modified in the `configs/default.yaml` file. This includes paths to pre-trained models and settings for the tracking algorithms.

## Testing

Unit tests are provided for the object detection and tracking functionalities. To run the tests, use:
```
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.