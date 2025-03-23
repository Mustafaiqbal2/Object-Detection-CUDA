from setuptools import setup, find_packages

setup(
    name="object-detection",
    version="1.0.0",
    description="High-performance object detection and tracking system",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/object-detection",
    packages=find_packages(),
    python_requires=">=3.9.0",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.22.0",
        "opencv-python>=4.7.0",
        "pillow>=9.4.0",
    ],
    extras_require={
        "yolo": ["ultralytics>=8.0.0"],
        "dev": ["pytest>=7.3.1", "black>=23.3.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="object detection, computer vision, CUDA, PyTorch, deep learning",
    project_urls={
        "Source": "https://github.com/yourusername/object-detection",
        "Bug Reports": "https://github.com/yourusername/object-detection/issues",
    },
    entry_points={
        "console_scripts": [
            "object-detection=src.main:main",
        ],
    },
)