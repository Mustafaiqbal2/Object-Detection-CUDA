from setuptools import setup, find_packages

setup(
    name='object-tracking-project',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python project for AI-powered object tracking using webcam.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'opencv-python',
        'tensorflow',  # or 'torch' if using PyTorch
        'numpy',
        'matplotlib',
        'pyyaml',
        'pytest'
    ],
    entry_points={
        'console_scripts': [
            'object-tracking=main:main',  # Adjust according to your main function
        ],
    },
)