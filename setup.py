from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="camera-lidar-fusion",
    version="1.0.0",
    author="B.Tech AI/ML Student",
    author_email="your.email@example.com",
    description="Camera-LiDAR Fusion System for Robust Object Detection in Autonomous Vehicles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cclasher916-spec/camera-lidar-fusion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "viz": ["mayavi", "vtk"],
    },
    entry_points={
        "console_scripts": [
            "camera-lidar-fusion=src.main:main",
        ],
    },
)