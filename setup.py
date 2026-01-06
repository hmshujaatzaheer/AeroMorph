"""
AeroMorph: Unified Perception-Driven Morphological Adaptation Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aeromorph",
    version="0.1.0",
    author="H M Shujaat Zaheer",
    author_email="shujabis@gmail.com",
    description="Unified framework for autonomous morphing decisions in aerial robots",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hmshujaatzaheer/AeroMorph",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "viz": [
            "matplotlib>=3.5.0",
        ],
        "full": [
            "torch>=2.0.0",
            "scipy>=1.9.0",
            "matplotlib>=3.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aeromorph-demo=examples.basic_p2ma:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
