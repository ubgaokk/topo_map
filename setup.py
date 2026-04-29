from setuptools import setup, find_packages

setup(
    name="topo_map",
    version="0.1.0",
    description="Endpoint-Aware Hierarchical Point-Lane Graph for Driving Scene Topology Reasoning",
    author="Eric Gao",
    author_email="ubgaokk@example.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "pyyaml>=6.0",
        "scipy>=1.10.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)