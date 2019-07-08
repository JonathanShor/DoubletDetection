from setuptools import setup, find_packages
import sys

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]

if sys.version_info.major != 3:
    print("doubletdetection requires Python 3.7")
    sys.exit(1)

setup(
    name="doubletdetection",
    version="3.0.0",
    description="Method to detect and enable removal of doublets from single-cell RNA-sequencing "
    "data",
    url="https://github.com/JonathanShor/DoubletDetection",
    author="Adam Gayoso, Jonathan Shor, Ambrose J. Carr",
    author_email="ajg2188@columbia.edu",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        "numpy>=1.14.2",
        "pandas>=0.22.0",
        "scipy>=1.0.1",
        "scikit-learn",
        "tables>=3.4.2",
        "umap-learn>=0.3.7",
        "matplotlib>=2.2.2",
    ],
)
