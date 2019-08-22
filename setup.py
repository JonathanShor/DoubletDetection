from setuptools import setup
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
    version="2.5.0",
    description="Method to detect and enable removal of doublets from single-cell RNA-sequencing "
    "data",
    url="https://github.com/JonathanShor/DoubletDetection",
    author="Adam Gayoso, Jonathan Shor, Ambrose J. Carr",
    author_email="ajg2188@columbia.edu",
    include_package_data=True,
    packages=["doubletdetection"],
    install_requires=[
        "phenograph @ https://api.github.com/repos/JonathanShor/PhenoGraph/tarball/v1.6",
        "matplotlib>=3.1",
        "scanpy>=1.4.4",
        "louvain",
    ],
)
