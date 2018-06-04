from setuptools import setup
import sys
import shutil
from subprocess import check_call

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]

if sys.version_info.major != 3:
    print('doubletdetection requires Python 3')
    sys.exit(1)

setup(
    name='doubletdetection',
    version='2.3.0',
    description='Method to detect and enable removal of doublets from single-cell RNA-sequencing '
                'data',
    url='https://github.com/JonathanShor/DoubletDetection',
    author='Adam Gayoso, Jonathan Shor, Ambrose J. Carr',
    author_email='ajg2188@columbia.edu',
    package_dir={'': 'src'},
    packages=['doubletdetection'],
    install_requires=[],
)
