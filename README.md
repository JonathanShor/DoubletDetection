# DoubletDetection

[![DOI](https://zenodo.org/badge/86256007.svg)](https://zenodo.org/badge/latestdoi/86256007)
[![Documentation Status](https://readthedocs.org/projects/doubletdetection/badge/?version=latest)](https://doubletdetection.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
![Build Status](https://github.com/JonathanShor/DoubletDetection/workflows/doubletdetection/badge.svg)

DoubletDetection is a Python3 package to detect doublets (technical errors) in single-cell RNA-seq count matrices.

## Installing DoubletDetection

Install from PyPI

```bash
pip install doubletdetection
```

Install from source

```bash
git clone https://github.com/JonathanShor/DoubletDetection.git
cd DoubletDetection
pip3 install .
```

If you are using `pipenv` as your virtual environment, it may struggle installing from the setup.py due to our custom Phenograph requirement.
If so, try the following in the cloned repo:

```bash
pipenv run pip3 install .
```

## Running DoubletDetection

To run basic doublet classification:

```Python
import doubletdetection
clf = doubletdetection.BoostClassifier()
# raw_counts is a cells by genes count matrix
labels = clf.fit(raw_counts).predict()
# higher means more likely to be doublet
scores = clf.doublet_score()
```

- `raw_counts` is a scRNA-seq count matrix (cells by genes), and is array-like
- `labels` is a 1-dimensional numpy ndarray with the value 1 representing a detected doublet, 0 a singlet, and `np.nan` an ambiguous cell.
- `scores` is a 1-dimensional numpy ndarray representing a score for how likely a cell is to be a doublet. The score is used to create the labels.

The classifier works best when

- There are several cell types present in the data
- It is applied individually to each run in an aggregated count matrix

In `v2.5` we have added a new experimental clustering method (`scanpy`'s Louvain clustering) that is much faster than phenograph. We are still validating results from this new clustering. Please see the notebook below for an example of using this new feature.

## Tutorial

See our [jupyter notebook](https://nbviewer.jupyter.org/github/JonathanShor/DoubletDetection/blob/master/tests/notebooks/PBMC_10k_vignette.ipynb) for an example on 10k PBMCs from 10x Genomics.

## Obtaining data

Data can be downloaded from the [10x website](https://support.10xgenomics.com/single-cell/datasets).

## Credits and citations

Gayoso, Adam, Shor, Jonathan, Carr, Ambrose J., Sharma, Roshan, Pe'er, Dana (2020, December 18). DoubletDetection (Version v3.0). Zenodo. http://doi.org/10.5281/zenodo.2678041

We also thank the participants of the 1st Human Cell Atlas Jamboree, Chun J. Ye for providing data useful in developing this method, and Itsik Pe'er for providing guidance in early development as part of the Computational genomics class at Columbia University.

This project is licensed under the terms of the MIT license.
