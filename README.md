# DoubletDetection
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2678042.svg)](https://doi.org/10.5281/zenodo.2678042)
[![Documentation Status](https://readthedocs.org/projects/doubletdetection/badge/?version=latest)](https://doubletdetection.readthedocs.io/en/latest/?badge=latest)


DoubletDetection is a Python3 package to detect doublets (technical errors) in single-cell RNA-seq count matrices.


To install DoubletDetection:

```
git clone https://github.com/JonathanShor/DoubletDetection.git
cd DoubletDetection
pip3 install .
```

To run basic doublet classification:

```
import doubletdetection
clf = doubletdetection.BoostClassifier()
# raw_counts is a cells by genes count matrix
labels = clf.fit(raw_counts).predict()
```

- `raw_counts` is a scRNA-seq count matrix (cells by genes), and is array-like
- `labels` is a 1-dimensional numpy ndarray with the value 1 representing a detected doublet, 0 a singlet, and `np.nan` an ambiguous cell.

The classifier works best when
- There are several cell types present in the data
- It is applied individually to each run in an aggregated count matrix

See our [jupyter notebook](https://nbviewer.jupyter.org/github/JonathanShor/DoubletDetection/blob/master/tests/notebooks/PBMC_8k_vignette.ipynb) for an example on 8k PBMCs from 10x.

## Obtaining data
Data can be downloaded from the [10x website](https://support.10xgenomics.com/single-cell/datasets).


## Citations
bioRxiv submission and journal publication expected in the coming months. Please use the following for now:

Gayoso, Adam, & Shor, Jonathan. (2018, July 17). DoubletDetection (Version v2.4). Zenodo. http://doi.org/10.5281/zenodo.2678042

This project is licensed under the terms of the MIT license.
