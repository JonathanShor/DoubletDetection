# DoubletDetection

DoubletDetection is a Python3 package to detect doublets (technical errors) in single-cell RNA-seq count matrices.

To install DoubletDetection:

```
git clone https://github.com/JonathanShor/DoubletDetection.git
cd DoubletDetection
pip3 install --upgrade .
```

To run basic doublet classification:

```
import doubletdetection
clf = doubletdetection.BoostClassifier()
# raw_counts is a cells by genes count matrix
labels = clf.fit(raw_counts).predict()
```

`raw_counts` is a scRNA-seq count matrix (cells by genes), and is array-like. `labels` is a binary 1-dimensional numpy ndarray with the value 1 representing a 
detected doublet.

See our [jupyter notebook](https://nbviewer.jupyter.org/github/JonathanShor/DoubletDetection/blob/master/docs/PBMC_8k_vignette.ipynb) for an example on 8k PBMCs from 10x.

## Obtaining data
Data can be downloaded from the [10x website](https://support.10xgenomics.com/single-cell/datasets).


## Citations

bioRxiv submission is in the works.

This project is licensed under the terms of the MIT license.
