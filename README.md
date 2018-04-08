# DoubletDetection

DoubletDetection is a Python3 package to detect doublets (technical errors) in single-cell RNA-seq count matrices.

To install DoubletDetection, clone or download this repository and run.

```bash
pip3 install --upgrade .
```

To run basic doublet classification:

```
import doubletdetection
raw_counts = doubletdetection.load_csv('/your/path/filename.csv')
clf = doubletdetection.BoostClassifier()
labels = clf.fit(raw_counts)
```

`raw_counts` is a scRNA-seq count matrix, and must be a 2-dimensional numpy ndarray, with rows 
being cells. `labels` is a binary 1-dimensional numpy ndarray with the value 1 representing a 
detected doublet. The length of `labels` is equal to the length of `raw_counts`.


## Obtaining data
Data can be downloaded from the [10x website](https://support.10xgenomics.com/single-cell/datasets).


This project is licensed under the terms of the MIT license.

## Citations

bioRxiv submission is in the works.

