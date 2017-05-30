# DoubletDetection

DoubletDetection is a Python3 package to detect doublets (technical errors) in Single-cell RNA-seq count tables.

To install DoubletDetection, clone or download this repository.

To run DoubletDetection you may need to install the following packages:
- numpy
- pandas
- sklearn
- phenograph

To install PhenoGraph:

```
pip3 install git+https://github.com/jacoblevine/phenograph.git
```

To run basic doublet classification:

```
import doubletdetection
raw_counts = doubletdetection.load_csv('/your/path/filename.csv')
clf = doubletdetection.BoostClassifier()
labels = clf.fit(raw_counts)
```

`raw_counts` is a scRNA-seq count table. `labels` is a binary vector with the value 1 representing a synthetic doublet. The length of `labels` is equal to the length of `raw_counts`.

Advanced usage:

See our [jupyter notebook](docs/walkthrough.ipynb).


## Obtaining data
Data can be downloaded from the [10x website](https://support.10xgenomics.com/single-cell/datasets).
