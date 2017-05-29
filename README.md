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
from doubletdetection import BoostClassifier
clf = BoostClassifier(boost_rate=0.5, n_pca=20, knn=20)
labels = clf.fit(raw_counts)
```

`raw_counts` is a scRNA-seq count table. `labels` is a binary vector with the value 1 representing a synthetic doublet. The length of `labels` is equal to the length of `raw_counts`.

Advanced usage:

See our [jupyter notebook](http://nbviewer.jupyter.org/github/JonathanShor/DoubletDetection/blob/update-scripts/walkthrough.ipynb).


## Obtaining data 
Data can be downloaded from the [10x website](https://support.10xgenomics.com/single-cell/datasets).
