# DoubletDetection

DoubletDetection is a package to detect doublets (technical errors) in Single-cell RNA-seq count tables. 

This version is for the final submission for COMS4761 Computational Genomics at Columbia University. For an up-to-date version, please check out this [repository](https://github.com/JonathanShor/Doublet-Detection).

To install DoubletDetection, clone or download this repository.

To run DoubletDetection you will need the following packages:
- numpy
- pandas
- sklearn
- collections
- phenograph

To install PhenoGraph:

```
pip3 install git+https://github.com/jacoblevine/phenograph.git
```

To run basic doublet classification:

```
import DoubletDetection
raw_counts = DoubletDetection.load_data(FILENAME)
counts, scores, communities, doublet_labels, cutoff = DoubletDetection.classify(raw_counts) 
```

The return values contain values for the augmented dataset (original data and synthetic doublets). To identify doublets within the original data you can do the following:

```
cell_count = raw_counts.shape[0]
doublets = np.where(scores[:cell_count]>=cutoff)[0]
```
`doublets` will contain the indices of the suggested doublets.

