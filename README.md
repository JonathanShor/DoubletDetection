# DoubletDetection

DoubletDetection is a package to detect doublets (technical errors) in Single-cell RNA-seq count tables.

This version is for the final submission for CMBF4761 Computational Genomics at Columbia University. For an up-to-date version, please check out this [repository](https://github.com/JonathanShor/Doublet-Detection).

To install DoubletDetection, clone or download this repository. This software is designed to run on personal computers, no extra computing resources necessary. This code is written in Python 3.

To run DoubletDetection you may need to install the following packages:
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
import doubletdetection
raw_counts = doubletdetection.load_data(FILENAME)
counts, scores, communities, doublet_labels, cutoff = doubletdetection.classify(raw_counts) 
```

The return values contain values for the augmented dataset (original data and synthetic doublets). Therefore, `counts, scores, communities, doublet_labels` are of length `N(1+boost_rate)`, where `N` is the number of rows in raw_counts. The default `boost_rate` is 15%. `doublet_labels` is a binary vector with the value 1 representing a synthetic doublet. Synthetic doublets are appended to the end of raw_counts. To identify doublets within the original data you can do the following:

```
cell_count = raw_counts.shape[0]
doublets = np.where(scores[:cell_count]>=cutoff)[0]
```
`doublets` will contain the indices of the suggested doublets.

## Testing
`visualization_script.py` contains the pipline we used to visualize our results for our presentations. You will also need to install the newest version of `matplotlib`

To run:
```
python3 visualization_script.py -f [file_name] -c [cutoff_score] -t
```
The option `-f` is mandatory and should contain the path to the dataset. The option `-t` is optional and creates tSNE scatter plots. Running with this option will make the code take longer. The option `-c` is optional and represents a user defined cutoff score to use. 

`validation_script.py` is a script that runs DoubletDetection on our validation dataset (the 50:50 dataset). Sourced from [10x](https://support.10xgenomics.com/single-cell/datasets/jurkat:293t_50:50).

To run:
```
python3 validation_script.py -f [file_name] -n [trials]
```
The option `-f` is mandatory and should contain the path to the validation dataset. The option `-n` is the number of trials to use in the validation simulation. This simulation runs the classification n times and prints summary statistics.

## Obtaining data 
Data can be downloaded from the [10x webiste](https://support.10xgenomics.com/single-cell/datasets). To use the same data we did, please see the ReadMe submitted with the report for instructions.
