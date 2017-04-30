"""Doublet detection in single-cell RNA-seq data."""

import numpy as np
import pandas as pd
import phenograph
import collections
from sklearn.decomposition import PCA


def classify(raw_counts, downsample=True, doublet_rate=0.25, k=20, n_pca=30):
    """Classifier for doublets in single-cell RNA-seq data.

    Args:
        raw_counts (ndarray): count table
        downsample (bool, optional): Downsample doublets.
        doublet_rate (TYPE, optional): Description

    Returns:
        ndarray, ndim=2: Normalized mixed counts (real and fake).
        ndarray: doublet score for each row in counts as column vector.
        TYPE: Phenograph community for each row in counts
        ndarray, ndim=1: indicator for each row in counts whether it is a fake
            doublet (doublets appended to end)
        ndarray, ndim=1: Parent cell for each row in counts when
            downsample="Same"
    """
    parents = None
    if downsample:
        counts, doublet_labels, parents = createLinearDoublets(raw_counts,doublet_rate=doublet_rate,
                                downsample=True)
    elif downsample == "Same":
        counts, doublet_labels, parents = createLinearDoublets(raw_counts,doublet_rate=doublet_rate,
                                downsample=True, duplicate_parents=True)
    else:
        # Simple linear combination
        counts, doublet_labels = createLinearDoublets(raw_counts, doublet_rate=doublet_rate, alpha1=0.6, alpha2=0.6, downsample=False, duplicate_parents=False)

    print("\nClustering mixed data set with Phenograph...\n")
    # Get phenograph results
    pca = PCA(n_components=n_pca)
    reduced_counts = pca.fit_transform(counts)
    communities, graph, Q = phenograph.cluster(reduced_counts, k=k)
    c_count = collections.Counter(communities)
    print('\n')

    # Count number of fake doublets in each community and assign score
    phenolabels = np.append(communities[:, np.newaxis], doublet_labels[:, np.newaxis], axis=1)

    synth_doub_count = {}
    scores = np.zeros((len(communities), 1))
    for c in np.unique(communities):
        c_indices = np.where(phenolabels[:, 0] == c)[0]
        synth_doub_count[c] = np.sum(phenolabels[c_indices, 1]) / float(c_count[c])
        scores[c_indices] = synth_doub_count[c]
        
    # Find a cutoff score
    potential_cutoffs = list(synth_doub_count.values())
    potential_cutoffs.sort(reverse=True)
    max_dropoff = 0
    for i in range(len(potential_cutoffs)-1):
        dropoff = potential_cutoffs[i] - potential_cutoffs[i+1]
        if dropoff > max_dropoff:
            max_dropoff = dropoff
            cutoff = potential_cutoffs[i]

    return counts, scores, communities, doublet_labels, parents, cutoff


# TODO: Further detail of downsampling algorithm?
def downsampleCellPair(cell1, cell2):
    """Downsample the sum of two cell gene expression profiles.

    Args:
        cell1 (ndarray, ndim=1): Gene count vector.
        cell2 (ndarray, ndim=1): Gene count vector.

    Returns:
        ndarray, ndim=1: Downsampled gene count vector.
    """
    new_cell = cell1 + cell2

    lib1 = np.sum(cell1)
    lib2 = np.sum(cell2)
    new_lib_size = int(max(lib1, lib2))
    mol_ind = np.random.permutation(int(lib1 + lib2))[:new_lib_size]
    mol_ind += 1
    bins = np.append(np.zeros((1)), np.cumsum(new_cell))
    new_cell = np.histogram(mol_ind, bins)[0]

    return new_cell


def createLinearDoublets(raw_counts, normalize=True, doublet_rate=0.25, downsample=True,
                         duplicate_parents=False, alpha1=1.0, alpha2=1.0):
    """Appends doublets to end of data

    Args:
        raw_counts (ndarray): count data
        normalize (bool, optional): normalize data before returning
        doublet_rate (float, optional): Proportion of cell_counts to produce as
            doublets.
        downsample (bool, optional): downsample doublets
        duplicate_parents (bool, optional): Create doublets of same cell.
        alpha1 (float, optional): weighting of row1 in sum
        alpha2 (float, optional): weighting of row2 in sum

    Returns:
        ndarray, ndims=2: synthetic data
        ndarray, ndims=1: 0 for original data, 1 for fake doublet
        ndarray, ndim=1: One parent cell for each row in counts when
            downsample="Same"
    """
    # Get shape
    cell_count = raw_counts.shape[0]
    gene_count = raw_counts.shape[1]

    # Number of doublets to add
    doublets = int(doublet_rate * cell_count)

    synthetic = np.zeros((doublets, gene_count))

    # Add labels column to know which ones are doublets
    labels = np.zeros(cell_count + doublets)
    labels[cell_count:] = 1

    parents = np.zeros(cell_count + doublets)

    for i in range(doublets):
        row1 = np.random.randint(cell_count)
        if duplicate_parents:
            row2 = row1
        else:
            row2 = np.random.randint(cell_count)

        if downsample:
            new_row = downsampleCellPair(raw_counts[row1], raw_counts[row2])
        else:
            new_row = np.array(np.around(alpha1 * raw_counts[row1] + alpha2 * raw_counts[row2]),
                               dtype=raw_counts.dtype)

        synthetic[i] = new_row
        parents[i] = row1

    synthetic = np.append(raw_counts, synthetic, axis=0)

    if normalize:
        synthetic = normalize_counts(synthetic)

    return synthetic, labels, parents


def load_data(FNAME, normalize=False, read_rows=None):
    """Load a csv table from the filesystem.

    Args:
        FNAME (string): Pathname of file to load.
        normalize (bool, optional): Runs normalize_counts on table.
        read_rows (None or int, optional): If specified, load only first
            read_rows of FNAME.

    Returns:
        ndarray: Loaded table.
    """
    counts = pd.read_csv(FNAME, index_col=0, nrows=read_rows).as_matrix()

    if normalize:
        counts = normalize_counts(counts)

    return counts


def normalize_counts(raw_counts, standardizeGenes=False):
    """Normalize count array using method in 10x pipeline.

    From http://www.nature.com/articles/ncomms14049.

    Args:
        raw_counts (ndarray): count data
        doStandardize (bool, optional): Standardizes each gene column.

    Returns:
        ndarray: Normalized data.
    """
    # Sum across cells
    cell_sums = np.sum(raw_counts, axis=1)

    # Mutiply by median and divide each cell by cell sum
    median = np.median(cell_sums)
    raw_counts = raw_counts * median / cell_sums[:, np.newaxis]

    raw_counts = np.log(raw_counts + 0.1)

    if standardizeGenes:
        # Normalize to have genes with mean 0 and std 1
        std = np.std(raw_counts, axis=0)[np.newaxis, :]

        # Fix potential divide by zero
        std[np.where(std == 0)[0]] = 1

        normed = (raw_counts - np.mean(raw_counts, axis=0)) / std
    else:
        normed = raw_counts

    return normed