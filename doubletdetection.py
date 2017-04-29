"""Doublet detection in single-cell RNA-seq data."""

import numpy as np
import phenograph
import collections
from sklearn.decomposition import PCA
from synthetic import create_simple_synthetic_data
from synthetic import downsampledDoublets
from synthetic import sameDownsampledDoublets

PCA_COMPONENTS = 30
DOUBLET_RATE = 0.25
KNN = 20
# TODO: pack these globals into classify as param defaults


def classify(raw_counts, downsample=True, doublet_rate=DOUBLET_RATE):
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
        D = doublet_rate
        synthetic, doublet_labels = downsampledDoublets(raw_counts, normalize=True, doublet_rate=D)
    elif downsample == "Same":
        D = doublet_rate
        synthetic, doublet_labels, parents = sameDownsampledDoublets(raw_counts, normalize=True,
                                                                     doublet_rate=D)
    else:
        # Simple synthetic data
        # Requires numpy.array
        D = doublet_rate
        synthetic, doublet_labels = create_simple_synthetic_data(raw_counts, 0.6, 0.6,
                                                                 normalize=True, doublet_rate=D)

    counts = synthetic

    print("\nClustering mixed data set with Phenograph...\n")
    # Get phenograph results
    pca = PCA(n_components=PCA_COMPONENTS)
    reduced_counts = pca.fit_transform(counts)
    communities, graph, Q = phenograph.cluster(reduced_counts, k=KNN)
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

    return counts, scores, communities, doublet_labels, parents
