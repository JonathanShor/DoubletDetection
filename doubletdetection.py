#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:45:28 2017

@author: adamgayoso, JonathanShor, ryanbrand
"""

import numpy as np
import time
import phenograph
import collections
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from synthetic import create_synthetic_data
from synthetic import create_simple_synthetic_data
from synthetic import getCellTypes
from synthetic import doubletFromCelltype
from synthetic import downsampledDoublets
from synthetic import sameDownsampledDoublets
import utils

PCA_COMPONENTS = 30
DOUBLET_RATE = 0.25
KNN = 20


def classify(raw_counts, downsample = True, doublet_rate=DOUBLET_RATE):
    """
    Classifier for doublets in single-cell RNA-seq data
    :param raw_counts: count table in numpy.array format
    :param probabilistic: option to use sampled doublets vs linear doublets
    :return counts: mixed counts (real and fake) in numpy ndarray format NORMALIZED
    :return scores: doublet score for each row in counts
    :return communities: Phenograph community for each row in counts
    :return doublet_labels: indicator for each row in counts whether it is a fake doublet (doublets appended to end)
    """
    parents = None
    if downsample == True:
        D = doublet_rate
        synthetic, doublet_labels = downsampledDoublets(raw_counts, normalize=True, doublet_rate=D)
    elif downsample == "Same":
        D = doublet_rate
        synthetic, doublet_labels, parents = sameDownsampledDoublets(raw_counts, normalize=True, doublet_rate=D)
    else:
        # Simple synthetic data
        # Requires numpy.array
        D = doublet_rate
        synthetic, doublet_labels = create_simple_synthetic_data(raw_counts, 0.6, 0.6, normalize=True, doublet_rate=D)

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
