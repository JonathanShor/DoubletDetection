#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:29:52 2017

@author: adamgayoso
"""

import doubletdetection
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import utils
import pandas as pd
import matplotlib.pyplot as plt
import visualize
import phenograph

FNAME = "~/Google Drive/Computational Genomics/pbmc8k_dense.csv"
SYN_FNAME = "~/Google Drive/Computational Genomics/synthetic.csv"

# DataFrame
raw_counts = utils.dataAcquisition(FNAME, normalize=False)

# Get scores
np_raw_counts, scores, communities_w_doublets = doubletdetection.classify(raw_counts, probabilistic=True)
validate = False

if validate:
    np_raw_counts, scores, communities, true_doublet_labels = doubletdetection.validate(raw_counts)
    for s in range(20, 80, 2):
        cutoff = s/float(100)
        test = scores[np.where(scores>cutoff)[0]]
        print(cutoff, len(test))
    
    print(np.sum(true_doublet_labels[np.where(scores>0.63)[0]])/float(len(scores[np.where(scores>0.63)[0]])))

# Visualize tSNE clustering
# Different color for each cluster and a different mark for identified doublet 
# Only visualize raw counts

# This turns 0s of np_raw_counts to NaN, not sure if that's important, but if you want to use np_raw_counts, 0s are NaN
counts = doubletdetection.utils.normalize_counts_10x(np_raw_counts)

# Default num_componenets is 2
# If you run without reducing counts it kills your memory
pca = PCA(n_components=30)
reduced_counts = pca.fit_transform(counts)

communities, graph, Q = phenograph.cluster(reduced_counts)

tsne = TSNE()
tsne_counts = tsne.fit_transform(reduced_counts)

cutoff = 0.37
doublet_labels = np.zeros((reduced_counts.shape[0],))
doublet_labels[np.where(scores>0.37)[0]] = 1

# Viz with phenograph results with combined data
visualize.tsne_scatter(tsne_counts, communities_w_doublets, doublet_labels)
# Viz with phenograph results with raw normed data, doublets are black
visualize.tsne_scatter(tsne_counts, communities, doublet_labels)

