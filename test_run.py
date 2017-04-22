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
from matplotlib.colors import LinearSegmentedColormap
import phenograph

FNAME = "~/Google Drive/Computational Genomics/pbmc8k_dense.csv"
SYN_FNAME = "~/Google Drive/Computational Genomics/synthetic.csv"

# DataFrame
raw_counts = utils.dataAcquisition(FNAME, normalize=False)


validate = True

if validate:
    counts, scores, communities, true_doublet_labels, fake_doublet_labels = doubletdetection.validate(raw_counts)
    true_scores = scores[:len(true_doublet_labels),:]
    for s in range(20, 80, 2):
        cutoff = s/float(100)
        test = true_scores[np.where(true_scores>cutoff)[0]]
        print(cutoff, len(test))
    print(np.sum(true_doublet_labels[np.where(scores>0.63)[0]])/float(len(true_scores[np.where(scores>0.63)[0]])))

else:
    # Get scores
    counts_w_doublets, scores_w_doublets, communities_w_doublets, doublet_labels = doubletdetection.classify(raw_counts, probabilistic=True)
    for s in range(20, 80, 2):
        cutoff = s/float(100)
        test = scores_w_doublets[np.where(scores_w_doublets>cutoff)[0]]
        print(cutoff, len(test))

# Visualize tSNE clustering
# Different color for each cluster and black doublet 
# Only visualize raw counts

#counts = doubletdetection.utils.normalize_counts_10x(counts_w_doublets)

# Default tsne num_componenets is 2
# If you run without reducing counts it kills your memory
pca = PCA(n_components=30)
reduced_counts = pca.fit_transform(counts_w_doublets)

communities, graph, Q = phenograph.cluster(reduced_counts)

print('\nCreating tSNE reduced counts\n')
tsne = TSNE()
tsne_counts = tsne.fit_transform(reduced_counts)

#cutoff = 0.59
#doublet_labels = np.zeros((reduced_counts.shape[0],))
#doublet_labels[np.where(scores>0.37)[0]] = 1

# data viz
set1i = LinearSegmentedColormap.from_list('set1i', plt.cm.Set1.colors, N=100)
    
colors = communities
x = tsne_counts[:,0]
y = tsne_counts[:,1]
plt.scatter(x,y, c=colors, s=10, cmap=set1i)
doublets = np.where(doublet_labels==1)[0]
plt.scatter(x[doublets], y[doublets], s=10, color='black')

