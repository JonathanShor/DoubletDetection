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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import phenograph
import sys
from optparse import OptionParser
import collections

FNAME = "~/Google Drive/Computational Genomics/pbmc8k_dense.csv"
# SYN_FNAME = "~/Google Drive/Computational Genomics/synthetic.csv"
# VALIDATE = True


def main(validate):
    # Read in data
    raw_counts = utils.dataAcquisition(FNAME)

    if validate:
        counts, scores, communities, true_doublet_labels, fake_doublet_labels = (
            doubletdetection.validate(raw_counts))
        true_scores = scores[:len(true_doublet_labels), :]
        for s in range(20, 80, 2):
            cutoff = s / float(100)
            test = true_scores[np.where(true_scores > cutoff)[0]]
            print(cutoff, len(test))
        print(np.sum(true_doublet_labels[np.where(true_scores > 0.63)[0]]) / (
            float(len(true_scores[np.where(true_scores > 0.63)[0]]))))
        return
    else:
        # Get scores
        counts_w_doublets, scores_w_doublets, communities_w_doublets, doublet_labels = (
            doubletdetection.classify(raw_counts, probabilistic=False))
        true_scores = scores_w_doublets[:raw_counts.shape[0], :]
        for s in range(20, 80, 2):
            cutoff = s / float(100)
            test = true_scores[np.where(true_scores > cutoff)[0]]
            print(cutoff, len(test))

    # Visualize tSNE clustering
    # Different color for each cluster and black doublet
    # Only visualize raw counts

    #counts = doubletdetection.utils.normalize_counts(counts_w_doublets)

    # Default tsne num_componenets is 2
    # If you run without reducing counts it kills your memory
    # This works with validation off
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

    colors = communities_w_doublets
    x = tsne_counts[:, 0]
    y = tsne_counts[:, 1]
    plt.scatter(x, y, c=colors, s=4, cmap=set1i)
    doublets = np.where(doublet_labels == 1)[0]
    plt.scatter(x[doublets], y[doublets], facecolors='none', edgecolors='black', s=7)
    plt.show()

    # Bar chart stacked of counts
    raw_com_count = collections.Counter(communities_w_doublets[:raw_counts.shape[0]])
    doublet_com_count = collections.Counter(communities_w_doublets[doublets])
    
    # Original
    labels, values = zip(*raw_com_count.items())
    indexes = np.arange(len(labels))
    width = 0.75
    plt.bar(indexes, values, width)
    
    fakes = []
    for com in labels:
        fakes.append(doublet_com_count[com])

    plt.bar(indexes, fakes, width, bottom = values)
    
    plt.xticks(indexes, labels)
    plt.show()
    
    # Score bar chart 
    labels, values = zip(*raw_com_count.items())
    indexes = np.arange(len(labels))
    width = 0.75
    
    scores = []
    for com in labels:
        score = np.unique(scores_w_doublets[np.where(communities_w_doublets == com)[0]])
        scores.append(score[0])
            
    plt.bar(indexes, scores, width)
    plt.xticks(indexes, labels)
    plt.show()
    
    
    

if __name__ == '__main__':
    argv = sys.argv[1:]
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file", help='read csv data from FILE', metavar='FILE')
    # parser.add_option("-g", type='string', dest="gcpathname", help='gc5Base file')
    # parser.add_option("-w", type='int', dest="wind", help='window size')
    parser.add_option("-v", dest="validate", action="store_true", default=False)
    (options, _args) = parser.parse_args()
    if options.file:
        FNAME = options.file
    if options.validate:
        print("Validation run starting.")

    main(options.validate)
