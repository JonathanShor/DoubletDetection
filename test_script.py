#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
	Sample usage of the DoubletDetection module.

    To run from within DoubletDetection directory:
    	python3 ./test_script.py -f [file_name] -c [cutoff_score] -t

    Note: all command line flags optional other than file name
"""

import numpy as np
import matplotlib.pyplot as plt
import doubletdetection
import phenograph
import sys
import collections
import math

from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from optparse import OptionParser

FNAME = "~/Google Drive/Computational Genomics/pbmc8k_dense.csv"
PCA_COMPONENTS=30
BOOST_RATE = 0.25
KNN=30

def main(cutoff_score, tsne):

    ############################################
    #
    # Process Data and Perform Doublet-Detection
    #
    ############################################

    # Read in data and perform Phenograph clustering
    print("Loading data...\n")
    raw_counts = doubletdetection.load_data(FNAME)
    print("Starting classification...\n")
    print("\nClustering original data set with Phenograph...\n")
    norm_counts = doubletdetection.normalize_counts(raw_counts)
    pca = PCA(n_components=PCA_COMPONENTS)
    reduced_counts = pca.fit_transform(norm_counts)
    communities, graph, Q = phenograph.cluster(reduced_counts, k=KNN)
    print("Found these communities: {0}, with sizes: {1}".format(np.unique(communities), [np.count_nonzero(communities == i) for i in np.unique(communities)]))

    # Compute doublet scores for each point
    counts_w_doublets, scores_w_doublets, communities_w_doublets, doublet_labels, cutoff_rec = (
        DoubletDetection.classify(raw_counts, downsample=True, boost_rate=BOOST_RATE, k=KNN, n_pca=PCA_COMPONENTS))
    true_scores = scores_w_doublets[:raw_counts.shape[0]]
    for s in range(0, 100, 2):
        cutoff = s / float(100)
        test = true_scores[np.where(true_scores >= cutoff)[0]]
        print(cutoff, len(test))

    print("Suggested Cutoff Score: " + str(cutoff_rec))
    if math.isnan(cutoff_score):
        cutoff_score = cutoff_rec
    reduced_counts_w_doublets = pca.fit_transform(counts_w_doublets)

    #############################################################
    #
    # Generate Histograms of Original Data and Synthetic Doublets
    #
    #############################################################

    # Stacked histogram of number of original and synthetic data points per community
    fig1 = plt.figure(figsize=(12, 5), dpi=300)
    ax1 = plt.subplot(121)
    doublets = np.where(doublet_labels == 1)[0]
    raw_com_count = collections.Counter(communities_w_doublets[:raw_counts.shape[0]])
    doublet_com_count = collections.Counter(communities_w_doublets[doublets])

    labels, values = zip(*raw_com_count.items())
    indexes = np.arange(len(labels))
    width = 0.75
    plt.bar(indexes, values, width)

    fakes = []
    for com in labels:
        fakes.append(doublet_com_count[com])
    plt.bar(indexes, fakes, width, bottom = values)

    plt.xticks(indexes, labels)
    ax1.set_title("Original Data and Synthetic Doublets per Phenograph Cluster")

    # Histogram of scores per community
    ax2 = plt.subplot(122)
    labels, values = zip(*raw_com_count.items())
    indexes = np.arange(len(labels))
    width = 0.75

    scores = []
    for com in labels:
        score = np.unique(scores_w_doublets[np.where(communities_w_doublets == com)[0]])
        scores.append(score[0])

    plt.bar(indexes, scores, width)
    plt.xticks(indexes, labels)
    plt.axhline(np.floor(100*cutoff_score)/100, color='r', linestyle='dashed', linewidth=2, label='Cutoff Score')
    plt.legend()
    ax2.set_title("Doublet Score per Phenograph Cluster")
    plt.savefig('bars.png', dpi=fig1.dpi)

    ################################################################
    #
    # Generate Scatter Plots of Original Data and Synthetic Doublets
    #
    ################################################################

    fig2 = plt.figure(figsize=(18, 6), dpi=300)

    if tsne:
        print('\nCreating tSNE reduced counts from original data with synthetic doublets\n')
        tsne = TSNE(random_state=1)
        plot_counts_synth = tsne.fit_transform(reduced_counts_w_doublets)
    else:
        plot_counts_synth = reduced_counts_w_doublets

    # Axis 0: Original data with synthetic doublets
    ax = plt.subplot(132)
    set1i = LinearSegmentedColormap.from_list('set1i', plt.cm.Set1.colors, N=100)
    colors_synth = communities_w_doublets
    doublets_synth = np.where(doublet_labels == 1)[0]
    x_synth = plot_counts_synth[:, 0]
    y_synth = plot_counts_synth[:, 1]
    plt.scatter(x_synth, y_synth, c=colors_synth, cmap=set1i, s=5) 
    plt.scatter(x_synth[doublets_synth], y_synth[doublets_synth], facecolors='black', edgecolors='black', marker='d', s=2)
    ax.set_title("Original Data with Synthetic Doublets")
    if tsne:
        ax.set_xlabel("tSNE 1")
        ax.set_ylabel("tSNE 2")
    else:
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")

    # Axis 1: Original Data only 
    ax1 = plt.subplot(131)
    colors_1 = colors_synth[:raw_counts.shape[0]]
    x = x_synth[:raw_counts.shape[0]]
    y = y_synth[:raw_counts.shape[0]]
    plt.scatter(x, y, c=colors_1, cmap=set1i, s=5)
    ax1.set_title("Original Data")
    if tsne:
        ax1.set_xlabel("tSNE 1")
        ax1.set_ylabel("tSNE 2")
    else:
        ax1.set_xlabel("PCA 1")
        ax1.set_ylabel("PCA 2")

    # Axis 2: Heat map of Doublet Scores for original data only
    ax2 = plt.subplot(133)
    colors_2 = scores_w_doublets[:raw_counts.shape[0]]
    plt.scatter(x, y, c=colors_2, cmap='autumn_r', s=5)
    cb = plt.colorbar(aspect=50)
    ax2.set_title("Doublet Scores")
    if tsne:
        ax2.set_xlabel("tSNE 1")
        ax2.set_ylabel("tSNE 2")
        plt.savefig('tsne_synth.png', dpi=fig2.dpi)
    else:
        ax2.set_xlabel("PCA 1")
        ax2.set_ylabel("PCA 2")
        plt.savefig('pca_synth.png', dpi=fig2.dpi)

    if tsne:
        print('\nCreating tSNE reduced counts from original data only\n')
        tsne = TSNE(random_state=1)
        plot_counts = tsne.fit_transform(reduced_counts)
    else:
        plot_counts = reduced_counts
        
    #########################################
    #
    # Generate Scatter Plots of Original Data
    #
    #########################################

    fig3 = plt.figure(figsize=(12, 5), dpi=300)

    doublet_labels_og = doublet_labels
    doublet_labels_og[np.where(scores_w_doublets>=cutoff_score)[0]] = 1
    doublets_original = np.where(doublet_labels_og[:raw_counts.shape[0]] == 1)[0]

   	# Axis 0: Scatter plot of original data with PhenoGraph clustering
    ax = plt.subplot(121)
    x = plot_counts[:, 0]
    y = plot_counts[:, 1]
    plt.scatter(x, y, c=communities, cmap=set1i, s=4)
    ax.set_title("Original Data")
    if tsne:
        ax.set_xlabel("tSNE 1")
        ax.set_ylabel("tSNE 2")
    else:
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")

    # Axis 1: Heat map of doublet scores in original data
    a1 = plt.subplot(122)
    colors_1 = scores_w_doublets[:raw_counts.shape[0]]
    scatterplot = plt.scatter(x, y, c=colors_1, cmap='autumn_r', s=5)
    cb = plt.colorbar(aspect=50)
    ax1.set_title("Doublet Scores")
    if tsne:
        ax1.set_xlabel("tSNE 1")
        ax1.set_ylabel("tSNE 2")
        plt.savefig('tsne_original.png', dpi=fig3.dpi)
    else:
        ax1.set_xlabel("PCA 1")
        ax1.set_ylabel("PCA 2")
        plt.savefig('pca_original.png', dpi=fig3.dpi)
       		
if __name__ == '__main__':

    argv = sys.argv[1:]
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file", help='read csv data from FILE', metavar='FILE')
    parser.add_option("-c", type='float', dest="cutoff_score", default=float('nan'), help='cutoff score')
    parser.add_option("-t", dest="tsne", action="store_true", default=False)
    (options, _args) = parser.parse_args()
    if options.file:
        FNAME = options.file
    main(options.cutoff_score, ooptions.tsne)
