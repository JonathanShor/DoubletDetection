#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:54:34 2017

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
    print("Loading data...\n")
    raw_counts = utils.load_data(FNAME)

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
        test_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5]
        cutoffs = np.zeros((50,len(test_rates)))
        i = 0
        for D in test_rates:
            counts_w_doublets, scores_w_doublets, communities_w_doublets, doublet_labels = (
                doubletdetection.classify(raw_counts, probabilistic=False, doublet_rate=D))
            true_scores = scores_w_doublets[:raw_counts.shape[0], :]
            for s in range(0, 100, 2):
                cutoff = s / float(100)
                test = true_scores[np.where(true_scores > cutoff)[0]]
                cutoffs[int(s/2),i] = len(test)
            i+=1
        x = np.zeros((50))
        for s in range(0,100,2):
            x[int(s/2)] = (int(s))
            
    # Plotting CDF for different cutoffs
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    for i in range(len(test_rates)):
        ax.plot(x[1:], cutoffs[1:,i], label=test_rates[i])
    ax.legend()
        
    graph = False
    if graph:
        # Visualize tSNE clustering
        # Default tsne num_componenets is 2
    
        pca = PCA(n_components=30)
        reduced_counts = pca.fit_transform(counts_w_doublets)
    
        communities, graph, Q = phenograph.cluster(reduced_counts)
    
        print('\nCreating tSNE reduced counts\n')
        tsne = TSNE(random_state=1)
        tsne_counts = tsne.fit_transform(reduced_counts)
    
        #cutoff = 0.59
        #doublet_labels = np.zeros((reduced_counts.shape[0],))
        #doublet_labels[np.where(scores>0.37)[0]] = 1
        
        # data viz
        #fig = plt.figure(figsize=(8, 8), dpi=300)
        set1i = LinearSegmentedColormap.from_list('set1i', plt.cm.Set1.colors, N=100)
        f, (ax1, ax2, ax3) = plt.subplots(1,3,sharey=True,figsize=(20, 8), dpi=300)
        # Sampled doublet tsne
        colors = communities_w_doublets
        x = tsne_counts[:, 0]
        y = tsne_counts[:, 1]
        ax1.scatter(x, y, c=colors, s=10, cmap=set1i)
        doublets = np.where(doublet_labels == 1)[0]
        ax1.scatter(x[doublets], y[doublets], facecolors='none', edgecolors='black', s=2)
        # If the data was from a mixed classification
        if len(np.unique(doublet_labels)) == 3:
            doublets = np.where(doublet_labels == 2)[0]
            ax1.scatter(x[doublets], y[doublets], facecolors='none', edgecolors='red', s=2)
        ax1.set_title("Fake and Real Data with Black Fake Doublets")
        
        # Original Data
        #fig = plt.figure(figsize=(8, 8), dpi=300)
        colors = communities_w_doublets[:raw_counts.shape[0]]
        ax2.scatter(x[:raw_counts.shape[0]], y[:raw_counts.shape[0]], c=colors, s=10, cmap=set1i)
        ax2.set_title("Original Data")
        
        # Original marked doublets
        #fig = plt.figure(figsize=(8, 8), dpi=300)
        ax3.scatter(x[:raw_counts.shape[0]], y[:raw_counts.shape[0]], c=colors, s=10, cmap=set1i)
        cutoff = 0.5
        doublets = np.where(scores_w_doublets[:raw_counts.shape[0]]>cutoff)[0]
        ax3.scatter(x[doublets], y[doublets], facecolors='none', edgecolors='black', s=2)
        ax3.set_title("Identified doublets with Score>" + str(cutoff))
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
