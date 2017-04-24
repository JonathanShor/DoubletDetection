#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:45:28 2017

@author: adamgayoso, JonathanShor, ryanbrand
"""

import pandas as pd
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
import utils
from classifiers import *

PCA_COMPONENTS = 30
DOUBLET_RATE = 0.25
KNN=20

def classify(raw_counts, probabilistic = False):
    """
    Classifier for doublets in single-cell RNA-seq data
    :param raw_counts: count table in pandas DF format
    :param probabilistic: option to use sampled doublets vs linear doublets
    :return counts: mixed counts (real and fake) in numpy ndarray format NORMALIZED
    :return scores: doublet score for each row in counts
    :return communities: Phenograph community for each row in counts
    :return doublet_labels: indicator for each row in counts whether it is a fake doublet (doublets appended to end)
    """
    
    if probabilistic == True:
        #Probabilistc doublets
        print("Gathering info about cell types...\n")
        cell_types = getCellTypes(raw_counts, PCA_components=PCA_COMPONENTS, shrink=0.01, knn=KNN)
        
        print("\nAdding fake doublets to data set...\n")
        doublets = np.zeros((int(DOUBLET_RATE*raw_counts.shape[0]), raw_counts.shape[1]))
        for i in range(int(DOUBLET_RATE*raw_counts.shape[0])):
            doublets[i] = doubletFromCelltype(cell_types)  
        
        synthetic = np.append(counts, doublets, axis=0)
        synthetic = utils.normalize_counts_10x(synthetic)

        doublet_labels = np.zeros((int(raw_counts.shape[0]*(1+DOUBLET_RATE)),))
        doublet_labels[raw_counts.shape[0]:] = 1
    else: 
        # Simple synthetic data
        # Requires pd DataFrame
        synthetic, doublet_labels = create_simple_synthetic_data(raw_counts, 0.7, 0.7, normalize=True, doublet_rate=DOUBLET_RATE)
    
    counts = synthetic
    
    print("\nClustering mixed data set with Phenograph...\n")
    # Get phenograph results  
    pca = PCA(n_components=PCA_COMPONENTS)
    reduced_counts = pca.fit_transform(counts)
    communities, graph, Q = phenograph.cluster(reduced_counts, k=KNN)
    c_count = collections.Counter(communities)
    print('\n')

    # Count number of fake doublets in each community and assign score
    phenolabels = np.append(communities[:,np.newaxis], doublet_labels[:,np.newaxis], axis=1)
    
    synth_doub_count = {}
    scores = np.zeros((len(communities), 1))
    for c in np.unique(communities):
        c_indices = np.where(phenolabels[:,0] == c)[0]
        synth_doub_count[c] = np.sum(phenolabels[c_indices,1])/float(c_count[c])
        scores[c_indices] = synth_doub_count[c]
        
    # Only keep scores for real points
    #scores = scores[:raw_counts.shape[0],:]
    #communities = communities[order]
    #communities = communities[:raw_counts.shape[0]]

    
    return counts, scores, communities, doublet_labels


def validate(raw_counts):
    """
    Validate methodology using only probabilistic synthetic data
    :param raw_counts: count table in pandas DF format    
    :return synthetic: synthetic normalized counts
    :return scores: doublet score for each row in test
    :return true_doublet_labels: validation "true doublets"
    :return fake_doublet_labels: appended fake doublets
    """

    # Probabilistic synthetic data
    print("Getting cell types...")
    cell_types = getCellTypes(raw_counts, PCA_components=PCA_COMPONENTS, shrink=0.01, knn=KNN)
    counts, true_doublet_labels = create_synthetic_data(cell_types)
        
    print("Creating new doublets")
    # Recreating cell types to reflect the new data set
    cell_types = getCellTypes(counts, PCA_components=PCA_COMPONENTS, shrink=0.01, knn=KNN)
    doublets = np.zeros((int(DOUBLET_RATE*counts.shape[0]), counts.shape[1]))
    doublet_labels = np.zeros((int(counts.shape[0]*(1+DOUBLET_RATE)),))
    doublet_labels[counts.shape[0]:] = 1
        
    for i in range(int(DOUBLET_RATE*counts.shape[0])):
        doublets[i] = doubletFromCelltype(cell_types)
            
    synthetic = np.append(counts, doublets, axis=0)
    synthetic = utils.normalize_counts_10x(synthetic)

        
    # Get phenograph results  
    pca = PCA(n_components=PCA_COMPONENTS)
    reduced_counts = pca.fit_transform(synthetic)
    communities, graph, Q = phenograph.cluster(reduced_counts, k=KNN)
    c_count = collections.Counter(communities)

    # Count number of fake doublets in each community and assign score
    phenolabels = np.append(communities[:,np.newaxis], doublet_labels[:,np.newaxis], axis=1)
    
    synth_doub_count = {}
    scores = np.zeros((len(communities), 1))
    for c in np.unique(communities):
        c_indices = np.where(phenolabels[:,0] == c)[0]
        synth_doub_count[c] = np.sum(phenolabels[c_indices,1])/float(c_count[c])
        scores[c_indices] = synth_doub_count[c]
        
    # Only keep scores for real points
    #scores = scores[:raw_counts.shape[0],:]
    #communities = communities[order]
    #communities = communities[:raw_counts.shape[0]]
    fake_doublet_labels = doublet_labels
    
    return synthetic, scores, communities, true_doublet_labels, fake_doublet_labels