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
DOUBLET_RATE = 0.15

def classify(raw_counts, probabilistic = False):
    """
    Classifier for doublets in single-cell RNA-seq data
    :param raw_counts: count table in pandas DF format
    :param probabilistic: option to use sampled doublets vs linear doublets
    :return raw_counts: raw_counts in numpy ndarray format
    :return scores: doublet score for each row in test
    """
    
    # Import counts
    # Normalize = False returns DataFrame
    
    if probabilistic == True:
        #Probabilistc doublets
        cell_types = getCellTypes(raw_counts, PCA_components=PCA_COMPONENTS, shrink=0.01)
        doublets = np.zeros((int(DOUBLET_RATE*raw_counts.shape[0]), raw_counts.shape[1]))
        for i in range(int(DOUBLET_RATE*raw_counts.shape[0])):
            doublets[i] = doubletFromCelltype(cell_types)  
        
        counts = raw_counts.as_matrix()
        synthetic = np.append(counts, doublets, axis=0)
        synthetic = utils.normalize_counts_10x(synthetic)

        doublet_labels = np.zeros((int(raw_counts.shape[0]*(1+DOUBLET_RATE)),))
        doublet_labels[raw_counts.shape[0]:] = 1
    else: 
        # Simple synthetic data
        # Requires pd DataFrame
        synthetic, doublet_labels = create_simple_synthetic_data(raw_counts, 0.7, 0.7, normalize=True, doublet_rate=DOUBLET_RATE)
    
    # Shuffle data
    perm = np.random.permutation(synthetic.shape[0])    
    counts = synthetic[perm]
    doublet_labels = doublet_labels[perm]
        
    # Get phenograph results  
    pca = PCA(n_components=PCA_COMPONENTS)
    reduced_counts = pca.fit_transform(counts)
    communities, graph, Q = phenograph.cluster(reduced_counts)
    c_count = collections.Counter(communities)

    # Count number of fake doublets in each community and assign score
    phenolabels = np.append(communities[:,np.newaxis], doublet_labels[:,np.newaxis], axis=1)
    
    synth_doub_count = {}
    scores = np.zeros((len(communities), 1))
    for c in np.unique(communities):
        c_indices = np.where(phenolabels[:,0] == c)[0]
        synth_doub_count[c] = np.sum(phenolabels[c_indices,1])/float(c_count[c])
        scores[c_indices] = synth_doub_count[c]
        
    # Reordering the scores to back out the original permutation
    order = np.argsort(perm) 
    scores = scores[order]
    # Only keep scores for real points
    scores = scores[:raw_counts.shape[0],:]
    
    
    return raw_counts.as_matrix().astype(np.float64), scores, communities[:raw_counts.shape[0]]



def validate(raw_counts):
    """
    Validate methodology using only probabilistic synthetic data
    :param raw_counts: count table in pandas DF format    
    :return raw_counts: raw_counts in numpy ndarray format
    :return scores: doublet score for each row in test
    """

    
    # Probabilistic synthetic data
    print("Getting cell types...")
    cell_types = getCellTypes(raw_counts, PCA_components=PCA_COMPONENTS, shrink=0.01)
    counts, true_doublet_labels = create_synthetic_data(cell_types)
        
    print("Creating new doublets")
    doublets = np.zeros((int(DOUBLET_RATE*raw_counts.shape[0]), raw_counts.shape[1]))
    doublet_labels = np.zeros((int(raw_counts.shape[0]*(1+DOUBLET_RATE)),))
    doublet_labels[raw_counts.shape[0]:] = 1
        
    for i in range(int(DOUBLET_RATE*raw_counts.shape[0])):
        doublets[i] = doubletFromCelltype(cell_types)
            
    synthetic = np.append(counts, doublets, axis=0)
    synthetic = utils.normalize_counts_10x(synthetic)
    
    # Shuffle data
    perm = np.random.permutation(synthetic.shape[0])    
    counts = synthetic[perm]
    doublet_labels = doublet_labels[perm]
        
    # Get phenograph results  
    pca = PCA(n_components=PCA_COMPONENTS)
    reduced_counts = pca.fit_transform(counts)
    communities, graph, Q = phenograph.cluster(reduced_counts)
    c_count = collections.Counter(communities)

    # Count number of fake doublets in each community and assign score
    phenolabels = np.append(communities[:,np.newaxis], doublet_labels[:,np.newaxis], axis=1)
    
    synth_doub_count = {}
    scores = np.zeros((len(communities), 1))
    for c in np.unique(communities):
        c_indices = np.where(phenolabels[:,0] == c)[0]
        synth_doub_count[c] = np.sum(phenolabels[c_indices,1])/float(c_count[c])
        scores[c_indices] = synth_doub_count[c]
        
    # Reordering the scores to back out the original permutation
    order = np.argsort(perm) 
    scores = scores[order]
    # Only keep scores for real points
    scores = scores[:raw_counts.shape[0],:]
    
    
    return raw_counts.as_matrix().astype(np.float64), scores, communities[:raw_counts.shape[0]], true_doublet_labels