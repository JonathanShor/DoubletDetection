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
from synthetic import DOUBLETRATE as SYNTHDOUBLETRATE
from synthetic import getCellTypes
import utils
from classifiers import *

PCA_COMPONENTS = 30

def classify(FNAME, probabilistic = False):
    """
    Classifier for doublets in single-cell RNA-seq data
    :param FNAME: path to csv file containing data
    :param probabilistic: option to use sampled doublets vs linear doublets
    :return raw_counts: raw_counts in numpy ndarray format
    :return scores: doublet score for each row in test
    """
    
    # Import counts
    # Normalize = False returns DataFrame
    raw_counts = utils.dataAcquisition(FNAME, normalize=False)
    
    probabilistic = False
    
    if probabilistic:
        
        # Probabilistic synthetic data
        counts, doublet_labels = create_synthetic_data(getCellTypes(raw_counts, PCA_components=50, shrink=0))

    else: 
        
        #Simple synthetic data
        synthetic, doublet_labels = create_simple_synthetic_data(raw_counts, 0.7, 0.7, normalize=True, doublet_rate=0.15)
        perm = np.random.permutation(synthetic.shape[0])    
        counts = synthetic[perm]
        doublet_labels = doublet_labels[perm]
        
    # Get phenograph results  
    pca = PCA(n_components=PCA_COMPONENTS)
    reduced_counts = pca.fit_transform(counts)
    communities, graph, Q = phenograph.cluster(reduced_counts)
    c_count = collections.Counter(communities)


    phenolabels = np.append(communities[:,np.newaxis], doublet_labels[:,np.newaxis], axis=1)
    
    synth_doub_count = {}
    score = np.zeros((len(communities), 1))
    for c in np.unique(communities):
        c_indices = np.where(phenolabels[:,0] == c)[0]
        synth_doub_count[c] = np.sum(phenolabels[c_indices,1])/float(c_count[c])
        score[c_indices] = synth_doub_count[c]
        
    # Reordering the scores to back out the original permutation
    order = np.argsort(perm) 
    score = score[order]
    # Only keep scores for real points
    score = score[:raw_counts.shape[0],:]
    
    
    return raw_counts, scores
    
    
    
#classify(FNAME)