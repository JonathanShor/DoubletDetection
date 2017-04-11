#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:55:34 2017

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

FNAME = "~/Google Drive/Computational Genomics/pbmc8k_dense.csv"
SYN_FNAME = "~/Google Drive/Computational Genomics/synthetic.csv"
DOUBLETRATE = SYNTHDOUBLETRATE


# This analysis needs work, this is an old version and might not work
def basic_analysis(counts, doublet_label, usePCA=True):
    # Dimensionality reduction
    if usePCA:
        pca = PCA(n_components=30)
        reduced_counts = pca.fit_transform(counts)

    # Run Phenograph
    communities, graph, Q = phenograph.cluster(reduced_counts)

    # Show distribution
    collections.Counter(communities)

    # Bernoulli Naive Bayes with labels from reduced data
    labels = communities
    predictions, probabilities = naive_bayes_bernoulli(counts, labels)

    probabilities.sort()
    max_2_values = probabilities[:,len(probabilities[0])-2:]

    # Entries where second max is greater than arbitrary number
    # Check out entropy of values
    outliers = max_2_values[max_2_values[:,0] > 0.005]


    # Gauss Naive Bayes with labels from reduced data
    predictionsG, probabilitiesG = naive_bayes_gauss(counts, labels)

    probabilitiesG.sort()
    max_2_valuesG = probabilitiesG[:,len(probabilitiesG[0])-2:]

    # Entries where second max is greater than arbitrary number
    # Check out entropy of values
    outliersG = max_2_valuesG[max_2_valuesG[:,0] > 0.0005]


    # Multinomial Naive Bayes with labels from reduced data
    frequencies = np.nan_to_num(raw_counts)
    predictionsM, probabilitiesM = naive_bayes_multinomial(raw_counts, labels)

    probabilitiesM.sort()
    max_2_valuesM = probabilitiesM[:,len(probabilitiesM[0])-2:]

    # Entries where second max is greater than arbitrary number
    # Check out entropy of values
    outliersM = max_2_valuesM[max_2_valuesM[:,0] > 0.0005]

    # KNN outlier detection
    distances = knn(GM_data, labels)
    far = distances[0][:,9]


def GMManalysis(counts, doublet_labels):
    # Gaussian Mixture Model
    library_size = counts.sum(axis=1)[:,np.newaxis]
    num_genes = np.count_nonzero(counts, axis=1)[:,np.newaxis]
    features = np.concatenate((library_size, num_genes), axis=1)
    predictionsGM, probabilitiesGM = gaussian_mixture(features)
    GMM_error1 = (len(doublet_labels) - np.sum(doublet_labels==predictionsGM))/len(doublet_labels)
    GMM_error2 = (len(doublet_labels) - np.sum(doublet_labels==(1-predictionsGM)))/len(doublet_labels)
    
    print("Test error over all cells = ")
    print(np.min([GMM_error1, GMM_error2]))
    
    # False positives, negative
    doublets = np.where(doublet_labels == 1)[0]
    GMM_error1 = (len(doublet_labels[doublets]) - np.sum(doublet_labels[doublets]==predictionsGM[doublets]))/len(doublet_labels[doublets])
    GMM_error2 = (len(doublet_labels[doublets]) - np.sum(doublet_labels[doublets]==(1-predictionsGM[doublets])))/len(doublet_labels[doublets])
    print("Test error over synthetic doublets = ")
    print(np.min([GMM_error1, GMM_error2]))
    
    # Attempting to do it within each phenograph cluster
    #pca = PCA(n_components=50)
    #reduced_counts = pca.fit_transform(counts)
    #communities, graph, Q = phenograph.cluster(reduced_counts)
    
    #labels = np.append(doublet_labels[:,np.newaxis], communities[:,np.newaxis], axis=1)    
    

if __name__ == '__main__':
    start_time = time.time()

    # Import counts
    # Normalize = False returns DataFrame
    raw_counts = utils.dataAcquisition(FNAME, normalize=False)
    
    probabilistic = False
    
    if probabilistic:
        
        # Probabilistic synthetic data
        synthetic, doublet_labels = create_synthetic_data(getCellTypes(raw_counts))
    else: 
        
        #Simple synthetic data
        #synthetic, doublet_labels = create_simple_synthetic_data(raw_counts, 1, 1)
        synthetic, doublet_labels = utils.synthAcquisition(SYN_FNAME, normalize=True)
        perm = np.random.permutation(synthetic.shape[0])
    
        counts = synthetic[perm]
        doublet_labels = doublet_labels[perm]
        
        
    GMManalysis(counts, doublet_labels)

    print("Total run time: {0:.2f} seconds".format(time.time() - start_time))
