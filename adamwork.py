#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:55:34 2017

@author: adamgayoso
"""

import pandas as pd
import numpy as np
import phenograph
import collections
from sklearn.decomposition import PCA
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Takes PD dataframe
# Following method in 10x paper
def normalize_counts_10x(raw_counts):
    
    # Grab cells and delete from DF
    cells = raw_counts['Unnamed: 0']
    del raw_counts['Unnamed: 0']
    
    # Sum across cells and divide each cell by sum
    cell_sums = raw_counts.sum(axis=1).as_matrix()
    raw_counts = raw_counts.as_matrix()
    
    # Mutiply by median and divide by cell sum
    median = np.median(cell_sums)
    raw_counts = raw_counts*median/cell_sums[:, np.newaxis]

    # Take log and normalize to have mean 0 and std 1 in each gene across all cells
    raw_counts = np.log(raw_counts)
    
    # Normalize to have genes with mean 0 and std 1
    std = np.nanstd(raw_counts, axis=0)[np.newaxis,:]
    normed = (raw_counts - np.nanmean(raw_counts, axis=0)) / std
          
    # Replace NaN with 0
    normed = np.nan_to_num(normed)         
    
    return cells, normed


# Elementary modeling
def naive_bayes_bernoulli(counts, labels):
    
    clf = BernoulliNB()
    clf.fit(counts, labels)
    predictions = clf.predict(counts)
    probabilities = clf.predict_proba(counts)
    
    return predictions, probabilities

# Elementary modeling
def naive_bayes_gauss(counts, labels):
    
    clf = GaussianNB()
    clf.fit(counts, labels)
    predictions = clf.predict(counts)
    probabilities = clf.predict_proba(counts)
    
    return predictions, probabilities

# Elementary modeling
def naive_bayes_multinomial(counts, labels):
    
    clf = MultinomialNB()
    clf.fit(counts, labels)
    predictions = clf.predict(counts)
    probabilities = clf.predict_proba(counts)
    
    return predictions, probabilities

# Elementary modeling
def gaussian_mixture(counts, labels):
    
    clf = GaussianMixture(n_components=2, weights_init = [0.93, 0.07])
    clf.fit(counts)
    predictions = clf.predict(counts)
    probabilities = clf.predict_proba(counts)
    
    return predictions, probabilities

def knn(counts, labels):
    
    
    clf = NearestNeighbors(n_neighbors=10)
    clf.fit(counts)
    clf.kneighbors(counts, 10)
    
    return clf.kneighbors(counts, 10)[0]


def main():
    
    # Import counts 
    raw_counts = pd.read_csv("/Users/adamgayoso/Google Drive/Computational Genomics/pbmc8k_dense.csv")
    
    # Replacing with NaN makes it easier to ignore these values
    counts[raw_counts == 0] = np.nan

    # Normalize           
    cells, counts = normalize_counts_10x(counts)
    
    # Dimensionality reduction
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
    
    
    # Gaussian Mixture Model
    pca = PCA(n_components=50)
    GM_data = pca.fit_transform(raw_counts)
    predictionsGM, probabilitiesGM = gaussian_mixture(GM_data, labels)
    
    # KNN outlier detection
    distances = knn(GM_data, labels)
    far = distances[0][:,9]
  
    
def main2():
    
    # Import counts 
    raw_counts = pd.read_csv("/Users/adamgayoso/Google Drive/Computational Genomics/pbmc8k_dense.csv", index_col=0)
    
    synthetic, labels = create_synthetic_data(raw_counts)
    
    synthetic['labels'] = labels
    
    synthetic.to_csv("/Users/adamgayoso/Google Drive/Computational Genomics/synthetic.csv")
    
    
    
# Slow but works
def create_synthetic_data(raw_counts):
    
    synthetic = pd.DataFrame()
    
    cell_count = raw_counts.shape[0]
    doublet_rate = 0.07
    doublets = int(doublet_rate*cell_count/(1-doublet_rate))
    
    # Add labels column to know which ones are doublets
    labels = np.zeros(cell_count + doublets)
    labels[cell_count:] = 1
    
    
    for i in range(doublets):
        row1 = int(np.random.rand()*cell_count)
        row2 = int(np.random.rand()*cell_count)
        
        new_row = raw_counts.iloc[row1] + raw_counts.iloc[row2]
        
        synthetic = synthetic.append(new_row, ignore_index=True)
    
    return raw_counts.append(synthetic), labels

main2()
    
    
     
    