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
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from synthetic
from synthetic import DOUBLETRATE as SYNTHDOUBLETRATE
from synthetic import getCellTypes
import utils

FNAME = "~/Google Drive/Computational Genomics/pbmc8k_dense.csv"
DOUBLETRATE = SYNTHDOUBLETRATE


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

def analysisSuite(counts, doublet_label, usePCA=True):
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
    
    
def GMManalysisSuite(counts, doublet_labels):
    
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
    pca = PCA(n_components=50)
    reduced_counts = pca.fit_transform(counts)
    communities, graph, Q = phenograph.cluster(reduced_counts)
    
    labels = np.append(doublet_labels[:,np.newaxis], communities[:,np.newaxis], axis=1)    
    

if __name__ == '__main__':
    start_time = time.time()

    # Import counts
    raw_counts = utils.dataAcquisition(FNAME)
    # raw_counts = dataAcquisition(FNAME, normalize=True, useTFIDF=True)

    synthetic, labels = synthetic.create_synthetic_data(getCellTypes(raw_counts))

    # synthetic['labels'] = labels

    # pca = PCA(n_components=30)
    # synthetic.to_csv("/Users/adamgayoso/Google Drive/Computational Genomics/synthetic.csv")

    # syntheticTesting(synthetic.as_matrix(), labels)
    # analysisSuite(synthetic)
    
    # FROM OLD
    # Import counts
    # Normalize = False returns DataFrame
    raw_counts, _ = dataAcquisition(FNAME, normalize=False)
    
    synthetic, labels = create_synthetic_data(raw_counts, write=False, 0.5, 0.5)
    #synthetic, labels = dataAcquisition(SYN_FNAME, normalize=True, synthetic=True)
    perm = np.random.permutation(synthetic.shape[0])
    
    counts = synthetic[perm]
    doublet_labels = labels.as_matrix()
    doublet_labels = doublet_labels[perm]
    
    #synthetic = pd.read_csv(SYN_FNAME, index_col=0)
    #synthetic = synthetic.as_matrix()
    #doublet_label = synthetic[:, len(synthetic[0])-1]
    #syn_counts = synthetic[:,:len(synthetic[0])-1]

    
    #GMManalysisSuite(counts, doublet_labels)

    print("Total run time: {0:.2f} seconds".format(time.time() - start_time))
