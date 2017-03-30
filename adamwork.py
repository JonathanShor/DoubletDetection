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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# FNAME = "/Users/adamgayoso/Google Drive/Computational Genomics/pbmc8k_dense.csv"
FNAME = "/Users/jonathanshor/Google Drive/Computational Genomics/pbmc8k_dense.csv"


# Standardize columns of matrix X: (X - X.mean) / X.std
# Also returns StandardScaler object for consistent further (inverse) standardization
#from sklearn.preprocessing import StandardScaler
def standardize(X):
    scaleX = StandardScaler().fit(X)
    return scaleX.transform(X), scaleX


# tf-idf normalizing: cells as documents, genes as words
#from sklearn.feature_extraction.text import TfidfTransformer
def normalize_tf_idf(X):
    if isinstance(X, pd.dataframe):
        X = X.as_matrix()

    tfidf = TfidfTransformer(norm=None,smooth_idf=True, sublinear_tf=False)
    tfidf.fit(X)
    return tfidf.transform(X)


# Takes PD dataframe
# Following method in 10x paper
def normalize_counts_10x(raw_counts):
    # Grab cells and delete from DF
    cells = raw_counts.index
#    del raw_counts['Unnamed: 0']   # Assumed to be in index, not data column, see pd.read_csv option

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
    # TODO: Use standardize() if we need to inverse or repeat stardardization

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


def dataAcquisition(FNAME, normalize=False, useTFIDF=False):
    # Import counts
    raw_counts = pd.read_csv(FNAME, index_col=0)

    # Replacing with NaN makes it easier to ignore these values
    counts = raw_counts.copy()
    counts[raw_counts == 0] = np.nan

    # Normalize
    if normalize:
        if useTFIDF:
            counts = normalize_tf_idf(counts)
        else:   # 10x paper normalization
            _, counts = normalize_counts_10x(counts)
        return counts
    else:
        return counts


# Slow but works
def create_synthetic_data(raw_counts):

    cell_count = raw_counts.shape[0]
    doublet_rate = 0.07
    doublets = doublet_rate*cell_count/(1-doublet_rate)

    for i in range(int(doublets)):
        row1 = int(np.random.rand()*cell_count)
        row2 = int(np.random.rand()*cell_count)

        new_row = raw_counts.iloc[row1] + raw_counts.iloc[row2]

        raw_counts = raw_counts.append(new_row, ignore_index=True)

    return raw_counts


def analysisSuite(counts):
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


if __name__ == '__main__':

    # Import counts
    raw_counts = dataAcquisition(FNAME)
    # raw_counts = dataAcquisition(FNAME, normalize=True, useTFIDF=True)

    synthetic = create_synthetic_data(raw_counts)

    # synthetic.to_csv("/Users/adamgayoso/Google Drive/Computational Genomics/synthetic.csv")

    analysisSuite(synthetic)
