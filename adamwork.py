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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# FNAME = "/Users/adamgayoso/Google Drive/Computational Genomics/pbmc8k_dense.csv"
FNAME = "/Users/jonathanshor/Google Drive/Computational Genomics/pbmc8k_dense.csv"
DOUBLETRATE = 0.07


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
def normalize_counts_10x(raw_counts, doStandardize=True):
    # Sum across cells and divide each cell by sum
    cell_sums = raw_counts.sum(axis=1).as_matrix()
    raw_counts = raw_counts.as_matrix()

    # Mutiply by median and divide by cell sum
    median = np.median(cell_sums)
    raw_counts = raw_counts*median/cell_sums[:, np.newaxis]

    if doStandardize:
        # Take log and normalize to have mean 0 and std 1 in each gene across all cells
        raw_counts = np.log(raw_counts)

        # Normalize to have genes with mean 0 and std 1
        std = np.nanstd(raw_counts, axis=0)[np.newaxis,:]
        normed = (raw_counts - np.nanmean(raw_counts, axis=0)) / std
        # TODO: Use standardize() if we need to inverse or repeat stardardization
    else:
        normed = raw_counts

    # Replace NaN with 0
    normed = np.nan_to_num(normed)

    return normed


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
    counts = pd.read_csv(FNAME, index_col=0)

    # Normalize
    if normalize:
        # Replacing with NaN makes it easier to ignore these values
        counts[counts == 0] = np.nan

        if useTFIDF:
            counts = normalize_tf_idf(counts)
        else:   # 10x paper normalization
            counts = normalize_counts_10x(counts)

    return counts


# Slow but works
def create_synthetic_data(raw_counts):

    synthetic = pd.DataFrame()

    cell_count = raw_counts.shape[0]
    doublet_rate = DOUBLETRATE
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


# Score model on a 0.2 test/train split of X, y.
# Returns fit model.
# Use randomState when you want to fix the train/test set split, and any random
# start aspects of model, for comparable repeat runs.
# from sklearn.model_selection import train_test_split
def testModel(model, X, y, testName, testSize=0.2, randomState=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize,
                                                        random_state=randomState)
    if hasattr(model, "random_state"):
        model.set_params(random_state=randomState)

    model.fit(X_train, y_train)
    # predictionss = model.predict(X)
    # probabilities = model.predict_proba(X)
    print ("{0} train set score: {1:.4g}".format(testName, model.score(X_train, y_train)))
    print ("{0} test set score: {1:.4g}".format(testName, model.score(X_test, y_test)))
    return model #, predictions, probabilities
    # TODO: return indices to recover train/test sets


def analysisSuite(counts, usePCA=True):
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

    # Gaussian Mixture Model
    pca = PCA(n_components=50)
    GM_data = pca.fit_transform(raw_counts)
    predictionsGM, probabilitiesGM = gaussian_mixture(GM_data, labels)

    # KNN outlier detection
    distances = knn(GM_data, labels)
    far = distances[0][:,9]


# Supervised classification using sythetic data
def syntheticTesting(X_geneCounts, y_doubletLabels, useTruncSVD=False):
#    X_standardized = normalize_counts_10x(X_geneCounts)

    # Naive classifier test: library size
    librarySize = X_geneCounts.sum(axis=1).reshape(-1, 1)
    #librarySizeSt = X_standardized.sum(axis=1).reshape(-1, 1)
    print("librarySize.shape: ", (librarySize.shape))
#    print("X_stardardized.shape: ", (X_stardardized.shape))
    print("y_doubletLabels.shape: ", (y_doubletLabels.shape))
    testModel(BernoulliNB(), librarySize, y_doubletLabels, 'Library size; NBB')
#    testModel(BernoulliNB(), librarySizeSt, y_doubletLabels, 'Library size; NBB, standardized X')
    # clfGMM = GaussianMixture(n_components=2, weights_init=[1 - DOUBLETRATE, DOUBLETRATE])
    # testModel(clfGMM, librarySize, y_doubletLabels, 'Library size; GMM')

    numGenesExpressed = np.count_nonzero(X_geneCounts, axis=1).reshape(-1, 1)
    testModel(BernoulliNB(), numGenesExpressed, y_doubletLabels, 'Unique Genes; NBB')

    if useTruncSVD:
        print("TODO: Actually implement TruncatedSVD")
    else:
        pca = PCA(n_components=30)
        X_reduced_counts = pca.fit_transform(X_geneCounts)

    # Run Phenograph
    communities, graph, Q = phenograph.cluster(X_reduced_counts)
    for communityID in np.unique(communities):
        X_community = X_geneCounts[communities == communityID]
        y_community = y_doubletLabels[communities == communityID]

        librarySize = X_community.sum(axis=1).reshape(-1, 1)
        testModel(BernoulliNB(), librarySize, y_community,
                  "Community {} Library size; NBB".format(communityID))

        numGenesExpressed = np.count_nonzero(X_community, axis=1).reshape(-1, 1)
        testModel(BernoulliNB(), numGenesExpressed, y_community,
                  "Community {} Unique Genes; NBB".format(communityID))

        clfGMM = GaussianMixture(n_components=2, weights_init=[1 - DOUBLETRATE, DOUBLETRATE])
        testModel(clfGMM, X_community, y_doubletLabels[communities == communityID],
                  "Community {}; GMM".format(communityID))


if __name__ == '__main__':

    # Import counts
    raw_counts = dataAcquisition(FNAME)
    # raw_counts = dataAcquisition(FNAME, normalize=True, useTFIDF=True)

    synthetic, labels = create_synthetic_data(raw_counts)

    # synthetic['labels'] = labels

    # synthetic.to_csv("/Users/adamgayoso/Google Drive/Computational Genomics/synthetic.csv")

    syntheticTesting(synthetic.as_matrix(), labels)
    # analysisSuite(synthetic)
