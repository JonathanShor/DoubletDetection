#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:35:38 2017

@author: adamgayoso
"""
import pandas as pd
import numpy as np
import phenograph
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import BernoulliNB
from utils import testModel
from utils import normalize_counts_10x
import collections
from synthetic import create_synthetic_data
from synthetic import create_simple_synthetic_data
from synthetic import DOUBLETRATE as SYNTHDOUBLETRATE
from synthetic import getCellTypes
import utils
from utils import testModel
from classifiers import *
from sklearn.model_selection import train_test_split

FNAME = "~/Google Drive/Computational Genomics/pbmc8k_dense.csv"
SYN_FNAME = "~/Google Drive/Computational Genomics/synthetic.csv"

# OLD STUFF

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


def GMManalysis(counts, doublet_labels, includePCA=False):
    # Gaussian Mixture Model
    library_size = counts.sum(axis=1)[:,np.newaxis]
    num_genes = np.count_nonzero(counts, axis=1)[:,np.newaxis]
    
    if includePCA:
        pca = PCA(n_components=10)
        reduced_counts = pca.fit_transform(counts)
        features = np.concatenate((library_size, num_genes, reduced_counts), axis=1)
    else:   
        features = np.concatenate((library_size, num_genes), axis=1)
    
    # Changing weights based on method possible
    predictionsGM, probabilitiesGM = gaussian_mixture(features, weights=[0.93,0.07])
    
    # Error rates
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
        counts, doublet_labels = create_synthetic_data(getCellTypes(raw_counts, PCA_components=50, shrink=0))

    else: 
        
        #Simple synthetic data
        synthetic, doublet_labels = create_simple_synthetic_data(raw_counts, 0.7, 0.7, normalize=True, doublet_rate=0.4)
        #synthetic, doublet_labels = utils.synthAcquisition(SYN_FNAME, normalize=True)
        #synthetic = synthetic.as_matrix()
        perm = np.random.permutation(synthetic.shape[0])
    
        counts = synthetic[perm]
        doublet_labels = doublet_labels[perm]
        
        

        GMManalysis(counts, doublet_labels)

    print("Total run time: {0:.2f} seconds".format(time.time() - start_time))


# Supervised classification using sythetic data
def syntheticTesting(X_geneCounts, y_doubletLabels, useTruncSVD=False):
    try:
        X_geneCounts = X_geneCounts.as_matrix()
    except AttributeError:
        pass
    try:
        y_doubletLabels = y_doubletLabels.as_matrix()
    except AttributeError:
        pass

    # X_standardized = normalize_counts_10x(X_geneCounts)

    # Naive classifier test: library size
    librarySize = np.sum(X_geneCounts, axis=1).reshape(-1, 1)
    # librarySizeSt = X_standardized.sum(axis=1).reshape(-1, 1)
    print("librarySize.shape: {}".format(librarySize.shape))
    # print("X_stardardized.shape: ", (X_stardardized.shape))
    print("y_doubletLabels.shape: {}".format(y_doubletLabels.shape))
    testModel(BernoulliNB(), librarySize, y_doubletLabels, 'Library size; NBB')
    # testModel(BernoulliNB(), librarySizeSt, y_doubletLabels, 'Library size; NBB, standardized X')
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
    blockPrint()
    communities, graph, Q = phenograph.cluster(X_reduced_counts)
    enablePrint()
    print("Found these communities: {0}, with sizes: {1}".format(np.unique(communities),
          [np.count_nonzero(communities == i) for i in np.unique(communities)]))

    for communityID in np.unique(communities):
        X_community = X_geneCounts[communities == communityID]
        X_reduced_community = X_reduced_counts[communities == communityID]
        y_community = y_doubletLabels[communities == communityID]
        print("Community {0}: {1} cells".format(communityID, len(y_community)))

        librarySize = X_community.sum(axis=1).reshape(-1, 1)
        testModel(BernoulliNB(), librarySize, y_community,
                  "Community {} Library size; NBB".format(communityID))

        numGenesExpressed = np.count_nonzero(X_community, axis=1).reshape(-1, 1)
        testModel(BernoulliNB(), numGenesExpressed, y_community,
                  "Community {} Unique Genes; NBB".format(communityID))

        clfGMM = GaussianMixture(n_components=2, weights_init=[1 - DOUBLETRATE, DOUBLETRATE])
        testModel(clfGMM, X_reduced_community, y_community,
                  "Community {}; GMM".format(communityID))
        
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
    predictions = model.predict(X_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predictions,
                                                                     average='micro')
    # probabilities = model.predict_proba(X)
    print ("{0} train set score: {1:.4g}".format(testName, model.score(X_train, y_train)))
    print ("{0} test set score: {1:.4g}".format(testName, model.score(X_test, y_test)))
    print("{0} test set precision: {1:.4g}".format(testName, precision))
    print("{0} test set recall: {1:.4g}".format(testName, recall))
    print("{0} test set f1 score: {1:.4g}".format(testName, f1_score))
    return model
    # TODO: return indices to recover train/test sets

