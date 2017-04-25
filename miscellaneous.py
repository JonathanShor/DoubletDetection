#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:35:38 2017

@author: jonathanshor
"""
import numpy as np
import phenograph
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import synthetic

DOUBLETRATE = synthetic.SYNTHDOUBLETRATE


# Supervised classification using sythetic data
def syntheticTesting(X_geneCounts, y_doubletLabels, useTruncSVD=False):
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
    synthetic.blockPrint()
    communities, graph, Q = phenograph.cluster(X_reduced_counts)
    synthetic.enablePrint()
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
