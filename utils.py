#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 3, 2017

@author: JonathanShor
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


# Standardize columns of matrix X: (X - X.mean) / X.std
# Also returns StandardScaler object for consistent further (inverse) standardization
# from sklearn.preprocessing import StandardScaler
def standardize(X):
    scaleX = StandardScaler().fit(X)
    return scaleX.transform(X), scaleX


# tf-idf normalizing: cells as documents, genes as words
# from sklearn.feature_extraction.text import TfidfTransformer
def normalize_tf_idf(X):
    if isinstance(X, pd.dataframe):
        X = X.as_matrix()

    tfidf = TfidfTransformer(norm=None, smooth_idf=True, sublinear_tf=False)
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
    raw_counts = raw_counts * median / cell_sums[:, np.newaxis]

    if doStandardize:
        # Take log and normalize to have mean 0 and std 1 in each gene across all cells
        raw_counts = np.log(raw_counts)

        # Normalize to have genes with mean 0 and std 1
        std = np.nanstd(raw_counts, axis=0)[np.newaxis, :]
        normed = (raw_counts - np.nanmean(raw_counts, axis=0)) / std
        # TODO: Use standardize() if we need to inverse or repeat stardardization
    else:
        normed = raw_counts

    # Replace NaN with 0
    normed = np.nan_to_num(normed)

    return normed


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
    precision, recall, f1_score = precision_recall_fscore_support(y_test, predictions)
    # probabilities = model.predict_proba(X)
    print ("{0} train set score: {1:.4g}".format(testName, model.score(X_train, y_train)))
    print ("{0} test set score: {1:.4g}".format(testName, model.score(X_test, y_test)))
    print("{0} test set precision: {1:.4g}".format(testName, precision))
    print("{0} test set recall: {1:.4g}".format(testName, recall))
    print("{0} test set f1 score: {1:.4g}".format(testName, f1_score))
    return model
    # TODO: return indices to recover train/test sets
