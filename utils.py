#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 3, 2017

@author: adamgayoso, JonathanShor, ryanbrand
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from os.path import expanduser


# To read only the first X rows, set read_rows=X
def dataAcquisition(FNAME, normalize=False, read_rows=None):
    # Import counts
    counts = np.loadtxt(expanduser(FNAME), delimiter=",", skiprows=1)
    counts = counts[:read_rows, 1:]  # Peel off pandas index labels column

    # Normalize
    if normalize:
        
        if normalize == "TFIDF":
            counts = normalize_tf_idf(counts)
        else:   # 10x paper normalization
            counts = normalize_counts(counts)

    return counts


def synthAcquisition(FNAME, normalize=True):
    # Get raw counts in DataFrame format
    counts = dataAcquisition(FNAME, normalize=False)

    # Separate labels
    labels = counts[:, -1]
    counts = counts[:, :-1]

    # Normalize counts
    if normalize:
        counts = normalize_counts_10x(counts)

    return counts, labels


# Standardize columns of matrix X: (X - X.mean) / X.std
# Also returns StandardScaler object for consistent further (inverse) standardization
# from sklearn.preprocessing import StandardScaler
def standardize(X):
    scaleX = StandardScaler().fit(X)
    return scaleX.transform(X), scaleX


# tf-idf normalizing: cells as documents, genes as words
# from sklearn.feature_extraction.text import TfidfTransformer
def normalize_tf_idf(X):
    tfidf = TfidfTransformer(norm=None, smooth_idf=True, sublinear_tf=False)
    tfidf.fit(X)
    return tfidf.transform(X)


def normalize_counts(raw_counts, doStandardize=False):
    """
    Normalizes count array using method in 10x pipeline
    :param raw_counts: numpy array of count data
    :return normed: normalized data
    """

    # Sum across cells and divide each cell by sum
    cell_sums = np.sum(raw_counts, axis=1)

    # Mutiply by median and divide by cell sum
    median = np.median(cell_sums)
    raw_counts = raw_counts * median / cell_sums[:, np.newaxis]

    raw_counts = np.log(raw_counts + 0.1)

    if doStandardize:
        # Normalize to have genes with mean 0 and std 1
        std = np.std(raw_counts, axis=0)[np.newaxis, :]

        # Fix potential divide by zero
        std[np.where(std == 0)[0]] = 1

        normed = (raw_counts - np.mean(raw_counts, axis=0)) / std
    else:
        normed = raw_counts

    return normed
