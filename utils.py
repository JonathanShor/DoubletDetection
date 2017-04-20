#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 3, 2017

@author: adamgayoso, JonathanShor, ryanbrand
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler


# To read only the first X rows, set read_rows=X
def dataAcquisition(FNAME, normalize=False, read_rows=None):
    # Import counts
    counts = pd.read_csv(FNAME, index_col=0, nrows=read_rows)

    # Normalize
    if normalize:
        # Replacing with NaN makes it easier to ignore these values
        counts[counts == 0] = np.nan

        if normalize == "TFIDF":
            counts = normalize_tf_idf(counts)
        else:   # 10x paper normalization
            counts = normalize_counts_10x(counts)

    return counts


def synthAcquisition(FNAME, normalize=True):
    #Get raw counts in DataFrame format
    counts = dataAcquisition(FNAME, normalize=False)
    
    #Separate labels
    labels = counts['labels']
    del counts['labels']
    doublet_labels = labels.as_matrix()
    
    #Normalize counts
    if normalize:
        counts = normalize_counts_10x(counts)
        
    return counts, doublet_labels


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


# Takes np array
# Following method in 10x paper
def normalize_counts_10x(raw_counts):
    """
    Normalizes count array using method in 10x pipeline  
    :param raw_counts: numpy array of count data
    :return normed: normalized data
    """
    
    # Sum across cells and divide each cell by sum
    cell_sums = np.sum(raw_counts, axis=1)
    
    # Set 0s to NaN to make calculations work more smoothly
    raw_counts[raw_counts == 0] = np.nan


    # Mutiply by median and divide by cell sum
    median = np.median(cell_sums)
    raw_counts = raw_counts * median / cell_sums[:, np.newaxis]

    # Take log and normalize to have mean 0 and std 1 in each gene across all cells
    raw_counts = np.log(raw_counts)
        
    # Replace NaN with 0
    raw_counts = np.nan_to_num(raw_counts)

    # Normalize to have genes with mean 0 and std 1
    std = np.std(raw_counts, axis=0)[np.newaxis, :]
    
    #Fix potential divide by zero
    std[np.where(std == 0)[0]] = 1
    
    normed = (raw_counts - np.mean(raw_counts, axis=0)) / std
    # TODO: Use standardize() if we need to inverse or repeat stardardization

    return normed


