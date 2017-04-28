#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 3, 2017

@author: adamgayoso, JonathanShor, ryanbrand
"""
import numpy as np
import pandas as pd



# To read only the first X rows, set read_rows=X
def load_data(FNAME, normalize=False, read_rows=None):
    # Import counts
    counts = pd.read_csv(FNAME, index_col=0, nrows=read_rows).as_matrix()

    # Normalize
    if normalize:

        if normalize == "TFIDF":
            counts = normalize_tf_idf(counts)
        else:   # 10x paper normalization
            counts = normalize_counts(counts)

    return counts


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