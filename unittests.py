#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 27, 2017

@author: JonathanShor
"""

import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import phenograph
import doubletdetection

TEST_CLASSIFIERINIT = False
TEST_FIT = False
TEST_UNIQUE = False
TEST_TOPVARGENES = True


# import numpy as np
# from sklearn.decomposition import PCA
# import phenograph
# CELLTYPESAMPLEMEAN = 0.05


# FNAME = "~/Google Drive/Computational Genomics/pbmc8k_dense.csv"
# FNAME = "~/Google Drive/Computational Genomics/clean_5050.csv"
# FNAME = "~/Google Drive/Computational Genomics/pbmc_4k_dense.csv"
FPATH = "/Users/jonathanshor/Google Drive/dataset"
DATASETNUM = 5050


def classifierInitTest(raw_counts):
    pass


def fitTest(raw_counts):
    pass


def uniqueTest(raw_counts):
    (n, p) = raw_counts
    print("Start PCA")
    pca = PCA(n_components=30)
    reduced_counts = pca.fit_transform(raw_counts)
    print("Start phenograph")
    communities, graph, Q = phenograph.cluster(reduced_counts[:raw_counts.shape[0], :])
    print("Found communities [{0}, ... {2}], with sizes: {1}".format(min(communities),
          [np.count_nonzero(communities == i) for i in np.unique(communities)], max(communities)))

    uniqueG = doubletdetection.getUniqueGenes(raw_counts, communities)
    assert np.all(np.unique(uniqueG) == [0, 1])

    geneNames = pd.read_csv(FNAME, index_col=0, nrows=1).columns
    print("Unique genes: {}".format(geneNames[np.sum(uniqueG, axis=0) == 1]))
    print("Unique per cluster: {}".format(list(zip(range(uniqueG.shape[0]), np.sum(uniqueG, axis=1)))))
    print("XIST is in cluster {}".format(np.where(uniqueG[:, 13750])))
    print("CD3D is in cluster {}".format(np.where(uniqueG[:, 19801])))


def top_var_genes_test(raw_counts, n_top_var_genes=50):
    clf = doubletdetection.BoostClassifier(n_top_var_genes=n_top_var_genes)
    labels = clf.fit(raw_counts)
    num_labels = np.sum(labels == 1)

    print("Number of doublets =", num_labels)

    assert n_top_var_genes <= 0 or np.shape(clf._raw_counts)[1] == n_top_var_genes, (
        "Expected {0}, got {1} genes.".format(n_top_var_genes, np.shape(clf._raw_counts)[1]))

    return num_labels


if __name__ == '__main__':
    start_time = time.time()

    # raw_counts = doubletdetection.load_csv(FNAME)
    COUNTFNAME = FPATH + str(DATASETNUM) + "/d" + str(DATASETNUM) + "_counts_blosc.h5"
    raw_counts = pd.read_hdf(COUNTFNAME, 'table').as_matrix()

    if TEST_CLASSIFIERINIT:
        classifierInitTest(raw_counts)
    if TEST_FIT:
        fitTest(raw_counts)
    if TEST_UNIQUE:
        uniqueTest(raw_counts)
    if TEST_TOPVARGENES:
        top_var_genes_test(raw_counts, n_top_var_genes=200)

    # clf = doubletdetection.BoostClassifier()
    # labels = clf.fit(raw_counts)

    # print("Number of doublets =", np.sum(labels == 1))

    print("Total run time: {0:.2f} seconds".format(time.time() - start_time))
