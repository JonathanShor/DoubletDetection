#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 3, 2017

@author: JonathanShor
"""

import pandas as pd
import numpy as np
import phenograph
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import BernoulliNB
from utils import testModel
from utils import normalize_counts_10x

CELLTYPESAMPLEMEAN = 0.05   # Mean percent of cell gene expression captured per cell read
DOUBLETRATE = 0.07


# TODO: Remove these print blocking funcs
import sys
import os


# Disable
def blockPrint():
    if not any('SPYDER' in name for name in os.environ):
        sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    if not any('SPYDER' in name for name in os.environ):
        sys.stdout = sys.__stdout__
# TODO: Remove these print blocking funcs


def sampleCellRead(mean_reads, gene_probs, num_cells=1):
    num_genes = len(gene_probs)
    # draws = np.random.multinomial([mean_reads] * num_genes,
    #                               gene_probs,
    #                               size=num_cells)
    draws = np.random.binomial([mean_reads] * num_genes,
                               gene_probs,
                               size=(num_cells, num_genes))
    return draws


# Given n celltypes, and type_freqs indicating relative frequency of each celltype,
# Return one doublet: two celltypes (selected prop. to type_freqs) combined with
def doubletFromCelltype(celltypes, doublet_weight=1):
    genecounts = np.array(celltypes['genecounts'])
    cellcounts = np.array(celltypes['cellcounts'])
    mean_reads = np.array(np.sum(genecounts, axis=1) * CELLTYPESAMPLEMEAN, dtype=int)
    celltypesProb = genecounts / np.sum(genecounts, axis=1).reshape(-1, 1)
    [type1, type2] = np.random.choice(range(len(cellcounts)), p=cellcounts / sum(cellcounts),
                                      size=2)
    cell1 = sampleCellRead(mean_reads[type1], celltypesProb[type1]) * doublet_weight
    cell2 = sampleCellRead(mean_reads[type2], celltypesProb[type2])
    return cell1 + cell2


# Generate 2D synthetic data from celltypes
# Celltypes expected to be a dict with members 'genecounts':2d(celltypes x genes)
#  and 'frequences': 1d(number of cells to generate for each type)
def create_synthetic_data(celltypes=None, doublet_weight=1):
    if celltypes is None:
        celltypes = getCellTypes()

    genecounts = np.array(celltypes['genecounts'])
    cellcounts = np.array(celltypes['cellcounts'])

    num_genes = genecounts.shape[1]
    synthetic = np.empty((0, num_genes))

    # TODO: make mean_reads, celltypesProb, etc. members of celltypes, not stand-alone vars
    # TODO: likely want a type struct/class to gather all this
    # Mean number of transcript reads for one cell
    mean_reads = np.array(np.sum(genecounts, axis=1) * CELLTYPESAMPLEMEAN, dtype=int)

    celltypesProb = genecounts / np.sum(genecounts, axis=1).reshape(-1, 1)
    assert all(abs(np.sum(celltypesProb, axis=1) - 1) < 1e-8), (
        [[i, x] for i, x in enumerate(np.sum(celltypesProb, axis=1)) if abs() >= 1e-8])

    # Create non-doublet base data
    for i, cellcount in enumerate(cellcounts):
        synthetic = np.concatenate((synthetic,
                                    sampleCellRead(mean_reads[i], celltypesProb[i], cellcount)),
                                   axis=0)
    num_cells = synthetic.shape[0]
    assert num_cells == np.sum(cellcounts), (
        "made num_cells:{0}, expect np.sum(cellcounts): {1}".format(num_cells, np.sum(cellcounts)))
    assert np.count_nonzero(np.sum(synthetic, axis=1) == 0) == 0, (
        "{} cells with zero reads... ".format(np.count_nonzero(np.sum(synthetic, axis=1) == 0)))

    # Replace DOUBLETRATE * num_cells with doublets
    num_doublets = int(num_cells * DOUBLETRATE)
    doublets = np.random.permutation(num_cells)[:num_doublets]
    for doublet in doublets:
        type_i = np.random.choice(range(len(cellcounts)), p=cellcounts / sum(cellcounts))
        synthetic[doublet] = (synthetic[doublet] * doublet_weight +
                              sampleCellRead(mean_reads[type_i], celltypesProb[type_i]))

    # Set labels[i] == 1 where synthetic[i,:] is a doublet
    labels = np.zeros(num_cells)
    labels[doublets] = 1

    try:    # return pandas if we got pandas
        synthetic = pd.DataFrame(synthetic, columns=celltypes['genecounts'].columns)
        labels = pd.Series(labels)
    except AttributeError:
        pass

    return synthetic, labels


# Return mean of cluster members for each cluster
def getCentroids(data, clusters):
    assert all(np.arange(max(clusters) + 1) == np.unique(clusters)), "gap in cluster IDs"
    centroids = np.empty_like(data[:max(clusters)])
    for i in range(len(centroids)):
        centroids[i] = np.mean(data[clusters == i], axis=0)
    return centroids


# Cell type generation for synthetic data generation
# Return dictionary of genecounts for each type (2d), and relative cellcounts (1d)
def getCellTypes(counts=None, PCA_components=30, shrink=0.01):
    if counts is None:
        # TODO: Implement randomly generated celltypes?
        raise Exception("Random celltype generation not implemented.")
    else:
        try:
            npcounts = counts.as_matrix()
        except AttributeError:
            npcounts = counts

        # Basic doublet removal: each cluster pruned by shrink% most-distant-from-centroid cells
        # TODO: Better doublet removal techniques?
        reduced_counts = PCA(n_components=PCA_components).fit_transform(npcounts)
        blockPrint()
        communities, graph, Q = phenograph.cluster(reduced_counts)
        enablePrint()
        print("Found these communities: {0}, with sizes: {1}".format(np.unique(communities),
              [np.count_nonzero(communities == i) for i in np.unique(communities)]))

        # Throw out outlier cluster (ID = -1)
        npcounts = npcounts[communities >= 0]
        communities = communities[communities >= 0]

        preshrink_cellcounts = np.array([np.count_nonzero(communities == i) for i in
                                        np.unique(communities)])

        # assert all(np.arange(max(communities) + 1) == np.unique(communities)), (
        #     "gap in communities IDs")
        # Shrink each community by shrink, ranked by l_2 distance from cluster centroid
        centroids = getCentroids(npcounts, communities)
        distances = np.zeros(npcounts.shape[0])
        to_shrink = []
        for i, centroid in enumerate(centroids):
            members = np.nonzero(communities == i)[0]   # Current cluster's indexes in npcount

            # set each cell's distance from its centroid
            for member in members:
                distances[member] = np.linalg.norm(npcounts[member] - centroid, ord=2, axis=0)

            # Delete the smallest shrink%
            for _ in range(int(preshrink_cellcounts[i] * shrink)):
                smallest_member = np.argsort(distances[members])[0]
                to_shrink.append(members[smallest_member])  # Translate to index in full npcounts
                distances[to_shrink[-1]] = np.inf

        assert len(to_shrink) == len(np.unique(to_shrink)), (
            "repeats in to_shrink: {}".format(to_shrink))
        shrunk = np.setdiff1d(np.arange(len(npcounts)), to_shrink)
        npcounts = npcounts[shrunk]
        communities = communities[shrunk]

        cellcounts = np.array([np.count_nonzero(communities == i) for i in np.unique(communities)])
        assert sum(preshrink_cellcounts) * (1 - shrink) <= sum(cellcounts), "bad shrink"
        assert sum(cellcounts) == npcounts.shape[0], "cellcounts does not match npcounts.shape[0]"
        assert sum(cellcounts) == len(communities), "cellcounts does not match len(communities)"

        genecounts = np.array([np.sum(npcounts[communities == i], axis=0)
                               for i in np.unique(communities)])
        assert ~(any(np.max(genecounts, axis=1) == 0)), "zero vector celltype"
        genecounts = genecounts / (CELLTYPESAMPLEMEAN * cellcounts.reshape(-1, 1))

    try:    # return pandas if we got pandas
        genecounts = pd.DataFrame(genecounts, columns=counts.columns)
    except AttributeError:
        pass

    return {'genecounts': genecounts, 'cellcounts': cellcounts}


# Print l_2 distance from each synthetic cell to closest non-synth.
# Average distance from non-synth to closest other non-synth printed for comparison
def checkSyntheticDistance(synth, labels):
    try:
        synthetic = synth.as_matrix()
    except AttributeError:
        synthetic = synth
    raw_counts = synthetic[labels == 0]
    print("Mean minimum l_2 distance between cells: {0:.4f}".format(
          np.array([np.min(np.linalg.norm(raw_counts[np.arange(len(raw_counts)) != i] - x, ord=2,
                                          axis=1)) for i, x in enumerate(raw_counts)]).mean()))
    min_synth_sim = np.array([np.min(np.linalg.norm(synthetic[labels == 0] - i, ord=2, axis=1))
                              for i in synthetic[labels == 1]])
    print("Min distance from each doublet to non-doublet:\n")
    print(np.round(min_synth_sim, 4).reshape(-1, 1))


# Slow but works
# Takes a pd DataFrame
# Returns numpy matrices
def create_simple_synthetic_data(raw_counts, alpha1, alpha2, write=False, normalize=False, doublet_rate=DOUBLETRATE):

    synthetic = pd.DataFrame()

    cell_count = raw_counts.shape[0]
    #doublet_rate = DOUBLETRATE
    doublets = int(doublet_rate * cell_count / (1 - doublet_rate))

    # Add labels column to know which ones are doublets
    labels = np.zeros(cell_count + doublets)
    labels[cell_count:] = 1

    for i in range(doublets):
        row1 = int(np.random.rand() * cell_count)
        row2 = int(np.random.rand() * cell_count)

        new_row = alpha1 * raw_counts.iloc[row1] + alpha2 * raw_counts.iloc[row2]

        synthetic = synthetic.append(new_row, ignore_index=True)

    synthetic = raw_counts.append(synthetic)

    if write:
        synthetic['labels'] = labels
        synthetic.to_csv("~/Google Drive/Computational Genomics/synthetic.csv")

    if normalize:
        synthetic = normalize_counts_10x(synthetic)
        return synthetic, labels

    return synthetic.as_matrix(), labels


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
