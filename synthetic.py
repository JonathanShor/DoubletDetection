#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 3, 2017

@author: adamgayoso, JonathanShor, ryanbrand
"""
import numpy as np
from utils import normalize_counts

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
def doubletFromCelltype(celltypes, doublet_weight=0.5, allow_same_parent=True):
    try:
        w1 = doublet_weight[0]
        w2 = doublet_weight[1]
    except TypeError:   # assume scalar, use for both weights
        w1 = doublet_weight
        w2 = doublet_weight
    genecounts = np.array(celltypes['genecounts'])
    cellcounts = np.array(celltypes['cellcounts'])
    mean_reads = np.array(np.sum(genecounts, axis=1) * CELLTYPESAMPLEMEAN, dtype=int)
    celltypesProb = genecounts / np.sum(genecounts, axis=1).reshape(-1, 1)
    parents = np.random.choice(range(len(cellcounts)), p=cellcounts / sum(cellcounts), size=2,
                               replace=allow_same_parent)
    cell1 = sampleCellRead(mean_reads[parents[0]], celltypesProb[parents[0]])
    cell2 = sampleCellRead(mean_reads[parents[1]], celltypesProb[parents[1]])
    doublet = np.array(np.around(cell1 * w1 + cell2 * w2), dtype=cell1.dtype)
    return doublet, parents


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
def getCellTypes(counts=None, PCA_components=30, shrink=0.01, knn=30):
    if counts is None:
        # TODO: Implement randomly generated celltypes?
        raise Exception("Random celltype generation not implemented.")
    else:
        # Basic doublet removal: each cluster pruned by shrink% most-distant-from-centroid cells
        # TODO: Better doublet removal techniques?
        reduced_counts = PCA(n_components=PCA_components).fit_transform(counts)
        blockPrint()
        communities, graph, Q = phenograph.cluster(reduced_counts, k=knn)
        enablePrint()
        print("Found these communities: {0}, with sizes: {1}".format(np.unique(communities),
              [np.count_nonzero(communities == i) for i in np.unique(communities)]))

        # Throw out outlier cluster (ID = -1)
        counts = counts[communities >= 0]
        communities = communities[communities >= 0]

        preshrink_cellcounts = np.array([np.count_nonzero(communities == i) for i in
                                        np.unique(communities)])

        # assert all(np.arange(max(communities) + 1) == np.unique(communities)), (
        #     "gap in communities IDs")
        # Shrink each community by shrink, ranked by l_2 distance from cluster centroid
        centroids = getCentroids(reduced_counts, communities)
        distances = np.zeros(counts.shape[0])
        to_shrink = []
        for i, centroid in enumerate(centroids):
            members = np.nonzero(communities == i)[0]   # Current cluster's indexes in npcount

            # set each cell's distance from its centroid
            for member in members:
                distances[member] = np.linalg.norm(reduced_counts[member] - centroid, ord=2, axis=0)

            # Delete the smallest shrink%
            for _ in range(int(preshrink_cellcounts[i] * shrink)):
                smallest_member = np.argsort(distances[members])[0]
                to_shrink.append(members[smallest_member])  # Translate to index in full counts
                distances[to_shrink[-1]] = np.inf

        assert len(to_shrink) == len(np.unique(to_shrink)), (
            "repeats in to_shrink: {}".format(to_shrink))
        shrunk = np.setdiff1d(np.arange(len(counts)), to_shrink)
        counts = counts[shrunk]
        communities = communities[shrunk]

        cellcounts = np.array([np.count_nonzero(communities == i) for i in np.unique(communities)])
        assert sum(preshrink_cellcounts) * (1 - shrink) <= sum(cellcounts), "bad shrink"
        assert sum(cellcounts) == counts.shape[0], "cellcounts does not match counts.shape[0]"
        assert sum(cellcounts) == len(communities), "cellcounts does not match len(communities)"

        genecounts = np.array([np.sum(counts[communities == i], axis=0)
                               for i in np.unique(communities)])
        assert ~(any(np.max(genecounts, axis=1) == 0)), "zero vector celltype"
        genecounts = genecounts / (CELLTYPESAMPLEMEAN * cellcounts.reshape(-1, 1))

    return {'genecounts': genecounts, 'cellcounts': cellcounts}


# Print l_2 distance from each synthetic cell to closest non-synth.
# Average distance from non-synth to closest other non-synth printed for comparison
def checkSyntheticDistance(synthetic, labels):
    raw_counts = synthetic[labels == 0]
    print("Mean minimum l_2 distance between cells: {0:.4f}".format(
          np.array([np.min(np.linalg.norm(raw_counts[np.arange(len(raw_counts)) != i] - x, ord=2,
                                          axis=1)) for i, x in enumerate(raw_counts)]).mean()))
    min_synth_sim = np.array([np.min(np.linalg.norm(synthetic[labels == 0] - i, ord=2, axis=1))
                              for i in synthetic[labels == 1]])
    print("Min distance from each doublet to non-doublet:\n")
    print(np.round(min_synth_sim, 4).reshape(-1, 1))


def create_simple_synthetic_data(raw_counts, alpha1, alpha2, normalize=True, doublet_rate=DOUBLETRATE):
    """
    Appends doublets to end of data
    :param raw_counts: numpy of count data
    :param alpha1: weighting of row1 in sum
    :param alpha2: weighting of row2 in sum
    :param normalize: normalize data before returning
    :return synthetic: synthetic data in numpy array
    :return labels: 0 for original data, 1 for fake doublet as np array - 1d arrray
    """

    # Get shape
    cell_count = raw_counts.shape[0]
    gene_count = raw_counts.shape[1]

    # Number of doublets to add
    doublets = int(doublet_rate * cell_count)

    synthetic = np.zeros((doublets, gene_count))

    # Add labels column to know which ones are doublets
    labels = np.zeros(cell_count + doublets)
    labels[cell_count:] = 1

    for i in range(doublets):
        row1 = int(np.random.rand() * cell_count)
        row2 = int(np.random.rand() * cell_count)

        new_row = np.array(np.around(alpha1 * raw_counts[row1] + alpha2 * raw_counts[row2]),
                           dtype=raw_counts.dtype)

        synthetic[i] = new_row

    # Shouldn't change original raw_counts
    synthetic = np.append(raw_counts, synthetic, axis=0)

    if normalize:
        synthetic = normalize_counts(synthetic)

    return synthetic, labels

def downsampledDoublets(raw_counts, normalize=True, doublet_rate=DOUBLETRATE):  
    """
    Appends downsampled doublets to end of data
    :param raw_counts: numpy of count data
    :param alpha1: weighting of row1 in sum
    :param alpha2: weighting of row2 in sum
    :param normalize: normalize data before returning
    :return synthetic: synthetic data in numpy array
    :return labels: 0 for original data, 1 for fake doublet as np array - 1d arrray
    """

    # Get shape
    cell_count = raw_counts.shape[0]
    gene_count = raw_counts.shape[1]

    # Number of doublets to add
    doublets = int(doublet_rate * cell_count)

    synthetic = np.zeros((doublets, gene_count))

    # Add labels column to know which ones are doublets
    labels = np.zeros(cell_count + doublets)
    labels[cell_count:] = 1
    
    lib_size = np.mean(np.sum(raw_counts, axis=1))
    std = np.std(np.sum(raw_counts, axis=1))

    for i in range(doublets):
        row1 = int(np.random.rand() * cell_count)
        row2 = int(np.random.rand() * cell_count)

        new_cell = raw_counts[row1] + raw_counts[row2]
        
        lib1 = np.sum(raw_counts[row1])
        lib2 = np.sum(raw_counts[row2])
        #new_lib_size = int(np.random.normal(loc=lib_size, scale = std))
        new_lib_size = int(max(lib1, lib2))
        mol_ind = np.random.permutation(int(lib1+lib2))[:new_lib_size]
        bins = np.append(np.zeros((1)),np.cumsum(new_cell))
        new_cell = np.histogram(mol_ind, bins)[0]
        
        synthetic[i] = new_cell

    # Shouldn't change original raw_counts
    synthetic = np.append(raw_counts, synthetic, axis=0)

    if normalize:
        synthetic = normalize_counts(synthetic)

    return synthetic, labels

def sameDownsampledDoublets(raw_counts, normalize=True, doublet_rate=DOUBLETRATE):  
    """
    Appends downsampled doublets to end of data
    :param raw_counts: numpy of count data
    :param alpha1: weighting of row1 in sum
    :param alpha2: weighting of row2 in sum
    :param normalize: normalize data before returning
    :return synthetic: synthetic data in numpy array
    :return labels: 0 for original data, 1 for fake doublet as np array - 1d arrray
    """

    # Get shape
    cell_count = raw_counts.shape[0]
    gene_count = raw_counts.shape[1]

    # Number of doublets to add
    doublets = int(doublet_rate * cell_count)

    synthetic = np.zeros((doublets, gene_count))

    # Add labels column to know which ones are doublets
    labels = np.zeros(cell_count + doublets)
    labels[cell_count:] = 1
    
    parents = np.zeros(cell_count + doublets)
    
    lib_size = np.mean(np.sum(raw_counts, axis=1))
    std = np.std(np.sum(raw_counts, axis=1))

    for i in range(doublets):
        row1 = int(np.random.rand() * cell_count)


        new_cell = 2*raw_counts[row1]
        
        lib1 = np.sum(raw_counts[row1])
        #new_lib_size = int(np.random.normal(loc=lib_size, scale = std))
        new_lib_size = lib1
        mol_ind = np.random.permutation(2*lib1)[:new_lib_size]
        bins = np.append(np.zeros((1)),np.cumsum(new_cell))
        new_cell = np.histogram(mol_ind, bins)[0]
        
        synthetic[i] = new_cell
        parents[i] = row1

    # Shouldn't change original raw_counts
    synthetic = np.append(raw_counts, synthetic, axis=0)

    if normalize:
        synthetic = normalize_counts(synthetic)

    return synthetic, labels, parents
    
    