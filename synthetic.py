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

CELLTYPESAMPLEMEAN = 0.05   # Mean percent of cell gene expression captured per cell read
DOUBLETRATE = 0.07


# TODO: Remove these print blocking funcs
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
# TODO: Remove these print blocking funcs


# Generate 2D synthetic data from celltypes
# Celltypes expected to be a dict with members 'genecounts':2d(celltypes x genes)
#  and 'frequences': 1d(number of cells to generate for each type)
def create_synthetic_data(celltypes=None):
    if celltypes is None:
        celltypes = getCellTypes()

    def sampleCellRead(mean_reads, gene_probs, num_cells=1):
        num_genes = len(gene_probs)
        # draws = np.random.multinomial([mean_reads] * num_genes,
        #                               gene_probs,
        #                               size=num_cells)
        draws = np.random.binomial([mean_reads] * num_genes,
                                   gene_probs,
                                   size=(num_cells, num_genes))
        return draws

    genecounts = np.array(celltypes['genecounts'])
    cellcounts = np.array(celltypes['cellcounts'])

    num_genes = genecounts.shape[1]
    synthetic = np.empty((0, num_genes))

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

    # print ("pd.Series(np.sum(synthetic, axis=1)).describe()",
    #        pd.Series(np.sum(synthetic, axis=1)).describe())
    # celltypesProbnan = celltypesProb.copy()
    # celltypesProbnan[celltypesProbnan == 0] = np.nan
    # print ("(Mean, count, mean_reads), non-zero entries of celltypesProb: ",
    #        [(np.nanmean(x), np.count_nonzero(~np.isnan(x)), mean_reads[i])
    #         for i, x in enumerate([celltypesProbnan])])
    assert np.count_nonzero(np.sum(synthetic, axis=1) == 0) == 0, (
        "{} cells with zero reads... ".format(np.count_nonzero(np.sum(synthetic, axis=1) == 0)))

    # Replace DOUBLETRATE * num_cells with doublets
    num_doublets = int(num_cells * DOUBLETRATE)
    doublets = np.random.permutation(num_cells)[:num_doublets]
    for doublet in doublets:
        # TODO: pick celltype to mix with chance proportional to cellcounts
        type_i = np.random.randint(genecounts.shape[0])
        synthetic[doublet] = synthetic[doublet] + sampleCellRead(mean_reads[type_i],
                                                                 celltypesProb[type_i])

    # Set labels[i] == 1 where synthetic[i,:] is a doublet
    labels = np.zeros(num_cells)
    labels[doublets] = 1

    try:    # return pandas if we got pandas
        synthetic = pd.DataFrame(synthetic, columns=celltypes['genecounts'].columns)
        labels = pd.Series(labels)
    except AttributeError:
        pass

    return synthetic, labels


# Cell type generation for synthetic data generation
# Return dictionary of genecounts for each type (2d), and relative cellcounts (1d)
def getCellTypes(counts=None, PCA_components=30):
    if counts is None:
        raise Exception("Random celltype generation not implemented.")
    else:
        try:
            npcounts = counts.as_matrix()
        except AttributeError:
            npcounts = counts

        # Naive initial attempt: does not remove dublets in the orig data in any way
        # TODO: Revise to remove doublet noise (to at least some degree)
        reduced_counts = PCA(n_components=PCA_components).fit_transform(npcounts)
        blockPrint()
        communities, graph, Q = phenograph.cluster(reduced_counts)
        enablePrint()
        print("Found these communities: {0}, with sizes: {1}".format(np.unique(communities),
              [np.count_nonzero(communities == i) for i in np.unique(communities)]))

        # Throw out outliers in cluster -1
        npcounts = npcounts[communities >= 0]
        communities = communities[communities >= 0]

        cellcounts = np.array([np.count_nonzero(communities == i) for i in np.unique(communities)])

        # genecounts = np.zeros((max(communities), npcounts.shape[1]))
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
def checkSyntheticDistance(synthetic, labels):
    # synthetic = synthetic.as_matrix()
    raw_counts = synthetic[labels == 0]
    print("Mean minimum l_2 distance between cells: {0:.4f}".format(
          np.array([np.min(np.linalg.norm(raw_counts[np.arange(len(raw_counts)) != i] - x, ord=2,
                                          axis=1)) for i, x in enumerate(raw_counts)]).mean()))
    min_synth_sim = np.array([np.min(np.linalg.norm(synthetic[labels == 0] - i, ord=2, axis=1))
                              for i in synthetic[labels == 1]])
    print(np.round(min_synth_sim, 4).reshape(-1, 1))


# Slow but works
# Takes a pd DataFrame
# Returns numpy matrices 
def create_simple_synthetic_data(raw_counts, write=False, alpha1=1, alpha2=1):

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

        new_row = alpha1*raw_counts.iloc[row1] + alpha2*raw_counts.iloc[row2]

        synthetic = synthetic.append(new_row, ignore_index=True)

    synthetic = raw_counts.append(synthetic)
       
    synthetic['labels'] = labels
    if write:
        synthetic.to_csv("~/Google Drive/Computational Genomics/synthetic.csv")

    return synthetic.as_matrix(), labels.as_matrix()

# Supervised classification using sythetic data
def syntheticTesting(X_geneCounts, y_doubletLabels, useTruncSVD=False):
    # X_standardized = normalize_counts_10x(X_geneCounts)

    # Naive classifier test: library size
    librarySize = X_geneCounts.sum(axis=1).reshape(-1, 1)
    # librarySizeSt = X_standardized.sum(axis=1).reshape(-1, 1)
    print("librarySize.shape: ", (librarySize.shape))
    # print("X_stardardized.shape: ", (X_stardardized.shape))
    print("y_doubletLabels.shape: ", (y_doubletLabels.shape))
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
    communities, graph, Q = phenograph.cluster(X_reduced_counts)
    print("Num communities found: {}".format(communities.max_()))

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
