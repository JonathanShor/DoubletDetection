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

        new_row = alpha1 * raw_counts[row1] + alpha2 * raw_counts[row2]

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
    
    