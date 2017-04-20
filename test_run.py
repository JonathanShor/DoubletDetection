#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:29:52 2017

@author: adamgayoso
"""

import doubletdetection
import numpy as np
#import visualize

FNAME = "~/Google Drive/Computational Genomics/pbmc8k_dense.csv"
SYN_FNAME = "~/Google Drive/Computational Genomics/synthetic.csv"

# Get scores
raw_counts, scores = doubletdetection.classify(FNAME)

for s in range(20, 80, 5):
    cutoff = s/float(100)
    test = scores[np.where(scores>cutoff)[0]]
    print(cutoff, len(test))
    
# Visualize tSNE clustering
# Different color for each cluster and a different mark for identified doublet 
# Only visualize raw counts
counts = doubletdetection.utils.normalize_counts_10x(raw_counts)

