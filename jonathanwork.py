#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 3, 2017

@author: JonathanShor
"""

import time
from synthetic import getCellTypes
from synthetic import create_synthetic_data
from utils import dataAcquisition

FNAME = "~/Google Drive/Computational Genomics/pbmc8k_dense_100lines.csv"


if __name__ == '__main__':
    start_time = time.time()

    # Import counts
    raw_counts = dataAcquisition(FNAME)
    # raw_counts = dataAcquisition(FNAME, normalize=True, useTFIDF=True)

    syntheticData, labels = create_synthetic_data(getCellTypes(raw_counts))

    # synthetic['labels'] = labels

    # pca = PCA(n_components=30)
    # synthetic.to_csv("/Users/adamgayoso/Google Drive/Computational Genomics/synthetic.csv")

    # syntheticTesting(synthetic.as_matrix(), labels)
    # analysisSuite(synthetic)

    print("Total run time: {0:.2f} seconds".format(time.time() - start_time))
