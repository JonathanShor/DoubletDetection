#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 27, 2017

@author: JonathanShor
"""

import time
import numpy as np
import json
import doubletdetection
import unittests

FNAME = "~/Google Drive/Computational Genomics/pbmc_4k_dense.csv"
NUM_RUNS = 100
# GENE_CAPS = [0, 50, 100, 200]
GENE_CAPS = [x * 25 for x in range(13)]


def aggregate(data):
    return np.vstack((GENE_CAPS,
                      np.mean(data, axis=1),
                      np.var(data, axis=1),
                      np.min(data, axis=1),
                      np.max(data, axis=1))).T


if __name__ == '__main__':
    start_time = time.time()

    results = np.zeros((len(GENE_CAPS), NUM_RUNS))
    times = np.zeros((len(GENE_CAPS), NUM_RUNS))

    raw_counts = doubletdetection.load_csv(FNAME)

    # Quick hack to write variances out to a file, uncomment if so inclined
    # gene_variances = np.var(raw_counts, axis=0)
    # with open('vars.txt', 'w') as f:
    #     f.write(str(np.sort(gene_variances).tolist()))
    # f.closed

    for i in range(NUM_RUNS):
        for j, gene_count in enumerate(GENE_CAPS):
            test_start = time.time()
            results[j, i] = unittests.top_var_genes_test(raw_counts, n_top_var_genes=gene_count)
            times[j, i] = time.time() - test_start

    with open('results.txt', 'w') as f:
        for i, result in enumerate(results):
            gene_count = GENE_CAPS[i]
            json.dump([gene_count, result.tolist()], f)
            f.write("\n")
            json.dump((gene_count, times[i].tolist()), f)
            f.write("\n")

        aggregate_results = aggregate(results)
        aggregate_times = aggregate(times)
        msg = "Num Genes |  Mean  | Variance | [ Min, Max ]"
        print(msg)
        f.write(msg + "\n")
        for i, result in enumerate(aggregate_results):
            msg = "{0:9d} | {1:6.1f} | {2:8.1f} | [{3:4.0f},{4:4.0f}]".format(
                GENE_CAPS[i], result[1], result[2], result[3], result[4])
            print(msg)
            f.write(msg + "\n")
            this_time = aggregate_times[i]
            msg = " Run time | {0:6.1f} | {1:8.1f} | [{3:4.1f},{3:4.1f}]\n".format(
                this_time[1], this_time[2], this_time[3], this_time[4])
            print(msg)
            f.write(msg + "\n")

    f.closed

    print("Total run time: {0:.2f} seconds".format(time.time() - start_time))
