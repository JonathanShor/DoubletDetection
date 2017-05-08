#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
	Sample usage on our validation set.

    To run from within DoubletDetection directory:
    	python3 ./validation_script.py -f [file_name]

    Note: all command line flags optional other than file name
"""

import doubletdetection
import numpy as np
import sys
from optparse import OptionParser
import time


FNAME = "~/Google Drive/Computational Genomics/5050.csv"

def main(trials):
        
    # Load data
    raw_counts = doubletdetection.load_data(FNAME)
    
    idxs = {}
    for i in range(trials):
        # Classify
        counts, scores, communities, doublet_labels, cutoff = doubletdetection.classify(raw_counts, boost_rate=0.25) 
        
        doublets = np.where(scores[:raw_counts.shape[0]]>=cutoff)[0]
        
        for d in doublets:
            if d not in idxs:
                idxs[d] = 1
            else:
                idxs[d] += 1
                
            
    # Analyze doublets
    print("On trial number", trials, "...")
    print('Number of doublets =', len(doublets))
    identified_doublets = raw_counts[doublets]
    XIST = np.where(identified_doublets[:,13750]>0)
    CD3D = np.where(identified_doublets[:,19801]>0)
    np.intersect1d(XIST,CD3D)
    XandC = np.intersect1d(XIST,CD3D)
    print("Number of doublets which express both XIST and CD3D =", len(XandC))
    
    print("\nOut of", trials, "trials there were", len(idxs.keys()), "unique doublets.")
    print("The mean occurence of doublets is", np.mean(list(idxs.values())))
    print("The standard deviation of occurence is", np.std(list(idxs.values())))
    
    
if __name__ == '__main__':
    argv = sys.argv[1:]
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file", help='read csv data from FILE', metavar='FILE')
    parser.add_option("-n", type='int', dest="trials", default=10, help='number of simulation trials')
    (options, _args) = parser.parse_args()
    if options.file:
        FNAME = options.file
    
    start = time.time()
    main(options.trials)    
    end = time.time()
    print(end - start, "seconds")

