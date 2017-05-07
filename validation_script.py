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


FNAME = "~/Google Drive/Computational Genomics/5050.csv"

def main():
        
    # Load data
    raw_counts = doubletdetection.load_data(FNAME)

    # Classify
    counts, scores, communities, doublet_labels, cutoff = doubletdetection.classify(raw_counts) 

    # Analyze doublets
    doublets = np.where(scores[:raw_counts.shape[0]]>=cutoff)[0]
    print('Number of doublets = ', len(doublets))
    identified_doublets = raw_counts[doublets]
    XIST = np.where(identified_doublets[:,13750]>1)
    CD3D = np.where(identified_doublets[:,19801]>1)
    np.intersect1d(XIST,CD3D)
    XandC = np.intersect1d(XIST,CD3D)
    print("Number of doublets which express both XIST and CD3D = ", len(XandC))
    
    
if __name__ == '__main__':
    argv = sys.argv[1:]
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file", help='read csv data from FILE', metavar='FILE')

    (options, _args) = parser.parse_args()
    if options.file:
        FNAME = options.file

    main()