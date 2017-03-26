# -*- coding: utf-8 -*-
"""
Created on Mar 10, 2017

@author: jonathanshor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pickle

DS = 'pbmc8k_dense.csv'


rawDF = pd.read_csv(DS, index_col=0)

#pickle.dump(rawDF, open( DS[:-4] + ".pickle", "wb" ) )
from sklearn.externals import joblib
joblib.dump(rawDF, DS[:-4] + '.pkl') 
# rawDF = joblib.load(DS[:-4] + '.pkl')    # To retrieve


#genehits = rawDF[rawDF > 0].count()     # for each gene, number of cells with >0 reads
#genehits.describe()
#cellhits = rawDF[rawDF > 0].count(axis=1)
#cellhits.describe()
#genenormcounts = rawDF.sum()            # for each gene, total (normed) reads
#genenormcounts.describe()
#cellnormcounts = rawDF.sum(axis=1)
#cellnormcounts.describe()

genestats = pd.DataFrame({'Hits' : rawDF[rawDF > 0].count(), # for each gene, number of cells with >0 reads
                          'Normalized Counts': rawDF.sum(),
                          'Sparsity' : rawDF[rawDF == 0].count() / rawDF.shape[0]})
genestats[genestats['Normalized Counts'] > genestats['Normalized Counts'].quantile(0.99)].plot()
genestats.describe()

cellstats = pd.DataFrame({'Hits' : rawDF[rawDF > 0].count(axis=1),
                          'Normalized Counts': rawDF.sum(axis=1),
                          'Sparsity' : rawDF[rawDF == 0].count(axis=1) / rawDF.shape[1]})
cellstats[cellstats['Normalized Counts'] < cellstats['Normalized Counts'].quantile(0.99)].hist()
cellstats.describe()

## Some PCA
from sklearn.decomposition import PCA

#pca = PCA(n_components=2)
pca = PCA()
pca.fit(rawDF)
rawTransformed = pd.DataFrame(pca.transform(rawDF), index=rawDF.index)


rawTransformed.plot(kind='scatter',x=0, y=1)     # First two components plotted
rawTransformed[range(10)].describe()



