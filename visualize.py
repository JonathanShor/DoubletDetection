import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Create histograms from passed feature vector and doublet labels
def featureHist(feature, doublet_labels, axis_label='Library Size', file_name_prefix='library_size'):
	plot1 = plt.figure()
	plt.hist(feature, bins='auto', label='All Data', color='m')
	plt.legend(loc='upper left')
	plt.xlabel(axis_label)
	plt.ylabel('Sample Counts')
	plt.show()
	file_name_1 = file_name_prefix + '_whole.png'
	plt.savefig(file_name_1)
	plt.close(plot1)

	plot2 = plt.figure()
	plt.hist(feature[doublet_labels==0], bins='auto', label='Original', color='b')
	plt.hist(feature[doublet_labels==1], bins='auto', label='Synth', color='r')
	plt.legend(loc='upper left')
	plt.xlabel(axis_label)
	plt.ylabel('Sample Counts')
	plt.show()
	file_name_2 = file_name_prefix + '_separate.png'
	plt.savefig(file_name_2)
	plt.close(plot2)

def tsne_scatter(tsne_counts, doublet_labels, communities):
    
    #unique_communities = np.unique(communities)
    #colors = np.zeros((communities.shape[0],1))
    #for c in unique_communities:
        #colors[np.where(communities == c)[0]] = c/float(np.max(communities))
        
    set1i = LinearSegmentedColormap.from_list('set1i', plt.cm.Set1.colors, N=100)
    
    colors = communities
    x = tsne_counts[:,0]
    y = tsne_counts[:,1]
    plt.scatter(x,y, c=colors, s=10, cmap=set1i)
    doublets = np.where(doublet_labels==1)[0]
    plt.scatter(x[doublets], y[doublets], s=10, color='black')