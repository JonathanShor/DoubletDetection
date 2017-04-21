import matplotlib.pylot as plt

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
