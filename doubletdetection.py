"""Doublet detection in single-cell RNA-seq data."""

import numpy as np
import pandas as pd
import phenograph
import collections
from sklearn.decomposition import PCA
from scipy.stats import binom


class BoostClassifier(object):
    """Classifier for doublets in single-cell RNA-seq data.

    Parameters:
        boost_rate (float): Proportion of cell population size to produce as
            synthetic doublets.
        knn (int): value of k for input to Phenograph clustering
        n_pca (int): Number of PCA components to use for clustering

    Attributes:
        communities_ (sequence of ints): Cluster ID for corresponding
            cell.
        labels_ (ndarray, ndims=1):  0 for singlet, 1 for detected doublet
        parents_ (List of sequences of int): Parent indexes for each synthetic doublet.
        raw_synthetics_ (ndarray, ndims=2): Raw counts for augmented synthetic doublets
        scores_ (ndarray): Doublet score for each row in aug_counts as column vector.
        suggested_cutoff_ (float): Recommended cutoff to use (scores_ >= cutoff)
    """

    # def __init__(self, boost_rate=0.25, downsample=True, knn=20, n_pca=30, p_val=0.025):
    def __init__(self, boost_rate=0.25, knn=20, n_pca=30):
        # self.duplicate_parents = False
        self.boost_rate = boost_rate
        self.knn = knn
        self.n_pca = n_pca
        # self.p_val=p_val
        # self.conf_values = None

    def fit(self, raw_counts):
        """Classifier for doublets in single-cell RNA-seq data.

        Args:
            raw_counts (ndarray): count table

        Sets:
            ndarray, ndim=2: Normalized augmented counts (original data and
                synthetic doublets).
            ndarray: doublet score for each row in aug_counts.
            TYPE: Phenograph community for each row in aug_counts
            List of sequences of int: List of indexes for each synthetic doublet.
            float: Suggested cutoff score to identify doublets

        Returns:
            labels_ (ndarray, ndims=1):  0 for singlet, 1 for detected doublet
        """
        self._raw_counts = raw_counts
        (self._num_cells, self._num_genes) = self._raw_counts.shape
        self._createLinearDoublets()

        # Normalize combined augmented set
        aug_counts = normalize_counts(np.append(self._raw_counts, self.raw_synthetics_, axis=0))
        self._norm_counts = aug_counts[:self._num_cells]
        self._synthetics = aug_counts[self._num_cells:]

        print("\nClustering mixed data set with Phenograph...\n")
        # Get phenograph results
        pca = PCA(n_components=self.n_pca)
        # self._reduced_counts = pca.fit_transform(self._norm_counts)
        reduced_counts = pca.fit_transform(aug_counts)
        fullcommunities, _, _ = phenograph.cluster(reduced_counts, k=self.knn)
        min_ID = min(fullcommunities)
        if min_ID < 0:
            # print("Adjusting community IDs up {} to avoid negative.".format(abs(min_ID)))
            fullcommunities = fullcommunities + abs(min_ID)
        self.communities_ = fullcommunities[:self._num_cells]
        self._synth_communities = fullcommunities[self._num_cells:]
        print("Found communities [{0}, ... {2}], with sizes: {1}".format(min(fullcommunities),
              [np.count_nonzero(fullcommunities == i) for i in np.unique(fullcommunities)],
              max(fullcommunities)))
        print('\n')

        # Count number of fake doublets in each community and assign score
        # Number of cells (synth, orig) in each cluster.
        synth_cells_per_comm = collections.Counter(self._synth_communities)
        orig_cells_per_comm = collections.Counter(self.communities_)
        community_IDs = sorted(synth_cells_per_comm | orig_cells_per_comm)
        # self.orig_cells_per_comm_ = np.array([orig_cells_per_comm[i] for i in community_IDs])
        # self.synth_cells_per_comm_ = np.array([synth_cells_per_comm[i] for i in community_IDs])
        community_scores = [float(synth_cells_per_comm[i]) /
                            (synth_cells_per_comm[i] + orig_cells_per_comm[i])
                            for i in community_IDs]
        scores = [community_scores[i] for i in self.communities_]
        self.scores_ = np.array(scores)
        synth_scores = [community_scores[i] for i in self._synth_communities]
        self._synth_scores = np.array(synth_scores)

        # if self.p_val is None:
        #     # Find a cutoff score
        #     potential_cutoffs = np.unique(community_scores)
        #     potential_cutoffs.sort(reverse=True)
        #     max_dropoff = 0
        #     for i in range(len(potential_cutoffs) - 1):
        #         dropoff = potential_cutoffs[i] - potential_cutoffs[i + 1]
        #         if dropoff > max_dropoff:
        #             max_dropoff = dropoff
        #             cutoff = potential_cutoffs[i]
        #     self.suggested_cutoff_ = cutoff
        # else:
        #     # Find clusters with statistically significant synthetic doublet boosting
        #     conf_values = self._doubletConfidences(orig_cells_per_comm, synth_cells_per_comm)
        #     self.significant = np.where(conf_values <= self.p_val)[0]

        # Find a cutoff score
        potential_cutoffs = list(np.unique(community_scores))
        potential_cutoffs.sort(reverse=True)
        max_dropoff = 0
        for i in range(len(potential_cutoffs) - 1):
            dropoff = potential_cutoffs[i] - potential_cutoffs[i + 1]
            if dropoff > max_dropoff:
                max_dropoff = dropoff
                cutoff = potential_cutoffs[i]
            self.suggested_cutoff_ = cutoff

        self.labels_ = self.scores_ >= self.suggested_cutoff_
        return self.labels_

    def _downsampleCellPair(self, cell1, cell2):
        """Downsample the sum of two cell gene expression profiles.

        Args:
            cell1 (ndarray, ndim=1): Gene count vector.
            cell2 (ndarray, ndim=1): Gene count vector.

        Returns:
            ndarray, ndim=1: Downsampled gene count vector.
        """
        new_cell = cell1 + cell2

        lib1 = np.sum(cell1)
        lib2 = np.sum(cell2)
        new_lib_size = int(max(lib1, lib2))
        mol_ind = np.random.permutation(int(lib1 + lib2))[:new_lib_size]
        mol_ind += 1
        bins = np.append(np.zeros((1)), np.cumsum(new_cell))
        new_cell = np.histogram(mol_ind, bins)[0]

        return new_cell

    def _createLinearDoublets(self):
        """Create synthetic doublets.

        Sets .raw_synthetics_ and .parents_

        """
        # Number of synthetic doublets to add
        num_synths = int(self.boost_rate * self._num_cells)
        synthetic = np.zeros((num_synths, self._num_genes))

        parents = []
        for i in range(num_synths):
            row1 = np.random.randint(self._num_cells)
            # if self.duplicate_parents:
            #     row2 = row1
            # else:
            row2 = np.random.randint(self._num_cells)

            new_row = self._downsampleCellPair(self._raw_counts[row1], self._raw_counts[row2])

            synthetic[i] = new_row
            parents.append([row1, row2])

        self.raw_synthetics_ = synthetic
        self.parents_ = parents

    # def _doubletConfidences(self, orig_community_sizes, synths_added):
    #     """Return significance for synthetic doublets assigned to each community.
    #
    #     Args:
    #         orig_community_sizes (ndarray, ndims=1): Number of cells in each
    #             original community.
    #         synths_added (ndarray, ndims=1): Number of synthetics added to each
    #             community.
    #
    #     Returns:
    #         ndarray, ndims=1: z-scores for each community.
    #     """
    #     assert orig_community_sizes.shape[0] == synths_added.shape[0], (
    #         "Original and added doublet sizes required for each cluster: {0} != {1}".format(
    #             orig_community_sizes.shape[0], synths_added.shape[0]))
    #     orig_cells = orig_community_sizes.reshape(-1,)
    #     num_synths = synths_added.reshape(-1,)
    #
    #     p = orig_cells / np.sum(orig_cells, dtype=np.float_)
    #     N = np.sum(num_synths)
    #     k = num_synths
    #     sf = binom.sf(k, N, p)
    #
    #     return sf


def getUniqueGenes(self, raw_counts, communities):
    """Identify (any) genes unique to each community.

    Args:
        raw_counts (ndarray, ndims=2): Cell x genes counts nupmy array.
        communities (ndarray, shape=(raw_counts.shape[0],)): Community ID for
            each cell.

    Returns:
        ndarray, dtype=int: 1 for each gene unique to that community.
    """
    # Sum each community's genecounts, and stack up those gene profile vectors
    profiles = np.concatenate([np.sum(raw_counts[communities == i], axis=0, keepdims=True) for i in
                               np.unique(communities)], axis=0)

    binary = np.zeros_like(profiles)
    binary[profiles != 0] = 1

    # Only 1 - sum(everything) + 1 > 0
    uniques = binary - np.sum(binary, axis=0) + binary
    uniques[uniques < 0] = 0

    return uniques


def load_csv(FNAME, normalize=False, read_rows=None):
    """Load a csv table from the filesystem.

    Args:
        FNAME (string): Pathname of file to load.
        normalize (bool, optional): Runs normalize_counts on table.
        read_rows (None or int, optional): If specified, load only first
            read_rows of FNAME.

    Returns:
        ndarray: Loaded table.
    """
    counts = pd.read_csv(FNAME, index_col=0, nrows=read_rows).as_matrix()

    if normalize:
        counts = normalize_counts(counts)

    return counts


def normalize_counts(raw_counts, standardizeGenes=False):
    """Normalize count array using method in 10x pipeline.

    From http://www.nature.com/articles/ncomms14049.

    Args:
        raw_counts (ndarray): count data
        standardizeGenes (bool, optional): Standardizes each gene column.

    Returns:
        ndarray: Normalized data.
    """
    # Sum across cells
    cell_sums = np.sum(raw_counts, axis=1)

    # Mutiply by median and divide each cell by cell sum
    median = np.median(cell_sums)
    raw_counts = raw_counts * median / cell_sums[:, np.newaxis]

    raw_counts = np.log(raw_counts + 0.1)

    if standardizeGenes:
        # Normalize to have genes with mean 0 and std 1
        std = np.std(raw_counts, axis=0)[np.newaxis, :]

        # Fix potential divide by zero
        std[np.where(std == 0)[0]] = 1

        normed = (raw_counts - np.mean(raw_counts, axis=0)) / std
    else:
        normed = raw_counts

    return normed
