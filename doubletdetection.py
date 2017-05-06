"""Doublet detection in single-cell RNA-seq data."""

import numpy as np
import pandas as pd
import phenograph
import collections
from sklearn.decomposition import PCA
from scipy.stats import binom


class BoostClassifier(object):
    """Classifier for doublets in single-cell RNA-seq data.

    Attributes:
        alpha1 (float): weighting of row1 in sum
        alpha2 (float): weighting of row2 in sum
        boost_rate (float): Proportion of cell population size to produce as
            synthetic doublets.
        communities (sequence of ints or None): Cluster ID for corresponding
            cell.
        conf_values (TYPE): Description
        downsample (bool): Use downsampling to generate synthetic doublets.
        duplicate_parents (bool): Create synthetics using one cell as both
            parents.
        knn (TYPE): Description
        n_pca (TYPE): Description
        p_val (TYPE): Description
        scores (ndarray): doublet score for each row in aug_counts as column vector.
        conf_values (TYPE)

    Deleted Attributes:
        parents (List of sequences of int): List of parent rows for each returned cell.
            Original cells are own single parent, given as singleton sequence.
    """

    def __init__(self, boost_rate=0.25, downsample=True, knn=20, n_pca=30, p_val=0.025):
        """Summary

        Args:
            boost_rate (float, optional): Description
            downsample (bool, optional): Description
            knn (int, optional): Description
            n_pca (int, optional): Description
            p_val (float, optional): Description
        """
        self.downsample=downsample
        self.duplicate_parents = False
        self.boost_rate=boost_rate
        self.knn=knn
        self.n_pca=n_pca
        self.p_val=p_val
        self.alpha1 = 0.6
        self.alpha2 = 0.6
        self.conf_values = None
        self._reduced_counts = None
        self.communities = None
        self.scores

    def fit(self, raw_counts):
        """Classifier for doublets in single-cell RNA-seq data.

        Args:
            raw_counts (ndarray): count table

        Returns:
            ndarray, ndim=2: Normalized augmented counts (original data and
                synthetic doublets).
            ndarray: doublet score for each row in aug_counts as column vector.
            TYPE: Phenograph community for each row in aug_counts
            List of sequences of int: List of parent rows for each returned cell.
                Original cells are own single parent, given as singleton sequence.
            float: Suggested cutoff score to identify doublets
        """
        self._raw_counts = raw_counts
        (self._num_cells, self._num_genes) = self._raw_counts.shape
        self._createLinearDoublets()

        # Normalize combined augmented set
        aug_counts = normalize_counts(np.append(self._raw_counts, self._synthetics, axis=0))
        self._norm_counts = aug_counts[:self._num_cells]
        self._synthetics = aug_counts[self._num_cells:]

        print("\nClustering mixed data set with Phenograph...\n")
        # Get phenograph results
        # TODO: logic to avoid PCA & Phenograph if previous calc'd & stored
        pca = PCA(n_components=self.n_pca)
        self._reduced_counts = pca.fit_transform(self._norm_counts)
        self.communities, _, _ = phenograph.cluster(self._reduced_counts, k=self.knn)
        min_ID = min(self.communities)
        if min_ID < 0:
            #TODO: remove this print?
            print("Adjusting community IDs up {} to avoid negative.".format(abs(min_ID)))
            self.communities = self.communities + abs(min_ID)
        print("Found communities [{0}, ... {2}], with sizes: {1}".format(min(self.communities),
              [np.count_nonzero(self.communities == i) for i in np.unique(self.communities)],
              max(self.communities)))
# self.communities now only for orig data cells, consistent with _*_counts attributes
        self._synth_communities = self.communities[self._num_cells:]
        synth_comm_sizes = collections.Counter(self._synth_communities)
        self.communities = self.communities[:self._num_cells]
        comm_sizes = collections.Counter(self.communities)
        print('\n')

        # Count number of fake doublets in each community and assign score
        community_scores = [float(synth_comm_sizes[i]) / (synth_comm_sizes[i] + comm_sizes[i])
                            for i in sorted(synth_comm_sizes | comm_sizes)]
        scores = [community_scores[i] for i in self.communities]
# TODO: Should these stay as column vectors?
        self.scores = np.array(scores).reshape(-1,1)
        synth_scores = [community_scores[i] for i in self._synth_communities]
        self._synth_scores = np.array(synth_scores).reshape(-1,1)

        if self.p_val is None:
            # Find a cutoff score
            potential_cutoffs = np.unique(community_scores)
            potential_cutoffs.sort(reverse=True)
            max_dropoff = 0
            for i in range(len(potential_cutoffs) - 1):
                dropoff = potential_cutoffs[i] - potential_cutoffs[i + 1]
                if dropoff > max_dropoff:
                    max_dropoff = dropoff
                    cutoff = potential_cutoffs[i]
            self.suggested_cutoff = cutoff
        else:
            # Find clusters with statistically significant synthetic doublet boosting
            conf_values = self._doubletConfidences(comm_sizes, synth_comm_sizes)
            self.significant = np.where(conf_values <= self.p_val)[0]

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
        """Append synthetic doublets to end of data.

        Sets _synthetics, .parents, ._norm_counts

        Args:
            normalize (bool, optional): normalize data before returning

        Returns:
            ndarray, ndims=2: augmented data
            ndarray, ndims=1: 0 for original data, 1 for fake doublet
            List of sequences of int: List of parent rows for each returned cell.
                Original cells are own single parent, given as singleton sequence.
        """
        # Number of synthetic doublets to add
        num_synths = int(self.boost_rate * self._num_cells)
        synthetic = np.zeros((num_synths, self._num_genes))

        parents = [[i] for i in range(self._num_cells)]

        for i in range(num_synths):
            row1 = np.random.randint(self._num_cells)
            if self.duplicate_parents:
                row2 = row1
            else:
                row2 = np.random.randint(self._num_cells)

            if self.downsample:
                new_row = self._downsampleCellPair(self._raw_counts[row1], self._raw_counts[row2])
            else:
                new_row = np.array(np.around(self.alpha1 * self._raw_counts[row1] + self.alpha2 *
                                             self._raw_counts[row2]), dtype=self._raw_counts.dtype)

            synthetic[i] = new_row
            parents.append([row1, row2])

        self._synthetics = synthetic
        self._parents = parents

    def _doubletConfidences(self, orig_community_sizes, synths_added):
        """Return significance for synthetic doublets assigned to each community.

        Args:
            orig_community_sizes (ndarray, ndims=1): Number of cells in each
                original community.
            synths_added (ndarray, ndims=1): Number of synthetics added to each
                community.

        Returns:
            ndarray, ndims=1: z-scores for each community.
        """
        assert orig_community_sizes.shape[0] == synths_added.shape[0], (
            "Original and added doublet sizes required for each cluster: {0} != {1}".format(
                orig_community_sizes.shape[0], synths_added.shape[0]))
        orig_cells = orig_community_sizes.reshape(-1,)
        num_synths = synths_added.reshape(-1,)

        p = orig_cells / np.sum(orig_cells, dtype=np.float_)
        N = np.sum(num_synths)
        k = num_synths
        sf = binom.sf(k, N, p)

        return sf

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


def load_data(FNAME, normalize=False, read_rows=None):
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
        doStandardize (bool, optional): Standardizes each gene column.

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
