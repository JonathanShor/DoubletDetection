"""Doublet detection in single-cell RNA-seq data."""

import numpy as np
import pandas as pd
import phenograph
import collections
import warnings
import logging
from sklearn.decomposition import PCA


class BoostClassifier(object):
    """Classifier for doublets in single-cell RNA-seq data.

    Parameters:
        boost_rate (float, optional): Proportion of cell population size to
            produce as synthetic doublets.
        knn (int, optional): Number of nearest neighbors used in Phenograph
            clustering. Ignored if 'k' specified in phenograph_parameters.
        n_pca (int, optional): Number of PCA components used for clustering.
        n_top_var_genes (int, optional): Number of highest variance genes to
            use; other genes discarded. Will use all genes when non-positive.
        new_lib_as: (([int, int]) -> int, optional): Method to use in choosing
            new library size for boosts. Defaults to np.mean. A common
            alternative is new_lib_as=max.
        replace (bool, optional): If true, creates boosts by choosing parents
            with replacement
        n_jobs (int, optional): Number of cores to use. Default is -1: all
            available.
        phenograph_parameters (dict, optional): Phenograph parameters to
            override and their corresponding requested values.

    Attributes:
        communities_ (sequence of ints): Cluster ID for corresponding cell.
        labels_ (ndarray, ndims=1): 0 for singlet, 1 for detected doublet.
        parents_ (list of sequences of int): Parent cells' indexes for each
            synthetic doublet.
        raw_synthetics_ (ndarray, ndims=2): Raw counts of synthetic doublets.
        scores_ (ndarray): Doublet score for each cell.
        suggested_cutoff_ (float): Recommended cutoff to use (scores_ >= cutoff).
        synth_communities_ (sequence of ints): Cluster ID for corresponding
            synthetic doublet.
    """

    def __init__(self, boost_rate=0.25, knn=20, n_pca=30, n_top_var_genes=0, new_lib_as=np.mean,
                 replace=True, n_jobs=-1, phenograph_parameters=None):
        logging.debug(locals())
        self.boost_rate = boost_rate
        self.new_lib_as = new_lib_as
        self.replace = replace
        self.n_jobs = n_jobs

        if n_pca == 30 and n_top_var_genes > 0:
            # If user did not change n_pca, silently cap it by n_top_var_genes if needed
            self.n_pca = min(n_pca, n_top_var_genes)
        else:
            self.n_pca = n_pca
        # Floor negative n_top_var_genes by 0
        self.n_top_var_genes = max(0, n_top_var_genes)

        if phenograph_parameters:
            if 'k' not in phenograph_parameters:
                phenograph_parameters['k'] = knn
            else:
                logging.info("Ignoring 'knn' parameter, as 'k' provided in phenograph_parameters.")
            if 'n_jobs' not in phenograph_parameters:
                phenograph_parameters['n_jobs'] = n_jobs
        else:
            phenograph_parameters = {'k': knn, 'n_jobs': n_jobs}
        self.phenograph_parameters = phenograph_parameters

        if not self.replace and self.boost_rate > 0.5:
            warn_msg = ("boost_rate is trimmed to 0.5 when replace=False." +
                        " Set replace=True to use greater boost rates.")
            warnings.warn(warn_msg)
            self.boost_rate = 0.5

        assert (self.n_top_var_genes == 0) or (self.n_pca <= self.n_top_var_genes), (
            "n_pca={0} cannot be larger than n_top_var_genes={1}".format(n_pca, n_top_var_genes))

    def fit(self, raw_counts):
        """Identify doublets in single-cell RNA-seq count table raw_counts.

        Args:
            raw_counts (ndarray): Count table, oriented cells by genes.

        Sets:
            communities_, parents_ , raw_synthetics_, scores_, suggested_cutoff_

        Returns:
            labels_ (ndarray, ndims=1):  0 for singlet, 1 for detected doublet
        """
        if self.n_top_var_genes > 0:
            if self.n_top_var_genes < raw_counts.shape[1]:
                gene_variances = np.var(raw_counts, axis=0)
                top_var_indexes = np.argsort(gene_variances)
                top_var_indexes = top_var_indexes[-self.n_top_var_genes:]
                raw_counts = raw_counts[:, top_var_indexes]

        print("\nCreating downsampled doublets...")
        self._raw_counts = raw_counts
        (self._num_cells, self._num_genes) = self._raw_counts.shape

        self._createLinearDoublets()

        # Normalize combined augmented set
        print("Normalizing...")
        aug_counts = normalize_counts(np.append(self._raw_counts, self.raw_synthetics_, axis=0))
        self._norm_counts = aug_counts[:self._num_cells]
        self._synthetics = aug_counts[self._num_cells:]

        print("Running PCA...")
        # Get phenograph results
        pca = PCA(n_components=self.n_pca)
        print("Clustering augmented data set with Phenograph...\n")
        reduced_counts = pca.fit_transform(aug_counts)
        fullcommunities, _, _ = phenograph.cluster(reduced_counts, **self.phenograph_parameters)
        min_ID = min(fullcommunities)
        if min_ID < 0:
            logging.info("Adjusting community IDs up {} to avoid negative.".format(abs(min_ID)))
            fullcommunities = fullcommunities + abs(min_ID)
        self.communities_ = fullcommunities[:self._num_cells]
        self.synth_communities_ = fullcommunities[self._num_cells:]
        community_sizes = [np.count_nonzero(fullcommunities == i)
                           for i in np.unique(fullcommunities)]
        print("Found communities [{0}, ... {2}], with sizes: {1}\n".format(min(fullcommunities),
                                                                           community_sizes,
                                                                           max(fullcommunities)))

        # Count number of fake doublets in each community and assign score
        # Number of synth/orig cells in each cluster.
        synth_cells_per_comm = collections.Counter(self.synth_communities_)
        orig_cells_per_comm = collections.Counter(self.communities_)
        community_IDs = sorted(synth_cells_per_comm | orig_cells_per_comm)
        # self.orig_cells_per_comm_ = np.array([orig_cells_per_comm[i] for i in community_IDs])
        # self.synth_cells_per_comm_ = np.array([synth_cells_per_comm[i] for i in community_IDs])
        community_scores = [float(synth_cells_per_comm[i]) /
                            (synth_cells_per_comm[i] + orig_cells_per_comm[i])
                            for i in community_IDs]
        scores = [community_scores[i] for i in self.communities_]
        self.scores_ = np.array(scores)
        synth_scores = [community_scores[i] for i in self.synth_communities_]
        self._synth_scores = np.array(synth_scores)

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
        """Downsample the sum of two cells' gene expression profiles.

        Args:
            cell1 (ndarray, ndim=1): Gene count vector.
            cell2 (ndarray, ndim=1): Gene count vector.

        Returns:
            ndarray, ndim=1: Downsampled gene count vector.
        """
        new_cell = cell1 + cell2

        lib1 = np.sum(cell1)
        lib2 = np.sum(cell2)
        new_lib_size = int(self.new_lib_as([lib1, lib2]))
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

        choices = np.random.choice(self._num_cells, size=(num_synths, 2), replace=self.replace)
        for i, parent_pair in enumerate(choices):
            row1 = parent_pair[0]
            row2 = parent_pair[1]
            new_row = self._downsampleCellPair(self._raw_counts[row1], self._raw_counts[row2])

            synthetic[i] = new_row
            parents.append([row1, row2])

        self.raw_synthetics_ = synthetic
        self.parents_ = parents


def getUniqueGenes(raw_counts, communities):
    """Identify (any) genes unique to each community.

    Args:
        raw_counts (ndarray, ndims=2): Cell x genes counts numpy array.
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

    Note that the csv's first row and first column are assumed to be labels and
    are discarded.

    Args:
        FNAME (string): Pathname of file to load.
        normalize (bool, optional): Runs normalize_counts on table.
        read_rows (None or int, optional): If specified, load only first
            read_rows of data from FNAME.

    Returns:
        ndarray: Loaded table.
    """
    logging.info("Loading {}".format(FNAME))
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
