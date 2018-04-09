"""Doublet detection in single-cell RNA-seq data."""

import collections
import warnings
import logging

import numpy as np
import pandas as pd
import phenograph
from sklearn.decomposition import PCA
from scipy.stats import hypergeom


class BoostClassifier:
    """Classifier for across-type doublets in single-cell RNA-seq data.

    Parameters:
        boost_rate (float, optional): Proportion of cell population size to
            produce as synthetic doublets.
        knn (int, optional): Number of nearest neighbors used in Phenograph
            clustering. Ignored if 'k' specified in phenograph_parameters.
        n_pca (int, optional): Number of principal components used for clustering.
        n_top_var_genes (int, optional): Number of highest variance genes to
            use; other genes discarded. Will use all genes when non-positive.
        new_lib_as: (([int, int]) -> int, optional): Method to use in choosing
            new library size for synthetic doublets. Defaults to np.max. A common
            alternative is new_lib_as=np.mean.
        replace (bool, optional): If true, creates synthetic doublets by choosing
            parents with replacement
        n_jobs (int, optional): Number of cores to use. Default is -1: all
            available.
        phenograph_parameters (dict, optional): Phenograph parameters to
            override and their corresponding requested values.
        n_iters (int, optional): Defualt value is 25. Number of fit operations from
            which to produce p-values. This default is sufficient for convergence in
            the number of cells called doublets.
            NOTE: that most informational attributes will be set to None when
            running more than once.

    Attributes:
        communities_ (ndarray): Cluster ID for corresponding cell. 2D ndarary
            when n_iters > 1, with shape (n_iters, num_cells). -1 represents
            unclustered cells.
        labels_ (ndarray, ndims=1): 0 for singlet, 1 for detected doublet.
        parents_ (list of sequences of int): Parent cells' indexes for each
            synthetic doublet. When n_iters > 1, this is a list wrapping the
            results from each run.
        raw_synthetics_ (ndarray, ndims=2): Raw counts of synthetic doublets.
            Not produced when n_iters > 1.
        scores_ (ndarray): Doublet score for each cell. The fraction of a cell's
            cluster which is synthetic doublets. This is the mean across all
            runs when n_iter > 1.
        p_values_ (ndarray): Mean hypergeometric test value across n_iters runs
             for each cell.
        suggested_cutoff_ (float): Recommended cutoff to use (scores_ >= cutoff).
            Not produced when n_iters > 1.
        synth_communities_ (sequence of ints): Cluster ID for corresponding
            synthetic doublet. 2D ndarary when n_iters > 1, with shape
            (n_iters, num_cells * boost_rate).
    """

    def __init__(self, boost_rate=0.25, knn=30, n_pca=30, n_top_var_genes=10000, new_lib_as=np.max,
                 replace=False, n_jobs=-1, phenograph_parameters={'prune': True}, n_iters=25):
        self.boost_rate = boost_rate
        self.new_lib_as = new_lib_as
        self.replace = replace
        self.n_jobs = n_jobs
        self.n_iters = n_iters

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
        if (self.n_iters == 1) and (phenograph_parameters.get('prune') is True):
            warn_msg = ("Using phenograph parameter prune=False is strongly recommended when " +
                        "running only one iteration. Otherwise, expect many NaN labels.")
            warnings.warn(warn_msg)

        if not self.replace and self.boost_rate > 0.5:
            warn_msg = ("boost_rate is trimmed to 0.5 when replace=False." +
                        " Set replace=True to use greater boost rates.")
            warnings.warn(warn_msg)
            self.boost_rate = 0.5

        assert (self.n_top_var_genes == 0) or (self.n_pca <= self.n_top_var_genes), (
            "n_pca={0} cannot be larger than n_top_var_genes={1}".format(n_pca, n_top_var_genes))

    def fit(self, raw_counts, p_thresh=0.99, voter_thresh=0.9):
        """Identify doublets in single-cell RNA-seq count table raw_counts.

        Args:
            raw_counts (ndarray): Count table, oriented cells by genes.

        Sets:
            communities_, parents_ , raw_synthetics_, scores_, suggested_cutoff_

        Returns:
            labels_ (ndarray, ndims=1):  0 for singlet, 1 for detected doublet.
                Cells with p-values > 0.99 in >= 90% of the runs by default.
        """
        if self.n_top_var_genes > 0:
            if self.n_top_var_genes < raw_counts.shape[1]:
                gene_variances = np.var(raw_counts, axis=0)
                top_var_indexes = np.argsort(gene_variances)
                top_var_indexes = top_var_indexes[-self.n_top_var_genes:]
                raw_counts = raw_counts[:, top_var_indexes]

        self._raw_counts = raw_counts
        (self._num_cells, self._num_genes) = self._raw_counts.shape

        self._all_scores = np.zeros((self.n_iters, self._num_cells))
        self._all_p_values = np.zeros((self.n_iters, self._num_cells))
        all_communities = np.zeros((self.n_iters, self._num_cells))
        all_parents = []
        all_synth_communities = np.zeros((self.n_iters, int(self.boost_rate * self._num_cells)))

        for i in range(self.n_iters):
            self._all_scores[i], self._all_p_values[i] = self._one_fit()
            if self.n_iters > 1:
                all_communities[i] = self.communities_
                all_parents.append(self.parents_)
                all_synth_communities[i] = self.synth_communities_

        # NaNs correspond to unclustered cells. Ignore those runs.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=RuntimeWarning)
            self.scores_ = np.nanmean(self._all_scores, axis=0)
            self.p_values_ = np.nanmean(self._all_p_values, axis=0)
        if w:
            warnings.warn("One or more cells failed to join a cluster across all runs.",
                          category=RuntimeWarning)

        if self.n_iters > 1:
            self.communities_ = all_communities
            self.parents_ = all_parents
            self.synth_communities_ = all_synth_communities
            del self.raw_synthetics_
            with np.errstate(invalid='ignore'):  # Silence numpy warning about NaN comparison
                self._voting_average = np.mean(np.ma.masked_invalid(self._all_p_values) > p_thresh,
                                               axis=0)
                self.labels_ = np.ma.filled(self._voting_average >= voter_thresh, np.nan)
        else:
            # Find a cutoff score
            potential_cutoffs = np.unique(self.scores_)
            if len(potential_cutoffs) > 1:
                max_dropoff = np.argmax(potential_cutoffs[1:] - potential_cutoffs[:-1]) + 1
            else:   # Most likely pathological dataset, only one (or no) clusters
                max_dropoff = 0
            self.suggested_cutoff_ = potential_cutoffs[max_dropoff]
            with np.errstate(invalid='ignore'):  # Silence numpy warning about NaN comparison
                self.labels_ = self.scores_ >= self.suggested_cutoff_

        return self.labels_

    def _one_fit(self):
        print("\nCreating downsampled doublets...")
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
        community_IDs = orig_cells_per_comm.keys()
        community_scores = {i: float(synth_cells_per_comm[i]) /
                            (synth_cells_per_comm[i] + orig_cells_per_comm[i])
                            for i in community_IDs}
        scores = np.array([community_scores[i] for i in self.communities_])

        community_p_values = {i: hypergeom.cdf(synth_cells_per_comm[i], aug_counts.shape[0],
                                            self._synthetics.shape[0],
                                            synth_cells_per_comm[i] + orig_cells_per_comm[i])
                              for i in community_IDs}
        p_values = np.array([community_p_values[i] for i in self.communities_])

        if min_ID < 0:
            scores[self.communities_ == -1] = np.nan
            p_values[self.communities_ == -1] = np.nan

        return scores, p_values

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


def get_unique_genes(raw_counts, communities):
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
