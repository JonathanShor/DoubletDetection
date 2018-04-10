"""Doublet detection in single-cell RNA-seq data."""

import collections
import warnings
import logging

import numpy as np
import phenograph
from sklearn.decomposition import PCA
from scipy.stats import hypergeom


class BoostClassifier:
    """Classifier for doublets in single-cell RNA-seq data.

    Parameters:
        boost_rate (float, optional): Proportion of cell population size to
            produce as synthetic doublets.
        n_components (int, optional): Number of principal components used for
            clustering.
        n_top_var_genes (int, optional): Number of highest variance genes to
            use; other genes discarded. Will use all genes when zero.
        new_lib_as: (([int, int]) -> int, optional): Method to use in choosing
            library size for synthetic doublets. Defaults to np.max; append
            alternative is new_lib_as=np.mean.
        replace (bool, optional): If False, a cell will be selected as a
            synthetic doublet's parent no more than once.
        phenograph_parameters (dict, optional): Parameter dict to pass directly
            to Phenograph.
        n_iters (int, optional): Number of fit operations from which to collect
            p-values. Defualt value is 25.

    Attributes:
        all_p_values_ (ndarray): Hypergeometric test p-value per cell for cluster
            enrichment of synthetic doublets. Shape (n_iters, num_cells).
        communities_ (ndarray): Cluster ID for corresponding cell. Shape
            (n_iters, num_cells).
        labels_ (ndarray, ndims=1): 0 for singlet, 1 for detected doublet.
        parents_ (list of sequences of int): Parent cells' indexes for each
            synthetic doublet. A list wrapping the results from each run.
        scores_ (ndarray): The fraction of a cell's cluster that is synthetic
            doublets. Mean across all runs when n_iter > 1.
        suggested_score_cutoff_ (float): Cutoff used to classify cells when
            n_iters == 1 (scores_ >= cutoff). Not produced when n_iters > 1.
        synth_communities_ (sequence of ints): Cluster ID for corresponding
            synthetic doublet. Shape (n_iters, num_cells * boost_rate).
        top_var_genes_ (ndarray): Indices of the n_top_var_genes used. Not
            generated if n_top_var_genes <= 0.
        voting_average_ (ndarray): Fraction of iterations each cell is called a
            doublet.
    """

    def __init__(self, boost_rate=0.25, n_components=30, n_top_var_genes=10000, new_lib_as=np.max,
                 replace=False, phenograph_parameters={'prune': True}, n_iters=25):
        self.boost_rate = boost_rate
        self.new_lib_as = new_lib_as
        self.replace = replace
        self.n_iters = n_iters

        if n_components == 30 and n_top_var_genes > 0:
            # If user did not change n_components, silently cap it by n_top_var_genes if needed
            self.n_components = min(n_components, n_top_var_genes)
        else:
            self.n_components = n_components
        # Floor negative n_top_var_genes by 0
        self.n_top_var_genes = max(0, n_top_var_genes)

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

        assert (self.n_top_var_genes == 0) or (self.n_components <= self.n_top_var_genes), (
            "n_components={0} cannot be larger than n_top_var_genes={1}".format(n_components,
                                                                                n_top_var_genes))

    def fit(self, raw_counts):
        """Identify doublets in single-cell RNA-seq count table raw_counts.

        Args:
            raw_counts (ndarray): Count table, oriented cells by genes.

        Sets:
            communities_, parents_ , scores_, suggested_score_cutoff_

        Returns:
            labels_ (ndarray, ndims=1):  0 for singlet, 1 for detected doublet
        """
        if self.n_top_var_genes > 0:
            if self.n_top_var_genes < raw_counts.shape[1]:
                gene_variances = np.var(raw_counts, axis=0)
                top_var_indexes = np.argsort(gene_variances)
                self.top_var_genes_ = top_var_indexes[-self.n_top_var_genes:]
                raw_counts = raw_counts[:, self.top_var_genes_]

        self._raw_counts = raw_counts
        (self._num_cells, self._num_genes) = self._raw_counts.shape

        self._all_scores = np.zeros((self.n_iters, self._num_cells))
        self.all_p_values_ = np.zeros((self.n_iters, self._num_cells))
        all_communities = np.zeros((self.n_iters, self._num_cells))
        all_parents = []
        all_synth_communities = np.zeros((self.n_iters, int(self.boost_rate * self._num_cells)))

        for i in range(self.n_iters):
            self._all_scores[i], self.all_p_values_[i] = self._one_fit()
            all_communities[i] = self.communities_
            all_parents.append(self.parents_)
            all_synth_communities[i] = self.synth_communities_

        # Release unneeded large data vars
        del self._raw_counts
        del self._norm_counts
        del self._raw_synthetics
        del self._synthetics

        # NaNs correspond to unclustered cells. Ignore those runs.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=RuntimeWarning)
            self.scores_ = np.nanmean(self._all_scores, axis=0)
        if w:
            warnings.warn("One or more cells failed to join a cluster across all runs.",
                          category=RuntimeWarning)

        self.communities_ = all_communities
        self.parents_ = all_parents
        self.synth_communities_ = all_synth_communities

        return self

    def predict(self, p_thresh=0.99, voter_thresh=0.9):
        if self.n_iters > 1:
            with np.errstate(invalid='ignore'):  # Silence numpy warning about NaN comparison
                self.voting_average_ = np.mean(np.ma.masked_invalid(self.all_p_values_) > p_thresh,
                                               axis=0)
                self.labels_ = np.ma.filled(self.voting_average_ >= voter_thresh, np.nan)
                self.voting_average_ = np.ma.filled(self.voting_average_, np.nan)
        else:
            # Find a cutoff score
            potential_cutoffs = np.unique(self.scores_)
            if len(potential_cutoffs) > 1:
                max_dropoff = np.argmax(potential_cutoffs[1:] - potential_cutoffs[:-1]) + 1
            else:   # Most likely pathological dataset, only one (or no) clusters
                max_dropoff = 0
            self.suggested_score_cutoff_ = potential_cutoffs[max_dropoff]
            with np.errstate(invalid='ignore'):  # Silence numpy warning about NaN comparison
                self.labels_ = self.scores_ >= self.suggested_score_cutoff_

        return self.labels_

    def _one_fit(self):
        print("\nCreating downsampled doublets...")
        self._createDoublets()

        # Normalize combined augmented set
        print("Normalizing...")
        aug_counts = normalize_counts(np.append(self._raw_counts, self._raw_synthetics, axis=0))
        self._norm_counts = aug_counts[:self._num_cells]
        self._synthetics = aug_counts[self._num_cells:]

        print("Running PCA...")
        # Get phenograph results
        pca = PCA(n_components=self.n_components)
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

    def _createDoublets(self):
        """Create synthetic doublets.

        Sets .parents_
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

        self._raw_synthetics = synthetic
        self.parents_ = parents


def normalize_counts(raw_counts, pseudocount=0.1):
    """Normalize count array.

    Args:
        raw_counts (ndarray): count data
        pseudocount (float, optional): Count to add prior to log transform.

    Returns:
        ndarray: Normalized data.
    """
    # Sum across cells
    cell_sums = np.sum(raw_counts, axis=1)

    # Mutiply by median and divide each cell by cell sum
    median = np.median(cell_sums)
    normed = raw_counts * median / cell_sums[:, np.newaxis]

    normed = np.log(normed + pseudocount)

    return normed
