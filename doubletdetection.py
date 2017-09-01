"""Doublet detection in single-cell RNA-seq data."""

import numpy as np
import pandas as pd
import phenograph
import collections
from sklearn.decomposition import PCA


class BoostClassifier(object):
    """Classifier for doublets in single-cell RNA-seq data.

    Parameters:
        boost_rate (float, optional): Proportion of cell population size to
            produce as synthetic doublets.
        knn (int, optional): Number of nearest neighbors used in Phenograph
            clustering.
        n_pca (int, optional): Number of PCA components used for clustering.
        n_top_var_genes (int, optional): Number of highest variance genes to
            use; other genes discarded. Will use all genes when non-positive.
        jaccard (bool, optional): If true, uses Jaccard similarity in Phenograph
            if false, uses Gaussian kernel
        directed (bool, optional): If true, uses a directed graph in community
            detection in Phenograph
        replacement (bool, optional): If true, reates boosts by choosing parents
            with replacement
        downsample: (string, optional): Method to use in choosing new library size
            for boosts, options are "average" or "max"
        n_jobs (int, optional): Number of cores to use, default is -1 (all available)


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

    def __init__(self, boost_rate=0.25, knn=20, n_pca=30, n_top_var_genes=0, jaccard=True,
                 directed=False, replacement=True, downsample="average", n_jobs=-1):
        self.boost_rate = boost_rate
        self.knn = knn
        if n_pca == 30 and n_top_var_genes > 0:
            # If user did not change n_pca, silently cap it by n_top_var_genes if needed
            self.n_pca = min(n_pca, n_top_var_genes)
        else:
            self.n_pca = n_pca
        # Floor negative n_top_var_genes by 0
        self.n_top_var_genes = max(0, n_top_var_genes)
        self.jaccard = jaccard
        self.directed = directed
        self.replacement = replacement
        self.n_jobs = n_jobs
        self.downsample = downsample

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

        print("\nCreating downsampled doublets...\n")
        self._raw_counts = raw_counts
        (self._num_cells, self._num_genes) = self._raw_counts.shape

        self._createLinearDoublets()

        # Normalize combined augmented set
        print("\nNormalizing...\n")
        aug_counts = normalize_counts(np.append(self._raw_counts, self.raw_synthetics_, axis=0))
        self._norm_counts = aug_counts[:self._num_cells]
        self._synthetics = aug_counts[self._num_cells:]

        print("\nRunning PCA...\n")
        # Get phenograph results
        pca = PCA(n_components=self.n_pca)
        print("\nClustering augmented data set with Phenograph...\n")
        reduced_counts = pca.fit_transform(aug_counts)
        fullcommunities, _, _ = phenograph.cluster(
            reduced_counts, k=self.knn, jaccard=self.jaccard, directed=self.directed, n_jobs=self.n_jobs)
        min_ID = min(fullcommunities)
        if min_ID < 0:
            # print("Adjusting community IDs up {} to avoid negative.".format(abs(min_ID)))
            fullcommunities = fullcommunities + abs(min_ID)
        self.communities_ = fullcommunities[:self._num_cells]
        self.synth_communities_ = fullcommunities[self._num_cells:]
        print("Found communities [{0}, ... {2}], with sizes: {1}".format(min(fullcommunities),
                                                                         [np.count_nonzero(fullcommunities == i)
                                                                          for i in np.unique(fullcommunities)],
                                                                         max(fullcommunities)))
        print('\n')

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
        if self.downsample == "max":
            new_lib_size = int(max(lib1, lib2))
        else:
            new_lib_size = int((lib1 + lib2) / 2.0)
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
        if self.replacement is True:
            for i in range(num_synths):
                row1 = np.random.randint(self._num_cells)
                row2 = np.random.randint(self._num_cells)

                new_row = self._downsampleCellPair(self._raw_counts[row1], self._raw_counts[row2])

                synthetic[i] = new_row
                parents.append([row1, row2])
        else:
            choices = np.random.choice(self._num_cells, size=2 * num_synths)
            for i in range(0, len(choices), 2):
                row1 = choices[i]
                row2 = choices[i + 1]

                new_row = self._downsampleCellPair(self._raw_counts[row1], self._raw_counts[row2])

                synthetic[int(i / 2)] = new_row
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
