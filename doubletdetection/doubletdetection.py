"""Doublet detection in single-cell RNA-seq data."""

import collections
import warnings

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1
from scipy.io import mmread
from scipy.stats import hypergeom
import scipy.sparse as sp_sparse
from scipy.sparse import csr_matrix
import tables
import scanpy as sc
import anndata
from tqdm.auto import tqdm
import phenograph


def load_10x_h5(file, genome):
    """Load count matrix in 10x H5 format
       Adapted from:
       https://support.10xgenomics.com/single-cell-gene-expression/software/
       pipelines/latest/advanced/h5_matrices

    Args:
        file (str): Path to H5 file
        genome (str): genome, top level h5 group

    Returns:
        ndarray: Raw count matrix.
        ndarray: Barcodes
        ndarray: Gene names
    """

    with tables.open_file(file, "r") as f:
        try:
            group = f.get_node(f.root, genome)
        except tables.NoSuchNodeError:
            print("That genome does not exist in this file.")
            return None
    # gene_ids = getattr(group, 'genes').read()
    gene_names = getattr(group, "gene_names").read()
    barcodes = getattr(group, "barcodes").read()
    data = getattr(group, "data").read()
    indices = getattr(group, "indices").read()
    indptr = getattr(group, "indptr").read()
    shape = getattr(group, "shape").read()
    matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)

    return matrix, barcodes, gene_names


def load_mtx(file):
    """Load count matrix in mtx format

    Args:
        file (str): Path to mtx file

    Returns:
        ndarray: Raw count matrix.
    """
    raw_counts = np.transpose(mmread(file))

    return raw_counts.tocsc()


class BoostClassifier:
    """Classifier for doublets in single-cell RNA-seq data.

    Parameters:
        boost_rate (float, optional): Proportion of cell population size to
            produce as synthetic doublets.
        n_components (int, optional): Number of principal components used for
            clustering.
        n_top_var_genes (int, optional): Number of highest variance genes to
            use; other genes discarded. Will use all genes when zero.
        replace (bool, optional): If False, a cell will be selected as a
            synthetic doublet's parent no more than once.
        use_phenograph (bool, optional): Set to False to disable PhenoGraph clustering
            in exchange for louvain clustering implemented in scanpy. Defaults to True.
        phenograph_parameters (dict, optional): Parameter dict to pass directly
            to PhenoGraph. Note that we change the PhenoGraph 'prune' default to
            True; you must specifically include 'prune': False here to change
            this. Only used when use_phenograph is True.
        n_iters (int, optional): Number of fit operations from which to collect
            p-values. Defualt value is 25.
        normalizer ((sp_sparse) -> ndarray): Method to normalize raw_counts.
            Defaults to normalize_counts, included in this package. Note: To use
            normalize_counts with its pseudocount parameter changed from the
            default 0.1 value to some positive float `new_var`, use:
            normalizer=lambda counts: doubletdetection.normalize_counts(counts,
            pseudocount=new_var)
        random_state (int, optional): If provided, passed to PCA and used to
            seedrandom seed numpy's RNG. NOTE: PhenoGraph does not currently
            admit a random seed, and so this will not guarantee identical
            results across runs.
        verbose (bool, optional): Set to False to silence all normal operation
            informational messages. Defaults to True.
        standard_scaling (bool, optional): Set to True to enable standard scaling
            of normalized count matrix prior to PCA. Recommended when not using
            Phenograph. Defaults to False.

    Attributes:
        all_log_p_values_ (ndarray): Hypergeometric test natural log p-value per
            cell for cluster enrichment of synthetic doublets. Shape (n_iters,
            num_cells).
        all_p_values_ (ndarray): DEPRECATED. Exponentiated all_log_p_values.
            Due to rounding point errors, use of all_log_p_values recommended.
            Will be removed in v3.0.
        all_scores_ (ndarray): The fraction of a cell's cluster that is
            synthetic doublets. Shape (n_iters, num_cells).
        communities_ (ndarray): Cluster ID for corresponding cell. Shape
            (n_iters, num_cells).
        labels_ (ndarray, ndims=1): 0 for singlet, 1 for detected doublet.
        parents_ (list of sequences of int): Parent cells' indexes for each
            synthetic doublet. A list wrapping the results from each run.
        suggested_score_cutoff_ (float): Cutoff used to classify cells when
            n_iters == 1 (scores >= cutoff). Not produced when n_iters > 1.
        synth_communities_ (sequence of ints): Cluster ID for corresponding
            synthetic doublet. Shape (n_iters, num_cells * boost_rate).
        top_var_genes_ (ndarray): Indices of the n_top_var_genes used. Not
            generated if n_top_var_genes <= 0.
        voting_average_ (ndarray): Fraction of iterations each cell is called a
            doublet.
    """

    def __init__(
        self,
        boost_rate=0.25,
        n_components=30,
        n_top_var_genes=10000,
        replace=False,
        use_phenograph=True,
        phenograph_parameters={"prune": True},
        n_iters=25,
        normalizer=None,
        random_state=0,
        verbose=False,
        standard_scaling=False,
    ):
        self.boost_rate = boost_rate
        self.replace = replace
        self.use_phenograph = use_phenograph
        self.n_iters = n_iters
        self.normalizer = normalizer
        self.random_state = random_state
        self.verbose = verbose
        self.standard_scaling = standard_scaling

        if self.random_state:
            np.random.seed(self.random_state)

        if n_components == 30 and n_top_var_genes > 0:
            # If user did not change n_components, silently cap it by n_top_var_genes if needed
            self.n_components = min(n_components, n_top_var_genes)
        else:
            self.n_components = n_components
        # Floor negative n_top_var_genes by 0
        self.n_top_var_genes = max(0, n_top_var_genes)

        if use_phenograph is True:
            if "prune" not in phenograph_parameters:
                phenograph_parameters["prune"] = True
            if ("verbosity" not in phenograph_parameters) and (not self.verbose):
                phenograph_parameters["verbosity"] = 1
            self.phenograph_parameters = phenograph_parameters
            if (self.n_iters == 1) and (phenograph_parameters.get("prune") is True):
                warn_msg = (
                    "Using phenograph parameter prune=False is strongly recommended when "
                    + "running only one iteration. Otherwise, expect many NaN labels."
                )
                warnings.warn(warn_msg)

        if not self.replace and self.boost_rate > 0.5:
            warn_msg = (
                "boost_rate is trimmed to 0.5 when replace=False."
                + " Set replace=True to use greater boost rates."
            )
            warnings.warn(warn_msg)
            self.boost_rate = 0.5

        assert (self.n_top_var_genes == 0) or (
            self.n_components <= self.n_top_var_genes
        ), "n_components={0} cannot be larger than n_top_var_genes={1}".format(
            n_components, n_top_var_genes
        )

    def fit(self, raw_counts):
        """Fits the classifier on raw_counts.

        Args:
            raw_counts (array-like): Count matrix, oriented cells by genes.

        Sets:
            all_scores_, all_p_values_, all_log_p_values_, communities_,
            top_var_genes, parents, synth_communities

        Returns:
            The fitted classifier.
        """

        raw_counts = check_array(
            raw_counts,
            accept_sparse="csr",
            force_all_finite=True,
            ensure_2d=True,
            dtype="float32",
        )

        if sp_sparse.issparse(raw_counts) is not True:
            if self.verbose:
                print("Sparsifying matrix.")
            raw_counts = csr_matrix(raw_counts)

        if self.n_top_var_genes > 0:
            if self.n_top_var_genes < raw_counts.shape[1]:
                gene_variances = (
                    np.array(raw_counts.power(2).mean(axis=0))
                    - (np.array(raw_counts.mean(axis=0))) ** 2
                )[0]
                top_var_indexes = np.argsort(gene_variances)
                self.top_var_genes_ = top_var_indexes[-self.n_top_var_genes :]
                # csc if faster for column indexing
                raw_counts = raw_counts.tocsc()
                raw_counts = raw_counts[:, self.top_var_genes_]
                raw_counts = raw_counts.tocsr()

        self._raw_counts = raw_counts
        (self._num_cells, self._num_genes) = self._raw_counts.shape
        if self.normalizer is None:
            # Memoize these; default normalizer treats these invariant for all synths
            self._lib_size = np.sum(raw_counts, axis=1).A1
            self._normed_raw_counts = self._raw_counts.copy()
            inplace_csr_row_normalize_l1(self._normed_raw_counts)

        self.all_scores_ = np.zeros((self.n_iters, self._num_cells))
        self.all_log_p_values_ = np.zeros((self.n_iters, self._num_cells))
        all_communities = np.zeros((self.n_iters, self._num_cells))
        all_parents = []
        all_synth_communities = np.zeros(
            (self.n_iters, int(self.boost_rate * self._num_cells))
        )

        for i in tqdm(range(self.n_iters)):
            if self.verbose:
                print("Iteration {:3}/{}".format(i + 1, self.n_iters))
            self.all_scores_[i], self.all_log_p_values_[i] = self._one_fit()
            all_communities[i] = self.communities_
            all_parents.append(self.parents_)
            all_synth_communities[i] = self.synth_communities_

        # Release unneeded large data vars
        del self._raw_counts
        del self._norm_counts
        del self._raw_synthetics
        del self._synthetics
        if self.normalizer is None:
            del self._normed_raw_counts
            del self._lib_size

        self.communities_ = all_communities
        self.parents_ = all_parents
        self.synth_communities_ = all_synth_communities
        self.all_p_values_ = np.exp(self.all_log_p_values_)

        return self

    def predict(self, p_thresh=1e-7, voter_thresh=0.9):
        """Produce doublet calls from fitted classifier

        Args:
            p_thresh (float, optional): hypergeometric test p-value threshold
                that determines per iteration doublet calls
            voter_thresh (float, optional): fraction of iterations a cell must
                be called a doublet

        Sets:
            labels_ and voting_average_ if n_iters > 1.
            labels_ and suggested_score_cutoff_ if n_iters == 1.

        Returns:
            labels_ (ndarray, ndims=1):  0 for singlet, 1 for detected doublet
        """
        log_p_thresh = np.log(p_thresh)
        if self.n_iters > 1:
            with np.errstate(
                invalid="ignore"
            ):  # Silence numpy warning about NaN comparison
                self.voting_average_ = np.mean(
                    np.ma.masked_invalid(self.all_log_p_values_) <= log_p_thresh, axis=0
                )
                self.labels_ = np.ma.filled(
                    (self.voting_average_ >= voter_thresh).astype(float), np.nan
                )
                self.voting_average_ = np.ma.filled(self.voting_average_, np.nan)
        else:
            # Find a cutoff score
            potential_cutoffs = np.unique(self.all_scores_[~np.isnan(self.all_scores_)])
            if len(potential_cutoffs) > 1:
                max_dropoff = (
                    np.argmax(potential_cutoffs[1:] - potential_cutoffs[:-1]) + 1
                )
            else:  # Most likely pathological dataset, only one (or no) clusters
                max_dropoff = 0
            self.suggested_score_cutoff_ = potential_cutoffs[max_dropoff]
            with np.errstate(
                invalid="ignore"
            ):  # Silence numpy warning about NaN comparison
                self.labels_ = self.all_scores_[0, :] >= self.suggested_score_cutoff_
            self.labels_[np.isnan(self.all_scores_)[0, :]] = np.nan

        return self.labels_

    def _one_fit(self):
        if self.verbose:
            print("\nCreating synthetic doublets...")
        self._createDoublets()

        # Normalize combined augmented set
        if self.verbose:
            print("Normalizing...")
        if self.normalizer is not None:
            aug_counts = self.normalizer(
                sp_sparse.vstack((self._raw_counts, self._raw_synthetics))
            )
        else:
            # Follows doubletdetection.plot.normalize_counts, but uses memoized normed raw_counts
            synth_lib_size = np.sum(self._raw_synthetics, axis=1).A1
            aug_lib_size = np.concatenate([self._lib_size, synth_lib_size])
            normed_synths = self._raw_synthetics.copy()
            inplace_csr_row_normalize_l1(normed_synths)
            aug_counts = sp_sparse.vstack((self._normed_raw_counts, normed_synths))
            aug_counts = np.log(aug_counts.A * np.median(aug_lib_size) + 0.1)

        self._norm_counts = aug_counts[: self._num_cells]
        self._synthetics = aug_counts[self._num_cells :]

        aug_counts = anndata.AnnData(aug_counts)
        aug_counts.obs["n_counts"] = aug_lib_size
        if self.standard_scaling is True:
            sc.pp.scale(aug_counts, max_value=15)

        if self.verbose:
            print("Running PCA...")
        sc.tl.pca(aug_counts, n_comps=self.n_components, random_state=self.random_state)
        if self.verbose:
            print("Clustering augmented data set...\n")
        sc.pp.neighbors(
            aug_counts, random_state=self.random_state, method="umap", n_neighbors=10
        )
        if self.use_phenograph:
            fullcommunities, _, _ = phenograph.cluster(
                aug_counts.obsm["X_pca"], **self.phenograph_parameters
            )
        else:
            sc.tl.louvain(
                aug_counts, random_state=self.random_state, resolution=4, directed=False
            )
            fullcommunities = np.array(aug_counts.obs["louvain"], dtype=int)
        min_ID = min(fullcommunities)
        self.communities_ = fullcommunities[: self._num_cells]
        self.synth_communities_ = fullcommunities[self._num_cells :]
        community_sizes = [
            np.count_nonzero(fullcommunities == i) for i in np.unique(fullcommunities)
        ]
        if self.verbose:
            print(
                "Found clusters [{0}, ... {2}], with sizes: {1}\n".format(
                    min(fullcommunities), community_sizes, max(fullcommunities)
                )
            )

        # Count number of fake doublets in each community and assign score
        # Number of synth/orig cells in each cluster.
        synth_cells_per_comm = collections.Counter(self.synth_communities_)
        orig_cells_per_comm = collections.Counter(self.communities_)
        community_IDs = orig_cells_per_comm.keys()
        community_scores = {
            i: float(synth_cells_per_comm[i])
            / (synth_cells_per_comm[i] + orig_cells_per_comm[i])
            for i in community_IDs
        }
        scores = np.array([community_scores[i] for i in self.communities_])

        community_log_p_values = {
            i: hypergeom.logsf(
                synth_cells_per_comm[i],
                aug_counts.shape[0],
                self._synthetics.shape[0],
                synth_cells_per_comm[i] + orig_cells_per_comm[i],
            )
            for i in community_IDs
        }
        log_p_values = np.array([community_log_p_values[i] for i in self.communities_])

        if min_ID < 0:
            scores[self.communities_ == -1] = np.nan
            log_p_values[self.communities_ == -1] = np.nan

        return scores, log_p_values

    def _createDoublets(self):
        """Create synthetic doublets.

        Sets .parents_
        """
        # Number of synthetic doublets to add
        num_synths = int(self.boost_rate * self._num_cells)

        # Parent indices
        choices = np.random.choice(
            self._num_cells, size=(num_synths, 2), replace=self.replace
        )
        parents = [list(p) for p in choices]

        parent0 = self._raw_counts[choices[:, 0], :]
        parent1 = self._raw_counts[choices[:, 1], :]
        synthetic = parent0 + parent1

        self._raw_synthetics = synthetic
        self.parents_ = parents
