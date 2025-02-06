"""Doublet detection in single-cell RNA-seq data."""

import collections
from collections.abc import Callable
import io
import warnings
from contextlib import redirect_stdout

import anndata
import numpy as np
from numpy.typing import NDArray
import phenograph
import scanpy as sc
import scipy.sparse as sp_sparse
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom
from sklearn.utils import check_array
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1
from tqdm.auto import tqdm


class BoostClassifier:
    """Classifier for doublets in single-cell RNA-seq data.

    Parameters:
        boost_rate: Proportion of cell population size to produce as synthetic doublets.
        n_components: Number of principal components used for clustering.
        n_top_var_genes: Number of highest variance genes to use. Other genes are
            discarded. Will use all genes when zero.
        replace: If False, a cell will be selected as a synthetic doublet's parent
            no more than once.
        clustering_algorithm: One of "louvain", "leiden", or "phenograph". "louvain"
            and "leiden" refer to the scanpy implementations.
        clustering_kwargs: Keyword args to pass directly to clustering algorithm.
            Note that PhenoGraph 'prune' default is changed to True. For Louvain and
            Leiden clustering, we set `directed=False` and `resolution=4`. Include
            these params explicitly to change them. Do not override `random_state`
            and `key_added` for Louvain/Leiden.
        n_iters: Number of fit operations from which to collect p-values. Default is 25.
        normalizer: Method to normalize raw_counts. Defaults to normalize_counts from
            this package. To use normalize_counts with a different pseudocount value,
            use: `lambda counts: doubletdetection.normalize_counts(counts,
            pseudocount=new_value)`
        pseudocount: Pseudocount used in normalize_counts. Using 1 with
            standard_scaling=False makes the classifier more memory efficient but may
            detect fewer doublets.
        random_state: Passed to PCA and doublet parent creation. Note: PhenoGraph does not
            support random seeds, so identical results aren't guaranteed across runs.
        verbose: Set to False to silence informational messages. Defaults to True.
        standard_scaling: Enable standard scaling of normalized count matrix prior to
            PCA. Recommended when not using Phenograph. Defaults to False.
        n_jobs: Number of jobs to use. Speeds up neighbor computation.

    Attributes:
        all_log_p_values_: Hypergeometric test natural log p-value per cell for
            cluster enrichment of synthetic doublets. Use for thresholding.
            Shape (n_iters, num_cells).
        all_scores_: The fraction of a cell's cluster that is synthetic doublets.
            Shape (n_iters, num_cells).
        communities_: Cluster ID for corresponding cell. Shape (n_iters, num_cells).
        labels_: 0 for singlet, 1 for detected doublet.
        parents_: Parent cells' indexes for each synthetic doublet. A list wrapping
            the results from each run.
        suggested_score_cutoff_: Cutoff used to classify cells when n_iters == 1
            (scores >= cutoff). Not produced when n_iters > 1.
        synth_communities_: Cluster ID for corresponding synthetic doublet.
            Shape (n_iters, num_cells * boost_rate).
        top_var_genes_: Indices of the n_top_var_genes used. Not generated if
            n_top_var_genes <= 0.
        voting_average_: Fraction of iterations each cell is called a doublet.
    """

    def __init__(
        self,
        boost_rate: float = 0.25,
        n_components: int = 30,
        n_top_var_genes: int = 10000,
        replace: bool = False,
        clustering_algorithm: str = "phenograph",
        clustering_kwargs: dict | None = None,
        n_iters: int = 10,
        normalizer: Callable | None = None,
        pseudocount: float = 0.1,
        random_state: int = 0,
        verbose: bool = False,
        standard_scaling: bool = False,
        n_jobs: int = 1,
    ) -> None:
        self.boost_rate = boost_rate
        self.replace = replace
        self.clustering_algorithm = clustering_algorithm
        self.n_iters = n_iters
        self.normalizer = normalizer
        self.random_state = random_state
        self.verbose = verbose
        self.standard_scaling = standard_scaling
        self.n_jobs = n_jobs
        self.pseudocount = pseudocount
        self.rng = np.random.default_rng(self.random_state)

        if self.clustering_algorithm not in ["louvain", "phenograph", "leiden"]:
            raise ValueError(
                "Clustering algorithm needs to be one of ['louvain', 'phenograph', 'leiden']"
            )
        if self.clustering_algorithm == "leiden":
            warnings.warn("Leiden clustering is experimental and results have not been validated.")

        if n_components == 30 and n_top_var_genes > 0:
            # If user did not change n_components, silently cap it by n_top_var_genes if needed
            self.n_components = min(n_components, n_top_var_genes)
        else:
            self.n_components = n_components
        # Floor negative n_top_var_genes by 0
        self.n_top_var_genes = max(0, n_top_var_genes)

        self.clustering_kwargs = (
            {} if not isinstance(clustering_kwargs, dict) else clustering_kwargs
        )
        self._set_clustering_kwargs()

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

    def fit(self, raw_counts: NDArray | sp_sparse.csr_matrix) -> "BoostClassifier":
        """Fits the classifier on raw_counts.

        Args:
            raw_counts: Count matrix, oriented cells by genes.

        Sets:
            all_scores_, all_log_p_values_, communities_,
            top_var_genes, parents, synth_communities

        Returns:
            The fitted classifier.
        """

        raw_counts = check_array(
            raw_counts,
            accept_sparse="csr",
            ensure_all_finite=True,
            ensure_2d=True,
            dtype="float32",
        )

        if sp_sparse.issparse(raw_counts) is not True:
            if self.verbose:
                print("Sparsifying matrix.")
            raw_counts = csr_matrix(raw_counts)

        old_n_jobs = sc.settings.n_jobs
        sc.settings.n_jobs = self.n_jobs

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
            self._lib_size = np.asarray(np.sum(raw_counts, axis=1)).ravel()
            self._normed_raw_counts = self._raw_counts.copy()
            inplace_csr_row_normalize_l1(self._normed_raw_counts)

        self.all_scores_ = np.zeros((self.n_iters, self._num_cells))
        self.all_log_p_values_ = np.zeros((self.n_iters, self._num_cells))
        all_communities = np.zeros((self.n_iters, self._num_cells))
        all_parents = []
        all_synth_communities = np.zeros((self.n_iters, int(self.boost_rate * self._num_cells)))

        for i in tqdm(range(self.n_iters)):
            if self.verbose:
                print("Iteration {:3}/{}".format(i + 1, self.n_iters))
            self.all_scores_[i], self.all_log_p_values_[i] = self._one_fit()
            all_communities[i] = self.communities_
            all_parents.append(self.parents_)
            all_synth_communities[i] = self.synth_communities_

        # Release unneeded large data vars
        del self._raw_counts
        del self._raw_synthetics
        if self.normalizer is None:
            del self._normed_raw_counts
            del self._lib_size

        # reset scanpy n_jobs
        sc.settings.n_jobs = old_n_jobs

        self.communities_ = all_communities
        self.parents_ = all_parents
        self.synth_communities_ = all_synth_communities

        return self

    def predict(self, p_thresh: float = 1e-7, voter_thresh: float = 0.9) -> NDArray:
        """Produce doublet calls from fitted classifier

        Args:
            p_thresh: hypergeometric test p-value threshold
                that determines per iteration doublet calls
            voter_thresh: fraction of iterations a cell must
                be called a doublet

        Sets:
            labels_ and voting_average_ if n_iters > 1.
            labels_ and suggested_score_cutoff_ if n_iters == 1.

        Returns:
            labels_ (ndarray, ndims=1):  0 for singlet, 1 for detected doublet
        """
        log_p_thresh = np.log(p_thresh)
        if self.n_iters > 1:
            with np.errstate(invalid="ignore"):  # Silence numpy warning about NaN comparison
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
                max_dropoff = np.argmax(potential_cutoffs[1:] - potential_cutoffs[:-1]) + 1
            else:  # Most likely pathological dataset, only one (or no) clusters
                max_dropoff = 0
            self.suggested_score_cutoff_ = potential_cutoffs[max_dropoff]
            with np.errstate(invalid="ignore"):  # Silence numpy warning about NaN comparison
                self.labels_ = self.all_scores_[0, :] >= self.suggested_score_cutoff_
            self.labels_[np.isnan(self.all_scores_)[0, :]] = np.nan

        return self.labels_

    def doublet_score(self) -> NDArray:
        """Produce doublet scores

        The doublet score is the average negative log p-value of doublet enrichment
        averaged over the iterations. Higher means more likely to be doublet.

        Returns:
            scores (ndarray, ndims=1):  Average negative log p-value over iterations
        """

        if self.n_iters > 1:
            with np.errstate(invalid="ignore"):  # Silence numpy warning about NaN comparison
                avg_log_p = np.mean(np.ma.masked_invalid(self.all_log_p_values_), axis=0)
        else:
            avg_log_p = self.all_log_p_values_[0]

        return -avg_log_p

    def _one_fit(self) -> tuple[NDArray, NDArray]:
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
            synth_lib_size = np.asarray(np.sum(self._raw_synthetics, axis=1)).ravel()
            aug_lib_size = np.concatenate([self._lib_size, synth_lib_size])
            normed_synths = self._raw_synthetics.copy()
            inplace_csr_row_normalize_l1(normed_synths)
            aug_counts = sp_sparse.vstack((self._normed_raw_counts, normed_synths))
            scaled_aug_counts = aug_counts * np.median(aug_lib_size)
            if self.pseudocount != 1:
                aug_counts = np.log(scaled_aug_counts.toarray() + self.pseudocount)
            else:
                aug_counts = np.log1p(scaled_aug_counts)
            del scaled_aug_counts

        aug_counts = anndata.AnnData(aug_counts)
        aug_counts.obs["n_counts"] = aug_lib_size
        if self.standard_scaling is True:
            sc.pp.scale(aug_counts, max_value=15)

        if self.verbose:
            print("Running PCA...")
        # "auto" solver faster for dense matrices
        solver = "arpack" if sp_sparse.issparse(aug_counts.X) else "auto"
        sc.tl.pca(
            aug_counts,
            n_comps=self.n_components,
            random_state=self.random_state,
            svd_solver=solver,
        )
        if self.verbose:
            print("Clustering augmented data set...\n")
        if self.clustering_algorithm == "phenograph":
            f = io.StringIO()
            with redirect_stdout(f):
                fullcommunities, _, _ = phenograph.cluster(
                    aug_counts.obsm["X_pca"], n_jobs=self.n_jobs, **self.clustering_kwargs
                )
            out = f.getvalue()
            if self.verbose:
                print(out)
        else:
            if self.clustering_algorithm == "louvain":
                clus = sc.tl.louvain
            else:
                clus = sc.tl.leiden
            sc.pp.neighbors(
                aug_counts,
                random_state=self.random_state,
                method="umap",
                n_neighbors=10,
            )
            clus(
                aug_counts,
                key_added="clusters",
                random_state=self.random_state,
                **self.clustering_kwargs,
            )
            fullcommunities = np.array(aug_counts.obs["clusters"], dtype=int)
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
            i: float(synth_cells_per_comm[i]) / (synth_cells_per_comm[i] + orig_cells_per_comm[i])
            for i in community_IDs
        }
        scores = np.array([community_scores[i] for i in self.communities_])

        community_log_p_values = {
            i: hypergeom.logsf(
                synth_cells_per_comm[i],
                aug_counts.shape[0],
                normed_synths.shape[0],
                synth_cells_per_comm[i] + orig_cells_per_comm[i],
            )
            for i in community_IDs
        }
        log_p_values = np.array([community_log_p_values[i] for i in self.communities_])

        if min_ID < 0:
            scores[self.communities_ == -1] = np.nan
            log_p_values[self.communities_ == -1] = np.nan

        return scores, log_p_values

    def _createDoublets(self) -> None:
        """Create synthetic doublets.

        Sets .parents_
        """
        # Number of synthetic doublets to add
        num_synths = int(self.boost_rate * self._num_cells)

        # Parent indices
        choices = self.rng.choice(self._num_cells, size=(num_synths, 2), replace=self.replace)
        parents = [list(p) for p in choices]

        parent0 = self._raw_counts[choices[:, 0], :]
        parent1 = self._raw_counts[choices[:, 1], :]
        synthetic = parent0 + parent1

        self._raw_synthetics = synthetic
        self.parents_ = parents

    def _set_clustering_kwargs(self) -> None:
        """Sets .clustering_kwargs"""
        if self.clustering_algorithm == "phenograph":
            if "prune" not in self.clustering_kwargs:
                self.clustering_kwargs["prune"] = True
            self.clustering_kwargs = self.clustering_kwargs
            if (self.n_iters == 1) and (self.clustering_kwargs.get("prune") is True):
                warn_msg = (
                    "Using phenograph parameter prune=False is strongly recommended when "
                    + "running only one iteration. Otherwise, expect many NaN labels."
                )
                warnings.warn(warn_msg)
        else:
            if "directed" not in self.clustering_kwargs:
                self.clustering_kwargs["directed"] = False
            if "resolution" not in self.clustering_kwargs:
                self.clustering_kwargs["resolution"] = 4
            if "key_added" in self.clustering_kwargs:
                raise ValueError("'key_added' param cannot be overriden")
            if "random_state" in self.clustering_kwargs:
                raise ValueError(
                    "'random_state' param cannot be overriden. Please use classifier 'random_state'."
                )
