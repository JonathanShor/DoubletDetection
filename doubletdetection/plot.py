from sklearn.decomposition import PCA
import umap

import os
import warnings
import numpy as np
from sklearn.utils import check_array

import matplotlib

try:
    os.environ["DISPLAY"]
except KeyError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize_counts(raw_counts, pseudocount=0.1):
    """Normalize count array. Default normalizer used by BoostClassifier.

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

    normed = np.log10(normed + pseudocount)

    return normed


def convergence(clf, show=False, save=None, p_thresh=1e-7, voter_thresh=0.9):
    """Produce a plot showing number of cells called doublet per iter

    Args:
        clf (BoostClassifier object): Fitted classifier
        show (bool, optional): If True, runs plt.show()
        save (str, optional): filename for saved figure,
            figure not saved by default
        p_thresh (float, optional): hypergeometric test p-value threshold
            that determines per iteration doublet calls
        voter_thresh (float, optional): fraction of iterations a cell must
            be called a doublet

    Returns:
        matplotlib figure
    """
    log_p_thresh = np.log(p_thresh)
    doubs_per_run = []
    # Ignore numpy complaining about np.nan comparisons
    with np.errstate(invalid="ignore"):
        for i in range(clf.n_iters):
            cum_log_p_values = clf.all_log_p_values_[: i + 1]
            cum_vote_average = np.mean(
                np.ma.masked_invalid(cum_log_p_values) <= log_p_thresh, axis=0
            )
            cum_doublets = np.ma.filled(
                (cum_vote_average >= voter_thresh).astype(float), np.nan
            )
            doubs_per_run.append(np.nansum(cum_doublets))

    # Ignore warning for convergence plot
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", module="matplotlib", message="^tight_layout"
        )

        f, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
        ax.plot(np.arange(len(doubs_per_run)), doubs_per_run)
        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel("Number of Predicted Doublets")
        ax.set_title("Predicted Doublets per Iteration")

        if show is True:
            plt.show()
        if isinstance(save, str):
            f.savefig(save, format="pdf", bbox_inches="tight")

    return f


def umap_plot(
    raw_counts,
    labels,
    n_components=30,
    show=False,
    save=None,
    normalizer=normalize_counts,
    random_state=None,
):
    """Produce a umap plot of the data with doublets in black.

        Count matrix is normalized and dimension reduced before plotting.

    Args:
        raw_counts (array-like): Count matrix, oriented cells by genes.
        labels (ndarray): predicted doublets from predict method
        n_components (int, optional): number of PCs to use prior to UMAP
        show (bool, optional): If True, runs plt.show()
        save (str, optional): filename for saved figure,
            figure not saved by default
        normalizer ((ndarray) -> ndarray, optional): Method to normalize
            raw_counts. Defaults to normalize_counts, included in this package.
            Note: To use normalize_counts with its pseudocount parameter changed
            from the default 0.1 value to some positive float `new_var`, use:
            normalizer=lambda counts: doubletdetection.normalize_counts(counts,
            pseudocount=new_var)
        random_state (int, optional): If provided, passed to PCA and UMAP

    Returns:
        matplotlib figure
        ndarray: umap reduction
    """

    try:
        raw_counts = check_array(
            raw_counts, accept_sparse=False, force_all_finite=True, ensure_2d=True
        )
    except TypeError:  # Only catches sparse error. Non-finite & n_dims still raised.
        warnings.warn("Sparse raw_counts is automatically densified.")
        raw_counts = raw_counts.toarray()
    norm_counts = normalizer(raw_counts)
    reduced_counts = PCA(
        n_components=n_components, svd_solver="randomized", random_state=random_state
    ).fit_transform(norm_counts)
    umap_dr = umap.UMAP(random_state=random_state, min_dist=0.5).fit_transform(
        reduced_counts
    )
    # Ensure only looking at positively identified doublets
    doublets = labels == 1

    fig, axes = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
    axes.scatter(
        umap_dr[:, 0],
        umap_dr[:, 1],
        c="grey",
        cmap=plt.cm.tab20,
        s=1,
        label="predicted singlets",
    )
    axes.scatter(
        umap_dr[:, 0][doublets],
        umap_dr[:, 1][doublets],
        s=3,
        c="black",
        label="predicted doublets",
    )
    axes.axis("off")
    axes.legend(frameon=False)
    axes.set_title(
        "{} doublets out of {} cells\n {}% cross-type doublet rate".format(
            np.sum(doublets),
            raw_counts.shape[0],
            np.round(100 * np.sum(doublets) / raw_counts.shape[0], 2),
        )
    )

    if show is True:
        plt.show()
    if isinstance(save, str):
        fig.savefig(save, format="pdf", bbox_inches="tight")

    return fig, umap_dr


def threshold(
    clf,
    show=False,
    save=None,
    log10=True,
    log_p_grid=None,
    voter_grid=None,
    v_step=2,
    p_step=5,
):
    """Produce a plot showing number of cells called doublet across
       various thresholds

    Args:
        clf (BoostClassifier object): Fitted classifier
        show (bool, optional): If True, runs plt.show()
        save (str, optional): If provided, the figure is saved to this
            filepath.
        log10 (bool, optional): Use log 10 if true, natural log if false.
        log_p_grid (ndarray, optional): log p-value thresholds to use.
            Defaults to np.arange(-100, -1). log base decided by log10
        voter_grid (ndarray, optional): Voting thresholds to use. Defaults to
            np.arange(0.3, 1.0, 0.05).
        p_step (int, optional): number of xlabels to skip in plot
        v_step (int, optional): number of ylabels to skip in plot


    Returns:
        matplotlib figure
    """
    # Ignore numpy complaining about np.nan comparisons
    with np.errstate(invalid="ignore"):
        all_log_p_values_ = np.copy(clf.all_log_p_values_)
        if log10:
            all_log_p_values_ /= np.log(10)
        if log_p_grid is None:
            log_p_grid = np.arange(-100, -1)
        if voter_grid is None:
            voter_grid = np.arange(0.3, 1.0, 0.05)
        doubs_per_t = np.zeros((len(voter_grid), len(log_p_grid)))
        for i in range(len(voter_grid)):
            for j in range(len(log_p_grid)):
                voting_average = np.mean(
                    np.ma.masked_invalid(all_log_p_values_) <= log_p_grid[j], axis=0
                )
                labels = np.ma.filled(
                    (voting_average >= voter_grid[i]).astype(float), np.nan
                )
                doubs_per_t[i, j] = np.nansum(labels)

    # Ignore warning for convergence plot
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", module="matplotlib", message="^tight_layout"
        )

        f, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
        cax = ax.imshow(doubs_per_t, cmap="hot", aspect="auto")
        ax.set_xticks(np.arange(len(log_p_grid))[::p_step])
        ax.set_xticklabels(np.around(log_p_grid, 1)[::p_step], rotation="vertical")
        ax.set_yticks(np.arange(len(voter_grid))[::v_step])
        ax.set_yticklabels(np.around(voter_grid, 2)[::v_step])
        cbar = f.colorbar(cax)
        cbar.set_label("Predicted Doublets")
        if log10 is True:
            ax.set_xlabel("Log10 p-value")
        else:
            ax.set_xlabel("Log p-value")
        ax.set_ylabel("Voting Threshold")
        ax.set_title("Threshold Diagnostics")

    if show is True:
        plt.show()
    if save:
        f.savefig(save, format="pdf", bbox_inches="tight")

    return f
