from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import phenograph

import os
import warnings
import numpy as np

import matplotlib
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
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


def convergence(clf, show=False, save=None, p_thresh=0.99, voter_thresh=0.9):
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
    doubs_per_run = []
    # Ignore numpy complaining about np.nan comparisons
    with np.errstate(invalid='ignore'):
        for i in range(clf.n_iters):
            cum_p_values = clf.all_p_values_[:i + 1]
            cum_vote_average = np.mean(np.ma.masked_invalid(cum_p_values) > p_thresh, axis=0)
            cum_doublets = np.ma.filled((cum_vote_average >= voter_thresh).astype(float), np.nan)
            doubs_per_run.append(np.nansum(cum_doublets))

    # Ignore warning for convergence plot
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", module="matplotlib", message="^tight_layout")

        f, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
        ax.plot(np.arange(len(doubs_per_run)), doubs_per_run)
        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel("Number of Predicted Doublets")
        ax.set_title('Predicted Doublets\n per Iteration')

        if show is True:
            plt.show()
        if isinstance(save, str):
            f.savefig(save, format='pdf', bbox_inches='tight')

    return f


def tsne(raw_counts, labels, n_components=30, n_jobs=-1, show=False, save=None,
         normalizer=normalize_counts):
    """Produce a tsne plot of the data with doublets in black.

        Count matrix is normalized and dimension reduced before plotting.

    Args:
        raw_counts (ndarray): cells by genes count matrix
        labels (ndarray): predicted doublets from predict method
        n_components (int, optional): number of PCs to use prior to TSNE
        n_jobs (int, optional): number of cores to use for TSNE, -1 for all
        show (bool, optional): If True, runs plt.show()
        save (str, optional): filename for saved figure,
            figure not saved by default
        normalizer ((ndarray) -> ndarray, optional): Method to normalize
            raw_counts. Defaults to normalize_counts, included in this package.
            Note: To use normalize_counts with its pseudocount parameter changed
            from the default 0.1 value to some positive float `new_var`, use:
            normalizer=lambda counts: doubletdetection.normalize_counts(counts,
            pseudocount=new_var)

    Returns:
        matplotlib figure
        ndarray: tsne reduction
    """
    norm_counts = normalizer(raw_counts)
    reduced_counts = PCA(n_components=n_components,
                         svd_solver='randomized').fit_transform(norm_counts)
    communities, _, _ = phenograph.cluster(reduced_counts)
    tsne_counts = TSNE(n_jobs=-1).fit_transform(reduced_counts)
    # Ensure only looking at positively identified doublets
    doublets = labels == 1

    fig, axes = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
    axes.scatter(tsne_counts[:, 0], tsne_counts[:, 1],
                 c=communities, cmap=plt.cm.tab20, s=1)
    axes.scatter(tsne_counts[:, 0][doublets], tsne_counts[:, 1][doublets],
                 s=3, edgecolor='k', facecolor='k')
    axes.set_title('Cells with Detected\n Doublets in Black')
    plt.xticks([])
    plt.yticks([])
    axes.set_xlabel('{} doublets out of {} cells.\n {}%  across-type doublet rate.'.format(
        np.sum(doublets),
        raw_counts.shape[0],
        np.round(100 * np.sum(doublets) / raw_counts.shape[0], 2)))

    if show is True:
        plt.show()
    if isinstance(save, str):
        fig.savefig(save, format='pdf', bbox_inches='tight')

    return fig, tsne_counts
