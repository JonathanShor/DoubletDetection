from .doubletdetection import normalize_counts
from sklearn.decomposition import PCA
import bhtsne
import phenograph

import os
import warnings
from cycler import cycler
import numpy as np
# Ignore warning for convergence plot
np.warnings.filterwarnings('ignore')

import matplotlib
from matplotlib import font_manager
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
with warnings.catch_warnings():
    # catch warnings that system can't find fonts
    warnings.simplefilter('ignore')
    fm = font_manager.fontManager
    fm.findfont('Raleway')
    fm.findfont('Lato')

warnings.filterwarnings(
    action="ignore", module="matplotlib", message="^tight_layout")

dark_gray = '.15'

_colors = ['#4C72B0', '#55A868', '#C44E52',
           '#8172B2', '#CCB974', '#64B5CD']

style_dictionary = {
    'figure.figsize': (3, 3),
    'figure.facecolor': 'white',

    'figure.dpi': 200,
    'savefig.dpi': 200,

    'text.color': 'k',

    "legend.frameon": False,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,

    'font.family': ['sans-serif'],
    'font.serif': ['Computer Modern Roman', 'serif'],
    'font.monospace': ['Inconsolata', 'Computer Modern Typewriter', 'Monaco'],
    'font.sans-serif': ['Helvetica', 'Lato', 'sans-serif'],

    'patch.facecolor': _colors[0],
    'patch.edgecolor': 'none',

    'grid.linestyle': "-",

    'axes.labelcolor': dark_gray,
    'axes.facecolor': 'white',
    'axes.linewidth': 1.,
    'axes.grid': False,
    'axes.axisbelow': False,
    'axes.edgecolor': dark_gray,
    'axes.prop_cycle': cycler('color', _colors),

    'lines.solid_capstyle': 'round',
    'lines.color': _colors[0],
    'lines.markersize': 4,

    'image.cmap': 'viridis',
    'image.interpolation': 'none',

    'xtick.direction': 'in',
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'xtick.color': dark_gray,

    'ytick.direction': 'in',
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    "ytick.color": dark_gray,

}

matplotlib.rcParams.update(style_dictionary)


def convergence(clf, show=False, save=False, p_thresh=0.99, voter_thresh=0.9):
    """Produce a plot showing number of cells called doublet per iter

    Args:
        clf (BoostClassifier object): Fitted classifier
        show (bool, optional): If True, runs plt.show()
        save (bool, optional): If True, saves plot as pdf
        p_thresh(float, optional): p-value threshold
        voter_thresh(float, optional):

    Returns:
        ndarray: Normalized data.
    """
    doubs_per_run = []
    for i in range(clf.n_iters):
        cum_p_values = clf.all_p_values_[:i + 1]
        cum_vote_average = np.mean(
            np.ma.masked_invalid(cum_p_values) > p_thresh, axis=0)
        cum_doublets = np.ma.filled(cum_vote_average >= voter_thresh, np.nan)
        doubs_per_run.append(np.sum(cum_doublets))

    plt.figure()
    plt.plot(np.arange(len(doubs_per_run)), doubs_per_run)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Number of cells called doublets")
    plt.title('Convergence of Number of Cells\n Called Doublets over Iterations')

    if show is True:
        plt.show()
    if save is True:
        plt.savefig('doublet_convergence.pdf', format='pdf')


def tsne(raw_counts, labels, n_components=30):

    norm_counts = normalize_counts(raw_counts)
    reduced_counts = PCA(n_components=n_components,
                         svd_solver='randomized').fit_transform(norm_counts)
    communities, _, _ = phenograph.cluster(reduced_counts)
    tsne_counts = bhtsne.tsne(reduced_counts, rand_seed=1)

    fig, axes = plt.subplots(1, 1)
    axes.scatter(tsne_counts[:, 0], tsne_counts[:, 1],
                 c=communities, cmap=plt.cm.tab20, s=1)
    axes.scatter(tsne_counts[:, 0][labels], tsne_counts[:, 1]
                 [labels], s=3, edgecolor='k', facecolor='k')
    axes.set_title('Cells with Detected\n Doublets in Black')
    plt.xticks([])
    plt.yticks([])
    axes.set_xlabel('{} out of {} cells called doublets.\n {}%  across-type doublet rate.'.format(
        np.sum(labels), raw_counts.shape[0], np.round(100 * np.sum(labels) / raw_counts.shape[0], 2)))
