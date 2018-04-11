import matplotlib.pyplot as plt
import numpy as np
# Ignore warning for convergence plot
np.warnings.filterwarnings('ignore')

def convergence(clf, show=True, save=False, p_thresh=0.99, voter_thresh=0.9):
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
        cum_p_values = clf.all_p_values_[:i+1]
        cum_vote_average = np.mean(np.ma.masked_invalid(cum_p_values) > p_thresh, axis=0)
        cum_doublets = np.ma.filled(cum_vote_average >= voter_thresh, np.nan)
        doubs_per_run.append(np.sum(cum_doublets))

    plt.figure(figsize=(4,4))
    plt.plot(np.arange(len(doubs_per_run)), doubs_per_run)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Number of cells called doublets")
    plt.title('Convergence of Number of Cells Called Doublets over Iterations')

    if show is True:
        plt.show()
    if save is True:
        plt.savefig('doublet_convergence.pdf', format='pdf')
