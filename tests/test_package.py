import numpy as np

import doubletdetection


def test_classifier():

    counts = np.random.poisson(size=(500, 100))

    # no phenograph
    clf = doubletdetection.BoostClassifier(n_iters=2, use_phenograph=False, standard_scaling=True)
    clf.fit(counts).predict(p_thresh=1e-16, voter_thresh=0.5)
    clf.doublet_score()

    # with phenograph
    clf = doubletdetection.BoostClassifier(n_iters=2, use_phenograph=True, standard_scaling=True)
    clf.fit(counts).predict(p_thresh=1e-16, voter_thresh=0.5)
    clf.doublet_score()

    doubletdetection.plot.convergence(clf, show=False, p_thresh=1e-16, voter_thresh=0.5)
    doubletdetection.plot.threshold(clf, show=False, p_step=6)
