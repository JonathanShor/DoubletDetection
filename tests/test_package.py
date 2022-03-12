import numpy as np
import pytest
import doubletdetection


def test_classifier():

    counts = np.random.poisson(size=(500, 100))

    # no phenograph
    clf = doubletdetection.BoostClassifier(
        n_iters=2, clustering_algorithm="louvain", standard_scaling=True
    )
    clf.fit(counts).predict(p_thresh=1e-16, voter_thresh=0.5)
    clf.doublet_score()

    # with phenograph
    clf = doubletdetection.BoostClassifier(
        n_iters=2, clustering_algorithm="phenograph", standard_scaling=True
    )
    clf.fit(counts).predict(p_thresh=1e-16, voter_thresh=0.5)
    clf.doublet_score()

    # with leiden
    clf = doubletdetection.BoostClassifier(
        n_iters=2, clustering_algorithm="leiden", standard_scaling=True, random_state=123
    )
    clf.fit(counts).predict(p_thresh=1e-16, voter_thresh=0.5)
    scores1 = clf.doublet_score()

    # test random state
    # with leiden
    clf = doubletdetection.BoostClassifier(
        n_iters=2, clustering_algorithm="leiden", standard_scaling=True, random_state=123
    )
    clf.fit(counts).predict(p_thresh=1e-16, voter_thresh=0.5)
    scores2 = clf.doublet_score()
    np.testing.assert_equal(scores1, scores2)

    # plotting
    doubletdetection.plot.convergence(clf, show=False, p_thresh=1e-16, voter_thresh=0.5)
    doubletdetection.plot.threshold(clf, show=False, p_step=6)

    # test that you can't use random clustering algorithm
    with pytest.raises(ValueError):
        clf = doubletdetection.BoostClassifier(
            n_iters=2, clustering_algorithm="my_clusters", standard_scaling=True
        )
