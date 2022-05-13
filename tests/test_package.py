import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

import doubletdetection


def test_sklearn_estimator():
    clf = doubletdetection.BoostClassifier(
        n_iters=2, clustering_algorithm="louvain", standard_scaling=True
    )
    check_estimator(clf)


def test_classifier():

    counts = np.random.poisson(size=(500, 100))

    # no phenograph
    clf = doubletdetection.BoostClassifier(
        n_iters=2, clustering_algorithm="louvain", standard_scaling=True
    )
    clf.fit(counts).get_predictions()
    clf.get_doublet_scores()

    # with phenograph
    clf = doubletdetection.BoostClassifier(
        n_iters=2, clustering_algorithm="phenograph", standard_scaling=True
    )
    clf.fit(counts).get_predictions()
    clf.get_doublet_scores()

    # with leiden
    clf = doubletdetection.BoostClassifier(
        n_iters=2, clustering_algorithm="leiden", standard_scaling=True, random_state=123
    )
    clf.fit(counts).get_predictions()
    scores1 = clf.get_doublet_scores()

    # test random state
    # with leiden
    clf = doubletdetection.BoostClassifier(
        n_iters=2, clustering_algorithm="leiden", standard_scaling=True, random_state=123
    )
    clf.fit(counts).get_predictions()
    scores2 = clf.get_doublet_scores()
    np.testing.assert_equal(scores1, scores2)

    # plotting
    doubletdetection.plot.convergence(
        clf,
        show=False,
    )
    doubletdetection.plot.threshold(clf, show=False, p_step=6)

    # test that you can't use random clustering algorithm
    with pytest.raises(ValueError):
        clf = doubletdetection.BoostClassifier(
            n_iters=2, clustering_algorithm="my_clusters", standard_scaling=True
        )
        clf.fit(counts)
