#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 13:46:04 2017

@author: adamgayoso
"""

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def naive_bayes_bernoulli(counts, labels):

    clf = BernoulliNB()
    clf.fit(counts, labels)
    predictions = clf.predict(counts)
    probabilities = clf.predict_proba(counts)

    return predictions, probabilities


def naive_bayes_gauss(counts, labels):

    clf = GaussianNB()
    clf.fit(counts, labels)
    predictions = clf.predict(counts)
    probabilities = clf.predict_proba(counts)

    return predictions, probabilities


def naive_bayes_multinomial(counts, labels):

    clf = MultinomialNB()
    clf.fit(counts, labels)
    predictions = clf.predict(counts)
    probabilities = clf.predict_proba(counts)

    return predictions, probabilities



def gaussian_mixture(counts, weights=[0.93,0.07]):


    clf = GaussianMixture(n_components=2, weights_init = weights)
    clf.fit(counts)
    predictions = clf.predict(counts)
    probabilities = clf.predict_proba(counts)

    return predictions, probabilities


def knn(counts, labels):

    clf = NearestNeighbors(n_neighbors=10)
    clf.fit(counts)
    clf.kneighbors(counts, 10)

    return clf.kneighbors(counts, 10)[0]