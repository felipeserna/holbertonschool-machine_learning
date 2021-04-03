#!/usr/bin/env python3
"""
Calculates a GMM from a dataset
"""


import sklearn.mixture


def gmm(X, k):
    """
    Returns: pi, m, S, clss, bic
    """
    GMM = sklearn.mixture.GaussianMixture(n_components=k)
    GMM.fit(X)
    pi = GMM.weights_
    m = GMM.means_
    S = GMM.covariances_
    clss = GMM.predict(X)
    bic = GMM.bic(X)

    return pi, m, S, clss, bic
