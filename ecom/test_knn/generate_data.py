import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import logger
from sklearn.neighbors import NearestNeighbors
import os, sys
from scipy import stats

import numpy as np
from numba import njit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@njit()
def sample_diagonal_multivariate_normal(means, variances, ns=1):
    dim = means.shape[0]
    stdev = np.sqrt(variances)
    samples = np.zeros((ns, dim))
    for i in range(dim):
        samples[:, i] = np.random.normal(loc=means[i], scale=stdev[i], size=ns)
    return samples

class MFA_model:
    def __init__(self, n_components, dim, dim_latent=None, dim_latent_range=None, seed_rng=123):
        self.n_components = n_components
        if n_components < 1:
            raise ValueError("Invalid value '{}' for 'n_components'".format(n_components))

        self.dim = dim
        if dim < 2:
            raise ValueError("Invalid value '{}' for 'dim'".format(dim))

        self.seed_rng = seed_rng
        np.random.seed(seed_rng)

        if dim_latent:
            if isinstance(dim_latent, int):
                self.dim_latent = [dim_latent] * self.n_components
            else:
                self.dim_latent = dim_latent
                if len(dim_latent) != self.n_components:
                    raise ValueError("Invalid value '{}' for 'dim_latent'".format(dim_latent))

        else:
            a, b = dim_latent_range
            if b >= self.dim:
                logger.warning("Range of the latent dimension should be smaller than {:d}".format(self.dim))
                b = self.dim - 1
                a = min(a, b)

            # Number of latent features is sampled uniformly from the range
            self.dim_latent = np.random.randint(a, high=(b + 1), size=self.n_components)

        self.parameters = self.generate_params()

    def generate_params(self):
        zs = np.zeros(self.dim)
        self.parameters = dict()

        # Mixture weights are sampled from a symmetric Dirichlet distribution
        alpha = 2 * np.ones(self.n_components)
        self.parameters['weights'] = np.random.dirichlet(alpha)

        # Diagonal sensor noise covariance matrix that is shared across the components. The inverse of each diagonal
        # element of this matrix is independently sampled from a Gamma density
        # First sample the shape parameter (of the Gamma density) uniformly at random from a predefined interval
        k = np.random.uniform(low=1., high=5., size=self.dim)
        self.parameters['covariance_noise'] = np.diag(1. / np.random.gamma(k))

        # Mean vector for each component is sampled from a multivariate Gaussian distribution.
        # The factor loading matrices for each component are sampled as follows. Each column of a factor loading
        # matrix is sampled from a multivariate Gaussian with 0 mean vector and diagonal covariance matrix. The
        # inverse of each diagonal element (precision) of the covariance matrix is sampled from a Gamma density.
        means = []
        factor_loading_mats = []
        covariance_mats = []
        for i in range(self.n_components):
            # Component mean
            m = np.random.normal(loc=0., scale=5., size=self.dim)
            means.append(
                sample_diagonal_multivariate_normal(m, np.ones(self.dim), ns=1).reshape(self.dim)
            )
            # Component factor loading matrix
            mat = np.zeros((self.dim, self.dim_latent[i]))
            for j in range(self.dim_latent[i]):
                # For each column of the factor loading matrix
                k = np.random.uniform(low=1., high=5., size=self.dim)
                mat[:, j] = sample_diagonal_multivariate_normal(zs, 1. / np.random.gamma(k), ns=1).reshape(self.dim)

            factor_loading_mats.append(mat)
            # Covariance matrix of the observed data conditioned on this component
            covariance_mats.append(
                np.dot(mat, np.transpose(mat)) + self.parameters['covariance_noise']
            )

        self.parameters['means'] = means
        self.parameters['factor_loading_mats'] = factor_loading_mats
        self.parameters['covariance_mats'] = covariance_mats

        return self.parameters

    def generate_data(self, N):
        """
        Generate `N` data samples from the model.
        :param N: (int) number of data samples to generate.
        :return: (data, labels) where
            - data: numpy array of shape `(N, self.dim)`.
            - labels: numpy array of component index labels of shape `(N, )`.
        """
        zs = np.zeros(self.dim)
        # Number of samples from each component
        counts = np.random.multinomial(N, self.parameters['weights'])
        data = []
        labels = []
        for i in range(self.n_components):
            if counts[i] < 1:
                continue

            # Latent random vector is generated from a zero mean, identity covariance multivariate Gaussian
            z = sample_diagonal_multivariate_normal(
                np.zeros(self.dim_latent[i]), np.ones(self.dim_latent[i]), ns=counts[i]
            )
            # Noise random vector is generated from a zero mean multivariate Gaussian with covariance
            # matrix `self.parameters['covariance_noise']`
            u = sample_diagonal_multivariate_normal(
                zs, np.diag(self.parameters['covariance_noise']), ns=counts[i]
            )

            x = (np.dot(z, np.transpose(self.parameters['factor_loading_mats'][i])) + u +
                 self.parameters['means'][i][np.newaxis, :])
            data.append(x)

            """
            # The observed feature vector can also be directly generated according to the multivariate Gaussian 
            # distribution in the observed feature space. But this can be slow when the number of observed 
            # dimensions is high
            data.append(np.random.multivariate_normal(self.parameters['means'][i],
                                                      self.parameters['covariance_mats'][i], counts[i]))
            """
            labels.append(i * np.ones(counts[i], dtype=np.int))

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels)
        # Randomize the order of the samples
        ind = np.random.permutation(N)

        return data[ind, :], labels[ind]