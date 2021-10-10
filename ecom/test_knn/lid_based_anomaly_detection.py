import numpy as np
import sys
from lid_estimators import lid_mle_amsaleg
from knn_index import KNNIndex
import utils

from test_knn.utils import get_num_jobs


class LID_based_anomaly_detection:
    def __init__(self,
                 neighborhood_constant=0.4, n_neighbors=None,
                 metric='euclidean', metric_kwargs=None,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 seed_rng=123):

        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = get_num_jobs(n_jobs)
        self.seed_rng = seed_rng

        self.num_samples = None
        self.index_knn = None
        self.lid_nominal = None

    def fit(self, data):
        self.num_samples = data.shape[0]
        # Build the KNN graph for the data
        self.index_knn = KNNIndex(
            data,
            neighborhood_constant=self.neighborhood_constant, n_neighbors=self.n_neighbors,
            metric=self.metric, metric_kwargs=self.metric_kwargs,
            shared_nearest_neighbors=False,
            approx_nearest_neighbors=self.approx_nearest_neighbors,
            n_jobs=self.n_jobs,
            seed_rng=self.seed_rng
        )
        # LID estimate at each point based on the nearest neighbor distances
        self.lid_nominal = self.estimate_lid(data, exclude_self=True)

    def score(self, data_test, exclude_self=False):
        # LID estimate at each point based on the nearest neighbor distances
        lid = self.estimate_lid(data_test, exclude_self=exclude_self)

        # Estimate the p-value of each test point based on the empirical distribution of LID on the nominal data
        pvalues = ((1. / self.lid_nominal.shape[0]) *
                   np.sum(self.lid_nominal[:, np.newaxis] > lid[np.newaxis, :], axis=0))

        # Negative log of the p-value is returned as the anomaly score
        return -1.0 * np.log(np.clip(pvalues, sys.float_info.min, None))

    def estimate_lid(self, data, exclude_self=False):
        # Find the distances from each point to its `self.n_neighbors` nearest neighbors.
        nn_indices, nn_distances = self.index_knn.query(data, exclude_self=exclude_self)

        return lid_mle_amsaleg(nn_distances)