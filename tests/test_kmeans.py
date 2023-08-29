"""
K-means algorithm test class

__author__ = "Youssef Achenchabe"
__license__ = "MIT"
__email__ = "youssef.achenchabe@gmail.com"
"""
import torch
from pymlfs.kmeans import Kmeans



def test_kmeans(example_data: torch.Tensor) -> None:
    """
    Light test to check if kmeans centroids are close to the actual
    means of the gaussians used to generate the input data.
    """

    X, mean1, mean2, _, _ = example_data
    K = 2
    max_iterations = 100

    km = Kmeans(K, max_iterations)
    km.fit(X)

    centroid_1 = km.centroids[0]
    centroid_2 = km.centroids[1]

    if sum(centroid_1) < 1:
        assert sum((centroid_1 - torch.tensor(mean1)) ** 2) < 1
        assert sum((centroid_2 - torch.tensor(mean2)) ** 2) < 1
    else:
        assert sum((centroid_1 - torch.tensor(mean2)) ** 2) < 1
        assert sum((centroid_2 - torch.tensor(mean1)) ** 2) < 1
