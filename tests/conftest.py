import pytest
import numpy as np
import torch


@pytest.fixture
def example_data() -> torch.Tensor:
    """
    Generates two clusters from two independent gaussian distributions
    """

    mean1 = [0, 0]
    mean2 = [5, 5]
    cov1 = [[1, 0.7], [0.7, 1]]
    cov2 = [[0.4, 0.2], [0.2, 0.4]]

    # Generate data from the mean and covariance
    data1 = np.random.multivariate_normal(mean1, cov1, size=200)
    data2 = np.random.multivariate_normal(mean2, cov2, size=200)

    return torch.tensor(np.concatenate([data1, data2])), mean1, mean2, cov1, cov2
