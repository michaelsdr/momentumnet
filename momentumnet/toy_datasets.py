# Authors: Michael Sander, Pierre Ablin
# License: MIT

"""Dataset utilities, for simulated examples
"""
import torch
import numpy as np


###########################################
# Fix random seed for reproducible figures
###########################################

torch.manual_seed(1)


def make_circle(radius=1, N=100, label=0, noise=0.2):
    """Create points clouds on a circle


    Parameters
    ----------
    radius : float (default: 1000)
        Radius of the circle
    N : int (default: 100)
        Number of points in the circle
    label : int (default: 0)
        Label of the points
    noise : float (default: .2)
        The noise

    Return
    ------
    x : tensor, shape (2, N)
        points
    y : tensor, shape (1, N)
        labels of the points

    """

    y = label * torch.ones(N)
    theta = 2 * np.pi * torch.rand(1, N)
    r = 3 * (radius + noise * torch.rand(1, N))
    x = torch.vstack((np.cos(theta) * r, np.sin(theta) * r))
    return x, y


def make_data(N=100, n_r=2):
    """Create the data to separate


    Parameters
    ----------

    N : int (default: 100)
        Number of points in each ring

    n_r : int (default: .2)
        Number of rings

    Return
    ------
    x : tensor, shape (N, 2)
        points
    y : tensor, shape (N, 1)
        labels of the points

    """

    radius = np.arange(1, n_r + 1) * 0.5
    y = torch.ones(n_r)
    y[::2] = 0
    data = [make_circle(r, N, label) for r, label in zip(radius, y)]
    x = np.concatenate([d[0] for d in data], axis=1).T
    y = np.concatenate([d[1] for d in data], axis=None)
    x = torch.tensor(x).float()
    y = torch.tensor(y).long()
    return x, y


def random_points_1D(N=100):
    """Sample random points in 1d


    Parameters
    ----------

    N : int (default: 100)
        Number of points

    Return
    ------
    x : tensor, shape (1, N)
        points

    """

    x = 2 * (torch.rand(1, N) - 0.5)
    return x


def make_data_1D(N):
    x = random_points_1D(N)
    return x.view((-1, 1))
