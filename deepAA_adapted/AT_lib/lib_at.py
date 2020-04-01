import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp


def centralize_matrix(M):
    """
    See https://en.wikipedia.org/wiki/Centering_matrix
    :param M:
    :return:
    """
    n, p = M.shape
    Q = np.full((n, n), -1 / n)
    Q = Q + np.eye(n)
    M = np.dot(np.dot(Q, M), Q)
    return M


def greedy_min_distance(z_f, z_true_mu):
    """
    Calculates the mean L2 Distance between found Archetypes and the true Archetypes (in latent space).
    1. Select the 2 vector with smallest pairwise distance
    2. Calculate the euclidean distance
    3. Remove the 2 vectors and jump to 1.
    :param z_f:
    :param z_true_mu:
    :return: mean loss
    """
    loss = []
    dist = sp.spatial.distance.cdist(z_f, z_true_mu)
    for i in range(z_f.shape[0]):
        z_fixed_idx, z_true_idx = np.unravel_index(dist.argmin(), dist.shape)
        loss.append(dist[z_fixed_idx, z_true_idx])
        dist = np.delete(np.delete(dist, z_fixed_idx, 0), z_true_idx, 1)
    return loss


def create_z_fix(dim_latent_space):
    """
    Creates Coordinates of the Simplex spanned by the Archetypes.

    The simplex will have its centroid at 0.
    The sum of the vertices will be zero.
    The distance of each vertex from the origin will be 1.
    The length of each edge will be constant.
    The dot product of the vectors defining any two vertices will be - 1 / M.
    This also means the angle subtended by the vectors from the origin
    to any two distinct vertices will be arccos ( - 1 / M ).

    :param dim_latent_space:
    :return:
    """

    z_fixed_t = np.zeros([dim_latent_space, dim_latent_space + 1])

    for k in range(0, dim_latent_space):
        s = 0.0
        for i in range(0, k):
            s = s + z_fixed_t[i, k] ** 2

        z_fixed_t[k, k] = np.sqrt(1.0 - s)

        for j in range(k + 1, dim_latent_space + 1):
            s = 0.0
            for i in range(0, k):
                s = s + z_fixed_t[i, k] * z_fixed_t[i, j]

            z_fixed_t[k, j] = (-1.0 / float(dim_latent_space) - s) / z_fixed_t[k, k]
            z_fixed = np.transpose(z_fixed_t)
    return z_fixed


def barycentric_coords(n_per_axis=5):
    """
    Creates coordinates for the traversal of 3 Archetypes (i.e. creates the a weights)
    :param n_per_axis:
    :return: [weights, n_perAxis]; weights has shape (?, 3)
    """
    weights = np.zeros([int((n_per_axis * (n_per_axis + 1)) / 2), 3])

    offset = np.sqrt(3 / 4) / (n_per_axis - 1)
    A = np.array([[1.5, 0, 0], [np.sqrt(3) / 2, np.sqrt(3), 0], [1, 1, 1, ]])
    cnt = 0
    innerCnt = 0
    for i in np.linspace(0, 1.5, n_per_axis):
        startX = i
        startY = cnt * offset

        if n_per_axis - cnt != 1:
            stpY = (np.sqrt(3) - 2 * startY) / (n_per_axis - cnt - 1)
        else:
            stpY = 1
        for j in range(1, n_per_axis - cnt + 1):
            P_x = startX
            P_y = startY + (j - 1) * stpY
            b = np.array([P_x, P_y, 1])
            sol = solve(A, b)

            out = np.abs(np.around(sol, 6))
            weights[innerCnt, :] = out
            innerCnt += 1
        cnt += 1

    return [weights, n_per_axis]



