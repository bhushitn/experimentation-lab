"""Social graph generation for interference simulations.

A stochastic block model: users belong to clusters, edges are dense
within a cluster and sparse between. This is the structure that makes
cluster randomization work, so the same graph demonstrates both the
naive estimator's bias and the cluster design's recovery.
"""

import numpy as np
import pandas as pd
from scipy import sparse


def sbm_graph(
    n_users: int,
    *,
    n_clusters: int = 100,
    p_within: float = 0.10,
    p_between: float = 5e-5,
    seed: int = 0,
) -> tuple[sparse.csr_matrix, pd.Series]:
    """Undirected stochastic block model.

    Within-cluster edges are exact Bernoulli draws per pair. The sparse
    between-cluster edges are sampled by count (binomial on the number
    of cross pairs, then uniform pair draws, deduplicated), which is the
    standard fast approximation for sparse SBM generation.

    Parameters
    ----------
    n_users : int
        Number of nodes; cluster sizes are as equal as possible.
    n_clusters : int
        Number of clusters.
    p_within, p_between : float
        Edge probability inside and across clusters.
    seed : int
        Seed for the random generator.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix
        Symmetric 0/1 adjacency, zero diagonal.
    clusters : pd.Series
        Cluster label per node.
    """
    rng = np.random.default_rng(seed)
    labels = np.arange(n_users) % n_clusters
    rows, cols = [], []

    for c in range(n_clusters):
        members = np.flatnonzero(labels == c)
        size = members.size
        upper_i, upper_j = np.triu_indices(size, k=1)
        keep = rng.random(upper_i.size) < p_within
        rows.append(members[upper_i[keep]])
        cols.append(members[upper_j[keep]])

    n_cross_pairs = (n_users * (n_users - 1)) // 2 - sum(
        (np.sum(labels == c) * (np.sum(labels == c) - 1)) // 2 for c in range(n_clusters)
    )
    n_cross_edges = rng.binomial(n_cross_pairs, p_between)
    i = rng.integers(0, n_users, size=3 * n_cross_edges)
    j = rng.integers(0, n_users, size=3 * n_cross_edges)
    cross = (labels[i] != labels[j]) & (i < j)
    i, j = i[cross][:n_cross_edges], j[cross][:n_cross_edges]
    rows.append(i)
    cols.append(j)

    r = np.concatenate(rows)
    c_arr = np.concatenate(cols)
    data = np.ones(r.size)
    upper = sparse.coo_matrix((data, (r, c_arr)), shape=(n_users, n_users))
    adjacency = (upper + upper.T).tocsr()
    adjacency.data = np.minimum(adjacency.data, 1.0)
    return adjacency, pd.Series(labels, name="cluster")
