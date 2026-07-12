"""Randomization schemes.

Both functions return a boolean treatment vector aligned to the input rows.
Complete randomization (a fixed treated count via permutation) is used rather
than independent Bernoulli draws, so arm sizes are exact.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]


def assign_users(
    users: pd.DataFrame,
    *,
    treat_fraction: float = 0.5,
    seed: int = 0,
) -> BoolArray:
    """User-level complete randomization.

    Parameters
    ----------
    users : pd.DataFrame
        Population; only its length is used.
    treat_fraction : float
        Fraction of users assigned to treatment.
    seed : int
        Seed for the random generator.

    Returns
    -------
    NDArray[np.bool_]
        True where the user is in the treatment arm.
    """
    rng = np.random.default_rng(seed)
    n = len(users)
    n_treat = round(n * treat_fraction)
    return np.asarray(rng.permutation(n) < n_treat)


def assign_clusters(
    users: pd.DataFrame,
    *,
    cluster_col: str,
    treat_fraction: float = 0.5,
    seed: int = 0,
) -> BoolArray:
    """Cluster randomization: whole clusters are assigned to one arm.

    Used when user-level randomization would violate SUTVA (network
    interference); the unit of randomization becomes the cluster.

    Parameters
    ----------
    users : pd.DataFrame
        Population with a cluster label column.
    cluster_col : str
        Name of the cluster label column.
    treat_fraction : float
        Fraction of clusters assigned to treatment.
    seed : int
        Seed for the random generator.

    Returns
    -------
    NDArray[np.bool_]
        True where the user's cluster is in the treatment arm.
    """
    rng = np.random.default_rng(seed)
    clusters = users[cluster_col].to_numpy()
    unique = np.unique(clusters)
    n_treat = round(len(unique) * treat_fraction)
    treated_clusters = unique[rng.permutation(len(unique)) < n_treat]
    return np.asarray(np.isin(clusters, treated_clusters))
