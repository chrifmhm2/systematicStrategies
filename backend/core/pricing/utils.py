import numpy as np


def cholesky_decompose(correlation: np.ndarray) -> np.ndarray:
    """
    Compute the lower-triangular Cholesky factor L of a correlation matrix.

    Guarantees: L @ L.T == correlation

    Used to introduce correlation between independent standard normal draws:
        Z = eps @ L.T   where eps ~ N(0, I)
        → Z has covariance = correlation

    Parameters
    ----------
    correlation : np.ndarray
        Square correlation matrix, shape (n, n).
        Must be symmetric and positive definite (all eigenvalues > 0).

    Returns
    -------
    np.ndarray
        Lower-triangular matrix L, shape (n, n).

    Raises
    ------
    ValueError
        If the matrix is not square or not positive definite.
    """
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError(
            f"correlation must be a square 2-D matrix, got shape {correlation.shape}."
        )

    try:
        return np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "correlation matrix is not positive definite — "
            "check that all eigenvalues are strictly positive."
        ) from exc


def generate_correlated_normals(
    n_assets: int,
    n_samples: int,
    chol: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Draw correlated standard-normal samples using a Cholesky factor.

    Steps:
        1. Draw independent N(0,1):  eps ~ shape (n_samples, n_assets)
        2. Correlate via Cholesky:   Z = eps @ chol.T

    The resulting Z has covariance matrix = chol @ chol.T = correlation.

    Parameters
    ----------
    n_assets : int
        Number of assets (columns in the output).
    n_samples : int
        Number of samples to draw (rows in the output).
    chol : np.ndarray
        Lower-triangular Cholesky factor, shape (n_assets, n_assets).
        Obtain it from cholesky_decompose().
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Correlated standard-normal matrix, shape (n_samples, n_assets).
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n_samples, n_assets))
    return eps @ chol.T
