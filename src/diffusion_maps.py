import numpy as np
from sacred import Ingredient
from sklearn.metrics import euclidean_distances

diffusion_maps_ingredient = Ingredient("diffusion_maps")


# sacred config convention.
@diffusion_maps_ingredient.config
def main():
    epsilon = 1.2  # noqa: F841
    alpha = 0  # noqa: F841
    t = 1  # noqa: F841


@diffusion_maps_ingredient.capture
def get_diffusion_coordinates(
    _log, epsilon: float, alpha: float, t: int, X: np.ndarray, d: int
) -> np.ndarray:
    """get_diffusion_coordinates

    compute diffusion coordinates.

    Args:
        _log (_type_): logger. sacred.
        epsilon (float): radial kernel bandwidth. sacred.
        alpha (float): density argument. sacred.
        t (int): diffusion steps. sacred.
        X (np.ndarray): dataset. numpy array. each row is a sample.
        d (int): diffusion coordinates dimensions.

    Returns:
        np.ndarray: diffusion coordinates. N x d numpy array.
    """
    _log.info("computing diffusion coordinates")

    # compute kernel matrix
    _log.debug("computing kernel matrix")
    dists = euclidean_distances(X, X)  # distance matrix
    L = np.exp(-((dists**2) / epsilon))  # kernel matrix

    # normalize row wise - row sums - \sum_{j} L_ij for each i
    q = np.sum(L, axis=1)
    # new kernel - k_epsilon^{(alpha)}
    L_alpha = np.diag(q**-alpha) @ L @ np.diag(q**-alpha)

    # anisotropic transition matrix defined by the new kernel - re-normalization
    M = L_alpha / np.sum(L_alpha, axis=1).reshape(-1, 1)

    # apply diffusion for t steps
    _log.debug(f"applying diffusion for {t} steps")
    M = np.linalg.matrix_power(M, t)  # type: ignore

    # compute eigenvalues and eigenvectors
    _log.debug("computing spectral decomposition of $M$")
    eigvals, eigvecs = np.linalg.eigh(M)  # hermitian matrix (as a real symmetric)

    # reverse the order of the eigenvectors - now in descending order - biggest eigenvalue (1) is first (eigh returns ascending order)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # final diffusion coordinates - keep the first NON-TRIVIAL d dimensions
    diffusion_coordinates = eigvecs * (eigvals**t)
    diffusion_coordinates = diffusion_coordinates[:, 1 : d + 1]

    return diffusion_coordinates
