"""Wasserstein tangent projection for full-rank Gaussian affine maps."""

import torch
import torch.nn.functional as F


def softplus_slope(raw):
    """Differentiate PyTorch's default softplus, including its linear branch."""
    return torch.where(raw > 20, torch.ones_like(raw), raw.sigmoid())


def project_factor_directions(factor, directions, *, canonical=False):
    """Return A M (or A Σ¹ᐟ²) for each factor direction dM.

    `factor` has shape (conditions, d, d) and `directions` has shape
    (conditions, directions, d, d). All factors must be invertible.
    The symmetric A solves A Σ + Σ A = dM Mᵀ + M dMᵀ, with Σ = M Mᵀ.
    """
    device = factor.device
    dtype = factor.dtype
    # MPS has no eigh. Use double precision for the small spectral solve.
    solve_device = "cpu" if device.type == "mps" else device
    factor = factor.to(device=solve_device, dtype=torch.float64)
    directions = directions.to(device=solve_device, dtype=torch.float64)
    covariance = factor @ factor.mT
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if torch.any(eigenvalues <= 0):
        raise ValueError("Gaussian tangent projection requires positive definite covariances")

    derivative = directions @ factor[:, None].mT
    derivative = derivative + derivative.mT
    basis = eigenvectors[:, None]
    eigenvalue_sums = eigenvalues[:, :, None] + eigenvalues[:, None, :]
    tangent = (basis.mT @ derivative @ basis) / eigenvalue_sums[:, None]
    tangent = basis @ tangent @ basis.mT

    if canonical:
        factor = (eigenvectors * eigenvalues.sqrt()[:, None, :]) @ eigenvectors.mT
    return (tangent @ factor[:, None]).to(device=device, dtype=dtype)


@torch.no_grad()
def compute_projected_transport_kernel(model, x, *, canonical=False):
    """Last-layer tNTK with each Cholesky direction projected before the Gram."""
    mean_features = model.mean_net.get_feature_embeddings(x)
    cov_features = model.cov_net.get_feature_embeddings(x)
    raw = model.cov_net.head(cov_features)
    # CovarianceNet uses one output per entry, ordered (L11, L21, L22).
    factor = torch.zeros((len(x), 2, 2), device=x.device, dtype=x.dtype)
    factor[:, 0, 0] = F.softplus(raw[:, 0])
    factor[:, 1, 0] = raw[:, 1]
    factor[:, 1, 1] = F.softplus(raw[:, 2])

    slopes = torch.stack(
        (softplus_slope(raw[:, 0]), torch.ones_like(raw[:, 1]), softplus_slope(raw[:, 2])), dim=1,
    )
    directions = factor.new_zeros((len(x), 3, 2, 2))
    directions[:, 0, 0, 0] = slopes[:, 0]
    directions[:, 1, 1, 0] = slopes[:, 1]
    directions[:, 2, 1, 1] = slopes[:, 2]
    projected = project_factor_directions(factor, directions, canonical=canonical).flatten(1)

    # Each direction's parameters share the same hidden features and a bias.
    mean_gram = 2.0 * (mean_features @ mean_features.T) + 2.0
    cov_gram = (cov_features @ cov_features.T + 1.0) * (projected @ projected.T)
    return mean_gram + cov_gram
