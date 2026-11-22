"""Float64 GP acquisition for the small affine Gaussian benchmark."""

import copy

import torch


def acquisition_model(model, inputs):
    """Keep training unchanged and evaluate acquisition on a CPU float64 copy."""
    return copy.deepcopy(model).to(device="cpu", dtype=torch.float64), inputs.to(
        device="cpu", dtype=torch.float64,
    )


def posterior_covariance(kernel, indices, *, n_observed, observed_noise, hypothetical_noise):
    """Recondition the original Gram with separate noise levels for each label type."""
    if kernel.dtype != torch.float64:
        raise TypeError("Recompute the acquisition Gram in float64 before conditioning")
    if not indices:
        return kernel
    idx = torch.as_tensor(indices, dtype=torch.long, device=kernel.device)
    blocks = kernel.index_select(0, idx)
    observed = blocks.index_select(1, idx)
    noise = kernel.new_full((len(idx),), hypothetical_noise)
    noise[:n_observed] = observed_noise
    factor = torch.linalg.cholesky(observed + noise.diag())
    projected = torch.linalg.solve_triangular(factor, blocks, upper=False)
    posterior = kernel - projected.T @ projected
    return (posterior + posterior.T) * 0.5


def variance_reduction_scores(posterior, hypothetical_noise):
    """Trace reduction over the entire pool for each possible next query."""
    return posterior.square().sum(dim=0) / (posterior.diagonal() + hypothetical_noise)


def select_gaussian_gp_batch(kernel, pool_indices, train_indices, batch_size, observed_noise,
                             *, hypothetical_noise=1.0):
    """Use a fresh Cholesky solve for every greedy step."""
    selected = list(train_indices)
    n_observed = len(selected)
    remaining = sorted(set(pool_indices) - set(selected))
    for _ in range(min(batch_size, len(remaining))):
        posterior = posterior_covariance(
            kernel, selected, n_observed=n_observed, observed_noise=observed_noise,
            hypothetical_noise=hypothetical_noise,
        )
        scores = variance_reduction_scores(posterior, hypothetical_noise)
        best = torch.argmax(scores[remaining]).item()
        selected.append(remaining.pop(best))
    return selected[n_observed:]
