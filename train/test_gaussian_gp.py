"""Tests for the float64 Gaussian acquisition path."""

import unittest

import numpy as np
import torch

from .gaussian_gp import (
    acquisition_model,
    posterior_covariance,
    select_gaussian_gp_batch,
    variance_reduction_scores,
)


def gram(features):
    features = torch.tensor(features, dtype=torch.float64)
    return features @ features.T


def direct_posterior(kernel, labels, noises):
    """Independent NumPy Schur complement with explicit per-label variances."""
    labeled = np.ix_(labels, labels)
    cross = kernel[:, labels]
    return kernel - cross @ np.linalg.solve(kernel[labeled] + np.diag(noises), cross.T)


class StableGaussianGPChecks(unittest.TestCase):
    def test_two_stage_mixed_noise_matches_direct_reconditioning(self):
        kernel = gram([
            [1.0, 0.3, -0.8], [1.8, -0.6, 0.1], [0.7, 1.8, 0.9],
            [-1.5, 0.2, 0.5], [1.1, -0.1, 1.7], [0.1, 2.8, -0.5],
        ])
        kernel += 0.12 * torch.eye(len(kernel), dtype=kernel.dtype)
        kernel_np = kernel.numpy()
        observed = [4, 0]
        labels = observed.copy()
        observed_noise, hypothetical_noise = 0.02, 1.6
        noises = [observed_noise] * len(observed)
        pool = list(range(len(kernel)))

        for stage in range(2):
            with self.subTest(stage=stage):
                posterior = posterior_covariance(
                    kernel, labels, n_observed=len(observed),
                    observed_noise=observed_noise, hypothetical_noise=hypothetical_noise,
                )
                expected = direct_posterior(kernel_np, labels, noises)
                np.testing.assert_allclose(posterior.numpy(), expected, atol=3e-14, rtol=3e-14)
                remaining = sorted(set(pool) - set(labels))
                scores = variance_reduction_scores(posterior, hypothetical_noise)
                candidate_traces = [
                    np.trace(direct_posterior(
                        kernel_np, [*labels, index], [*noises, hypothetical_noise],
                    ))
                    for index in remaining
                ]
                delta = np.trace(expected) - np.asarray(candidate_traces)
                np.testing.assert_allclose(scores[remaining].numpy(), delta, atol=3e-14, rtol=3e-14)
                self.assertGreater(np.sort(delta)[-1] - np.sort(delta)[-2], 1e-3)
                labels.append(remaining[int(np.argmax(delta))])
                noises.append(hypothetical_noise)

        self.assertEqual(
            select_gaussian_gp_batch(
                kernel, pool, observed, 2, observed_noise,
                hypothetical_noise=hypothetical_noise,
            ),
            labels[len(observed):],
        )

    def test_exact_ties_and_budget_larger_than_eligible_pool(self):
        kernel = torch.eye(5, dtype=torch.float64)
        # Every eligible point is tied. The pool intentionally skips index 2.
        self.assertEqual(select_gaussian_gp_batch(kernel, [4, 3, 1, 1, 0], [0], 7, 0.01),
                         [1, 3, 4])
        self.assertEqual(select_gaussian_gp_batch(kernel, [4, 3], [4, 3], 7, 0.01), [])
        self.assertEqual(select_gaussian_gp_batch(kernel, [2, 0], [0], 0, 0.01), [])

    def test_simultaneous_kernel_and_noise_scaling(self):
        kernel = gram([[1, 0, 0], [10, 0, 0], [1.5, 0.8, 0], [0, 1, 0], [0, 0, 1.1]])
        observed_noise, hypothetical_noise = 0.001, 1.0
        pool = list(range(len(kernel)))
        labels = [0, 3]

        def posterior(matrix, scale):
            return posterior_covariance(
                matrix, labels, n_observed=1,
                observed_noise=scale * observed_noise, hypothetical_noise=scale * hypothetical_noise,
            )

        def query(matrix, scale):
            return select_gaussian_gp_batch(
                matrix, pool, labels[:1], 2, scale * observed_noise,
                hypothetical_noise=scale * hypothetical_noise,
            )

        baseline_posterior = posterior(kernel, 1.0)
        baseline_scores = variance_reduction_scores(baseline_posterior, hypothetical_noise)
        self.assertEqual(query(kernel, 1.0), [3, 4])
        for scale in (0.001, 1.0 / 21.22, 1000.0):
            with self.subTest(scale=scale):
                scaled_posterior = posterior(scale * kernel, scale)
                scaled_scores = variance_reduction_scores(scaled_posterior, scale * hypothetical_noise)
                torch.testing.assert_close(
                    scaled_posterior / scale, baseline_posterior, atol=1e-12, rtol=1e-12,
                )
                torch.testing.assert_close(scaled_scores / scale, baseline_scores, atol=1e-12, rtol=1e-12)
                self.assertEqual(query(scale * kernel, scale), [3, 4])

        scale = float(kernel.diagonal().mean())
        fixed_noise_normalized = query(kernel / scale, 1.0)
        self.assertEqual(fixed_noise_normalized, [1, 3])
        self.assertEqual(fixed_noise_normalized, query(kernel, scale))

    def test_float64_keeps_jitter_on_an_ill_conditioned_psd_gram(self):
        kernel = gram([[1024, 0, 0], [1024, 0, 0], [1024, 2, 1], [1024, 0, 3], [1024, 1, 4]])
        observed_noise = 0.001
        raw_block = kernel[:2, :2].float()
        jitter = observed_noise * torch.eye(len(raw_block), dtype=raw_block.dtype)
        # The old float32 path rounds the regularized square block back to rank one.
        self.assertTrue(torch.equal(raw_block + jitter, raw_block))
        with self.assertRaisesRegex(TypeError, "float64"):
            posterior_covariance(
                kernel.float(), [0, 1], n_observed=2,
                observed_noise=observed_noise, hypothetical_noise=1.0,
            )

        posterior = posterior_covariance(
            kernel, [0, 1], n_observed=2,
            observed_noise=observed_noise, hypothetical_noise=1.0,
        )
        # Conditioning two identical observations leaves this analytic residual
        # shared variance. The independent shape Gram is unchanged.
        constant = 1024.0**2
        shared = constant * observed_noise / (2.0 * constant + observed_noise)
        shape_kernel = kernel - constant
        torch.testing.assert_close(posterior, shape_kernel + shared, atol=2e-9, rtol=1e-10)

    def test_acquisition_model_converts_a_copy_and_keeps_training_precision(self):
        model = torch.nn.Linear(2, 3, dtype=torch.float32)
        inputs = torch.tensor([[0.3, -0.7]], dtype=torch.float32)
        converted, converted_inputs = acquisition_model(model, inputs)
        self.assertIsNot(converted, model)
        self.assertEqual(model.weight.dtype, torch.float32)
        self.assertEqual(inputs.dtype, torch.float32)
        torch.testing.assert_close(converted.weight, model.weight.double())
        torch.testing.assert_close(converted.bias, model.bias.double())
        torch.testing.assert_close(converted_inputs, inputs.double())


if __name__ == "__main__":
    unittest.main()
