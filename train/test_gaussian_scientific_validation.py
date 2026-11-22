"""Numerical and compatibility checks for the Gaussian scientific audit."""

import unittest

import torch

from .experiment import (
    GaussianTransportNet,
    compute_mean_kernel,
    compute_transport_kernel,
    select_batch_min_total_var,
)
from .gaussian_projection import compute_projected_transport_kernel, project_factor_directions
from .test_gaussian_projection import gaussian_w2_squared


class GaussianScientificValidationChecks(unittest.TestCase):
    def setUp(self):
        self.conditions = torch.tensor(
            [[0.05, 0.21], [0.44, 0.82], [0.92, 0.37]],
            dtype=torch.float64,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(283)
            self.model = GaussianTransportNet(hidden_dim=5).double().eval()
            self.model.reset_weights()
            self.heads = {
                name: value.detach()
                for name, value in self.model.named_parameters()
                if ".head." in name
            }
            self.head_direction = {
                name: torch.randn_like(value) for name, value in self.heads.items()
            }

    def model_path(self, step):
        parameters = {
            name: parameter + step * self.head_direction[name]
            for name, parameter in self.heads.items()
        }
        return torch.func.functional_call(self.model, parameters, (self.conditions,))

    def test_metric_finite_difference_converges_on_nonlinear_head_path(self):
        at_zero = torch.zeros((), dtype=torch.float64)
        (mean, factor), (dmean, dfactor) = torch.func.jvp(
            self.model_path, (at_zero,), (torch.ones_like(at_zero),)
        )
        steps = (1e-2, 1e-3, 1e-4, 1e-5)
        finite_differences = []
        for step in steps:
            # Symmetric in the parameter step, with full nonlinear head outputs.
            forward = gaussian_w2_squared(mean, factor, *self.model_path(step))
            backward = gaussian_w2_squared(mean, factor, *self.model_path(-step))
            finite_differences.append((forward + backward) / (2 * step**2))

        for canonical in (False, True):
            with self.subTest(canonical=canonical):
                projected = project_factor_directions(factor, dfactor[:, None], canonical=canonical)
                metric = dmean.square().sum(dim=-1) + projected.square().sum(dim=(1, 2, 3))
                errors = [(fd - metric).abs().max().item() for fd in finite_differences]
                self.assertLess(errors[1], 0.015 * errors[0])
                self.assertLess(errors[2], 0.015 * errors[1])
                torch.testing.assert_close(finite_differences[-1], metric, rtol=1e-8, atol=1e-9)

    def test_upstream_mean_baseline_is_weights_only(self):
        mean_head = self.model.mean_net.head
        weight, bias = mean_head.weight.detach(), mean_head.bias.detach()

        def evaluate_mean_head(weight, bias):
            return torch.func.functional_call(
                self.model, {"mean_net.head.weight": weight, "mean_net.head.bias": bias},
                (self.conditions,),
            )[0]

        jac_weight, jac_bias = torch.autograd.functional.jacobian(
            evaluate_mean_head, (weight, bias)
        )
        weight_features = jac_weight.flatten(1)
        bias_features = jac_bias.flatten(1)
        full_mean_gram = weight_features @ weight_features.T + bias_features @ bias_features.T
        baseline = compute_mean_kernel(self.model, self.conditions)
        torch.testing.assert_close(baseline, weight_features @ weight_features.T)
        torch.testing.assert_close(full_mean_gram - baseline, torch.full_like(baseline, 2.0))

    def test_softplus_threshold_matches_true_raw_and_projected_head_jacobians(self):
        with torch.no_grad():
            self.model.cov_net.head.weight.zero_()
            self.model.cov_net.head.bias.copy_(
                self.conditions.new_tensor([20.1, 0.75, 20.0])
            )

        parameters = {
            name: value.detach() for name, value in self.model.named_parameters()
            if ".head." in name
        }

        def model_with_heads(*values):
            heads = dict(zip(parameters, values))
            return torch.func.functional_call(self.model, heads, (self.conditions,))

        mean_blocks, factor_blocks = torch.autograd.functional.jacobian(
            model_with_heads, tuple(parameters.values())
        )
        mean_jacobian = torch.cat([block.flatten(2) for block in mean_blocks], dim=-1).flatten(1)
        mean_gram = mean_jacobian @ mean_jacobian.T
        factor_jacobian = torch.cat(
            [block.flatten(3) for block in factor_blocks], dim=-1
        ).movedim(-1, 1)
        raw_features = factor_jacobian.flatten(1)
        expected_raw = mean_gram + raw_features @ raw_features.T
        raw = compute_transport_kernel(self.model, self.conditions)
        torch.testing.assert_close(raw, expected_raw, atol=1e-12, rtol=1e-12)

        with torch.no_grad():
            _, factors = self.model(self.conditions)
        for canonical in (False, True):
            with self.subTest(canonical=canonical):
                projected = project_factor_directions(
                    factors, factor_jacobian, canonical=canonical
                ).flatten(1)
                expected = mean_gram + projected @ projected.T
                actual = compute_projected_transport_kernel(self.model, self.conditions, canonical=canonical)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_diagonal_energy_gap_matches_two_dimensional_vertical_component(self):
        with torch.no_grad():
            _, factor = self.model(self.conditions)
            embeddings = self.model.cov_net.get_feature_embeddings(self.conditions)
        raw22 = self.model.cov_net.get_pre_activations(self.conditions)[:, 2].requires_grad_()
        slope, = torch.autograd.grad(torch.nn.functional.softplus(raw22).sum(), raw22)

        # In d=2 the law-preserving factor directions span L @ [[0, -1], [1, 0]].
        # Its inner products with (dL11, dL21, dL22) are 0, L22, and -L21*dL22.
        numerator = factor[:, 1, 1].square() + (factor[:, 1, 0] * slope).square()
        expected_gap = (embeddings.square().sum(dim=-1) + 1) * numerator
        expected_gap = expected_gap / factor.square().sum(dim=(-2, -1))
        raw_diagonal = compute_transport_kernel(self.model, self.conditions).diag()
        for canonical in (False, True):
            with self.subTest(canonical=canonical):
                projected_diagonal = compute_projected_transport_kernel(
                    self.model, self.conditions, canonical=canonical
                ).diag()
                torch.testing.assert_close(
                    raw_diagonal - projected_diagonal, expected_gap, atol=1e-12, rtol=1e-12
                )

    def test_selector_matches_direct_gp_reconditioning(self):
        features = torch.tensor(
            [
                [-2, 4, -2, -5], [3, -1, -5, 5], [0, 3, -4, -2], [-4, -1, -2, 5],
                [-1, 3, 5, -4], [3, 5, -3, -4], [-3, 4, -5, -5], [3, -4, 3, 4],
            ],
            dtype=torch.float64,
        )
        kernel = features @ features.T + 0.2 * torch.eye(len(features), dtype=features.dtype)
        pool = list(range(len(kernel)))
        labels = [0, 4]
        regularization = 1e-3

        def direct_selection(hypothetical_lam):
            observed = labels.copy()
            observation_noises = [regularization] * len(labels)
            for _ in range(3):
                candidate_traces = []
                for candidate in sorted(set(pool) - set(observed)):
                    combined = observed + [candidate]
                    noise = kernel.new_tensor(observation_noises + [hypothetical_lam])
                    cross = kernel[:, combined]
                    block = kernel[combined][:, combined] + torch.diag(noise)
                    posterior = kernel - cross @ torch.linalg.solve(block, cross.T)
                    candidate_traces.append((posterior.trace().item(), candidate))
                ranked = sorted(candidate_traces)
                self.assertGreater(ranked[1][0] - ranked[0][0], 0.02)
                observed.append(ranked[0][1])
                observation_noises.append(hypothetical_lam)
            return observed[len(labels):]

        legacy = direct_selection(1.0)
        coherent = direct_selection(regularization)
        self.assertNotEqual(legacy, coherent)
        self.assertEqual(
            select_batch_min_total_var(kernel, pool, labels, 3, regularization), legacy
        )
        self.assertEqual(
            select_batch_min_total_var(
                kernel, pool, labels, 3, regularization, hypothetical_lam=regularization
            ),
            coherent,
        )

    def test_gp_kernel_scale_is_confounded_with_both_regularization_terms(self):
        scales = (0.25, 4.0)
        labels = [0]
        features = torch.tensor(
            [[1.0, 0.1], [2.0, 0.3], [0.4, 0.2], [0.7, 2.1]], dtype=torch.float64,
        )
        kernel = features @ features.T
        pool = list(range(len(kernel)))
        lam, hypothetical_lam = 1e-3, 1.0
        baseline = select_batch_min_total_var(
            kernel, pool, labels, 2, lam, hypothetical_lam=hypothetical_lam
        )
        for scale in scales:
            with self.subTest(scale=scale):
                self.assertEqual(
                    select_batch_min_total_var(
                        scale * kernel, pool, labels, 2, scale * lam,
                        hypothetical_lam=scale * hypothetical_lam,
                    ),
                    baseline,
                )


if __name__ == "__main__":
    unittest.main()
