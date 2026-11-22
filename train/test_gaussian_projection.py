"""Run the Gaussian projection checks with `python -m unittest train.test_gaussian_projection`."""

import unittest

import torch

from .experiment import GaussianTransportNet, compute_transport_kernel, select_batch_min_total_var
from .gaussian_projection import compute_projected_transport_kernel, project_factor_directions


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def gaussian_w2_squared(mean, factor, other_mean, other_factor):
    """Evaluate Gaussian W2² via orthogonal Procrustes without cancellation."""
    left, _, right_transpose = torch.linalg.svd(other_factor.mT @ factor)
    aligned_factor = other_factor @ (left @ right_transpose)
    mean_error = (mean - other_mean).square().sum(dim=-1)
    shape_error = (factor - aligned_factor).square().sum(dim=(-2, -1))
    return mean_error + shape_error


class GaussianProjectionChecks(unittest.TestCase):
    def setUp(self):
        self.factor = tensor([[[1.3, 0.0], [0.6, 0.7]], [[0.3, 0.0], [0.2, 2.4]]])
        self.directions = tensor(
            [
                [[[0.2, 0.5], [-0.7, 0.3]], [[1.0, 0.3], [0.7, -0.4]]],
                [[[0.2, 0.1], [0.0, 0.1]], [[0.9, 0.1], [-0.5, 0.3]]],
            ]
        )

    def test_covariance_derivative_is_preserved_and_projected_field_is_symmetric(self):
        factor = self.factor[:, None]
        projected = project_factor_directions(self.factor, self.directions)
        gradient = torch.linalg.solve(factor.mT, projected.mT).mT
        torch.testing.assert_close(gradient, gradient.mT, atol=1e-12, rtol=1e-12)

        covariance_derivative = self.directions @ factor.mT + factor @ self.directions.mT
        projected_derivative = projected @ factor.mT + factor @ projected.mT
        torch.testing.assert_close(projected_derivative, covariance_derivative, atol=1e-12, rtol=1e-12)

        residual = self.directions - projected
        inner_product = (projected * residual).sum(dim=(-2, -1))
        torch.testing.assert_close(inner_product, torch.zeros_like(inner_product), atol=1e-12, rtol=0)

    def test_metric_derivative_matches_wasserstein_finite_difference(self):
        mean = tensor([[0.1, -0.5], [0.3, 0.4]])[:, None]
        mean_direction = tensor([[[0.4, 0.2], [0.0, -1.0]], [[0.2, 0.8], [0.3, 0.4]]])
        factor = self.factor[:, None]
        step = 1e-4
        forward = gaussian_w2_squared(
            mean, factor, mean + step * mean_direction, factor + step * self.directions
        )
        backward = gaussian_w2_squared(
            mean, factor, mean - step * mean_direction, factor - step * self.directions
        )
        finite_difference = (forward + backward) / (2 * step**2)

        for canonical in (False, True):
            with self.subTest(canonical=canonical):
                projected = project_factor_directions(self.factor, self.directions, canonical=canonical)
                metric_derivative = mean_direction.square().sum(dim=-1)
                metric_derivative += projected.square().sum(dim=(-2, -1))
                torch.testing.assert_close(metric_derivative, finite_difference, atol=1e-8, rtol=1e-7)

    def test_pure_latent_rotation_is_annihilated(self):
        skew = tensor([[[[0.0, -1.0], [1.0, 0.0]]], [[[0.0, 3.0], [-3.0, 0.0]]]])
        directions = self.factor[:, None] @ skew
        self.assertGreater(float(directions.square().sum()), 1.0)
        for canonical in (False, True):
            with self.subTest(canonical=canonical):
                projected = project_factor_directions(self.factor, directions, canonical=canonical)
                torch.testing.assert_close(projected, torch.zeros_like(projected), atol=1e-12, rtol=0)

    def test_canonical_field_is_invariant_to_input_and_parameter_dependent_rotation(self):
        angles = tensor([0.7, -1.3])
        cosine = angles.cos()
        sine = angles.sin()
        rotation = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(-1, 2, 2)
        skew_base = tensor([[0.0, -1.0], [1.0, 0.0]])
        skew = tensor([[2.3, -4.1], [0.05, 1.4]])[..., None, None] * skew_base
        rotated_factor = self.factor @ rotation
        rotated_directions = self.directions @ rotation[:, None] + rotated_factor[:, None] @ skew

        canonical = project_factor_directions(self.factor, self.directions, canonical=True)
        rotated_canonical = project_factor_directions(rotated_factor, rotated_directions, canonical=True)
        torch.testing.assert_close(canonical, rotated_canonical, atol=1e-12, rtol=1e-12)

        pullback = project_factor_directions(self.factor, self.directions)
        rotated_pullback = project_factor_directions(rotated_factor, rotated_directions)
        torch.testing.assert_close(rotated_pullback, pullback @ rotation[:, None], atol=1e-12, rtol=1e-12)
        pullback_flat = pullback.flatten(start_dim=1)
        rotated_flat = rotated_pullback.flatten(start_dim=1)
        original_gram = pullback_flat @ pullback_flat.T
        rotated_gram = rotated_flat @ rotated_flat.T
        self.assertGreater(float((original_gram - rotated_gram).abs().max()), 1e-3)

    def test_one_dimensional_positive_factor_is_a_fixed_point(self):
        factor = tensor([[[0.3]], [[1.4]]])
        directions = tensor([[[[0.5]], [[-0.2]]], [[[0.7]], [[0.2]]]])
        for canonical in (False, True):
            with self.subTest(canonical=canonical):
                projected = project_factor_directions(factor, directions, canonical=canonical)
                torch.testing.assert_close(projected, directions, atol=1e-12, rtol=1e-12)

    def test_cholesky_shear_at_identity_loses_half_its_energy(self):
        factor = torch.eye(2, dtype=torch.float64)[None]
        direction = tensor([[[[0.0, 0.0], [1.0, 0.0]]]])
        projected = project_factor_directions(factor, direction)
        torch.testing.assert_close(projected, (direction + direction.mT) / 2, atol=1e-12, rtol=0)
        self.assertAlmostEqual(float(projected.square().sum()), 0.5)

    def test_closed_form_kernels_match_explicit_head_jacobians(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(149)
            model = GaussianTransportNet(hidden_dim=5).double().eval()
            conditions = torch.rand(9, 2, dtype=torch.float64)

        head_parameters = {name: value for name, value in model.named_parameters() if ".head." in name}

        def head_outputs(*values):
            replacements = dict(zip(head_parameters, values))
            return torch.func.functional_call(model, replacements, (conditions,))

        mean_blocks, factor_blocks = torch.autograd.functional.jacobian(
            head_outputs, tuple(head_parameters.values())
        )
        mean_features = torch.cat([block.flatten(2) for block in mean_blocks], dim=-1).flatten(1)
        mean_gram = mean_features @ mean_features.T
        factor_directions = torch.cat([block.flatten(3) for block in factor_blocks], dim=-1).movedim(-1, 1)
        raw_features = factor_directions.flatten(1)
        expected_raw = mean_gram + raw_features @ raw_features.T
        torch.testing.assert_close(compute_transport_kernel(model, conditions), expected_raw)

        with torch.no_grad():
            _, factors = model(conditions)
        for canonical in (False, True):
            with self.subTest(canonical=canonical):
                projected = project_factor_directions(factors, factor_directions, canonical=canonical).flatten(1)
                expected = mean_gram + projected @ projected.T
                actual = compute_projected_transport_kernel(model, conditions, canonical=canonical)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_projection_steers_one_query_from_latent_rotation_to_covariance_change(self):
        # Observed condition, informative candidate, then pure latent rotation.
        factors = torch.eye(2, dtype=torch.float64).expand(3, -1, -1)
        directions = factors.new_zeros((3, 2, 2, 2))
        directions[1, 0, 0, 0] = 1
        directions[2, 1] = tensor([[0, -10], [10, 0]])
        raw_features = directions.flatten(1)
        projected_features = project_factor_directions(factors, directions).flatten(1)
        raw_gram = raw_features @ raw_features.T
        projected_gram = projected_features @ projected_features.T
        torch.testing.assert_close(raw_gram.diag(), tensor([0, 1, 200]))
        torch.testing.assert_close(projected_gram.diag(), tensor([0, 1, 0]))

        settings = dict(pool_indices=[0, 1, 2], train_indices=[0], batch_size=1, lam=1e-3)
        self.assertEqual(select_batch_min_total_var(raw_gram, **settings), [2])
        self.assertEqual(select_batch_min_total_var(projected_gram, **settings), [1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
