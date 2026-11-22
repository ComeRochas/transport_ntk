"""Check the compressed rotated Grams against the actual model Jacobians."""

import unittest

import torch

from eval.gaussian_gauge_experiment import first_score_top10, latent_rotation, rotated_kernels
from train.experiment import GaussianTransportNet, select_batch_min_total_var
from train.gaussian_projection import project_factor_directions


class RotatedGaussianKernelChecks(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(230)
            self.model = GaussianTransportNet(hidden_dim=4).double().eval()
            self.x = torch.rand(8, 2, dtype=torch.float64)

    def test_rotated_head_compression_matches_autograd(self):
        with torch.no_grad():
            self.model.cov_net.head.bias[0] = 21.0
            self.assertTrue((self.model.cov_net.get_pre_activations(self.x)[:, 0] > 20).all())
        rotation = latent_rotation(self.x, 2.7)
        factor, kernels = rotated_kernels(self.model, self.x, rotation)
        parameters = {
            name: value for name, value in self.model.named_parameters() if ".head." in name
        }

        def transformed_model(*values):
            replacement = dict(zip(parameters, values))
            mean, base_factor = torch.func.functional_call(self.model, replacement, (self.x,))
            return mean, base_factor @ rotation

        mean_blocks, factor_blocks = torch.autograd.functional.jacobian(
            transformed_model, tuple(parameters.values()),
        )
        mean_features = torch.cat([block.flatten(2) for block in mean_blocks], dim=-1).flatten(1)
        factor_directions = torch.cat(
            [block.flatten(3) for block in factor_blocks], dim=-1,
        ).movedim(-1, 1)
        representations = {
            "transport": factor_directions,
            "projected": project_factor_directions(factor, factor_directions),
            "projected_canonical": project_factor_directions(factor, factor_directions, canonical=True),
        }
        for arm, rep in representations.items():
            with self.subTest(arm=arm):
                flat = rep.flatten(1)
                expected_shape = flat @ flat.T
                expected_total = expected_shape + mean_features @ mean_features.T
                actual_total, actual_shape = kernels[arm]
                torch.testing.assert_close(actual_total, expected_total, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(actual_shape, expected_shape, atol=1e-12, rtol=1e-12)

    def test_local_and_global_rotation_controls(self):
        factor, baseline = rotated_kernels(self.model, self.x, latent_rotation(self.x, 0.0))
        global_factor, global_kernels = rotated_kernels(
            self.model, self.x, latent_rotation(self.x, 1.3, constant=True),
        )
        local_factor, local_kernels = rotated_kernels(
            self.model, self.x, latent_rotation(self.x, 3.1),
        )
        for rotated in (local_factor, global_factor):
            torch.testing.assert_close(factor @ factor.mT, rotated @ rotated.mT, atol=1e-13, rtol=1e-13)
        for arm in baseline:
            original = baseline[arm][0]
            global_gram = global_kernels[arm][0]
            local_gram = local_kernels[arm][0]
            torch.testing.assert_close(global_gram, original, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(local_gram.diag(), original.diag(), atol=1e-12, rtol=1e-12)
            if arm == "projected_canonical":
                torch.testing.assert_close(local_gram, original, atol=1e-12, rtol=1e-12)
                settings = (list(range(len(self.x))), [0, 1], 2, 1e-3)
                original_batch = select_batch_min_total_var(original, *settings)
                self.assertEqual(original_batch, select_batch_min_total_var(local_gram, *settings))
                self.assertEqual(original_batch[0], first_score_top10(original, [0, 1], 1e-3)[0])
            else:
                self.assertGreater(float((local_gram - original).abs().max()), 1e-2)


if __name__ == "__main__":
    unittest.main()
