from __future__ import annotations

import unittest

import numpy as np
import torch

from physics_informed_model import PhysicsInformedClassifier
from run_ogr_revision_experiments import BASELINE, Run, command_for, experiment_matrix, strict_zero_matrix
from train_multi_physics_model import build_parser, make_geological_prior


class OGRRevisionProtocolTests(unittest.TestCase):
    def test_zero_regularization_has_cross_entropy_network_gradients(self) -> None:
        torch.manual_seed(2026)
        regularized = PhysicsInformedClassifier(3)
        torch.manual_seed(2026)
        ce_only = PhysicsInformedClassifier(3)
        x = torch.randn(8, 3)
        labels = torch.tensor([0, 1] * 4)
        prior = torch.rand(8, 1)
        logits = regularized(x)
        ce = torch.nn.functional.cross_entropy(logits, labels)
        loss = ce + 0.0 * regularized.physics_loss(logits, prior, prior, prior)
        torch.testing.assert_close(loss.squeeze(), ce, rtol=0, atol=0)
        loss.backward()
        torch.nn.functional.cross_entropy(ce_only(x), labels).backward()
        for name, parameter in regularized.named_parameters():
            if not name.startswith("log_sigma"):
                torch.testing.assert_close(parameter.grad, dict(ce_only.named_parameters())[name].grad, rtol=0, atol=0)

    def test_experiment_axes_and_commands_are_not_conflated(self) -> None:
        runs = experiment_matrix()
        self.assertEqual(len(runs), 61)
        self.assertEqual(len({run.key for run in runs}), 61)
        self.assertEqual(sum(run.axis == "negative_realization" for run in runs), 40)
        self.assertEqual(sum(run.axis == "prior_scale" for run in runs), 6)
        self.assertEqual(sum(run.axis == "initialization" for run in runs), 9)
        self.assertEqual(sum(run.axis == "spatial_grid" for run in runs), 6)
        for run in [BASELINE] + runs:
            command = command_for(run, 2)
            self.assertEqual(command[command.index("--split-seed") + 1], "2026")
            self.assertEqual(command[command.index("--selection-metric") + 1], "pr_auc")
            self.assertEqual(command[command.index("--seed") + 1], str(run.seed))
            if run.omit_prior:
                self.assertNotIn(f"--{run.omit_prior}-tif", command)
                self.assertEqual(sum(f"--{name}-tif" in command for name in ("dome", "fault", "strata")), 2)
        mlp = command_for(Run("spatial_grid", "grid4_mlp", algorithm="mlp", grid=4), 2)
        self.assertNotIn("--dome-tif", mlp)
        self.assertEqual(len(strict_zero_matrix()), 10)
        for run in strict_zero_matrix():
            command = command_for(run, 2)
            self.assertEqual(command[command.index("--physics-weight") + 1], "0.0")
            self.assertEqual(command[command.index("--algorithm") + 1], "pinn")
            self.assertTrue(all(f"--{name}-tif" in command for name in ("dome", "fault", "strata")))

    def test_defaults_preserve_original_prior_functions(self) -> None:
        distances = np.array([0, 100, 500, 1000, 5000], dtype=np.float32)
        expected = {
            "dome": np.exp(-(distances**2 / (2 * 1000.0**2))),
            "fault": np.exp(-0.002 * distances),
            "strata": np.exp(-(distances**2 / (2 * 800.0**2))),
        }
        for name, values in expected.items():
            np.testing.assert_array_equal(make_geological_prior(distances, name), values)
        args = build_parser().parse_args([])
        self.assertFalse(args.fixed_prior_weights)
        self.assertIsNone(args.torch_threads)
        self.assertEqual((args.dome_width, args.fault_rate, args.strata_width), (1000.0, 0.002, 800.0))

    def test_fixed_weights_preserve_initialization_and_remain_zero(self) -> None:
        torch.manual_seed(2026)
        adaptive = PhysicsInformedClassifier(3)
        torch.manual_seed(2026)
        fixed = PhysicsInformedClassifier(3, fixed_prior_weights=True)
        for name, value in adaptive.state_dict().items():
            torch.testing.assert_close(value, fixed.state_dict()[name], rtol=0, atol=0)
        optimizer = torch.optim.AdamW(fixed.parameters(), lr=0.01)
        features = torch.randn(8, 3)
        prior = torch.full((8, 1), 0.25)
        before = fixed.fc1.weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad()
            logits = fixed(features)
            loss = fixed.physics_loss(logits, prior, prior, prior)
            expected = 1.5 * ((logits.softmax(dim=1)[:, 1:2] - prior) ** 2).mean()
            torch.testing.assert_close(loss.squeeze(), expected)
            loss.backward()
            optimizer.step()
        for name, parameter in fixed.named_parameters():
            if name.startswith("log_sigma"):
                self.assertFalse(parameter.requires_grad)
                self.assertEqual(parameter.item(), 0.0)
        self.assertFalse(torch.equal(before, fixed.fc1.weight))
        self.assertTrue(adaptive.log_sigma_dome.requires_grad)

    def test_scales_have_expected_direction_and_independence(self) -> None:
        d = np.array([500.0, 1000.0], dtype=np.float32)
        for name, argument, default in (("dome", "dome_width", 1000.0), ("strata", "strata_width", 800.0)):
            base = make_geological_prior(d, name)
            self.assertTrue(np.all(make_geological_prior(d, name, **{argument: default * 2}) > base))
            self.assertTrue(np.all(make_geological_prior(d, name, **{argument: default / 2}) < base))
        self.assertTrue(np.all(make_geological_prior(d, "fault", fault_rate=0.004) < make_geological_prior(d, "fault")))
        np.testing.assert_array_equal(make_geological_prior(d, "dome", fault_rate=0.004), make_geological_prior(d, "dome"))
        for invalid in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                make_geological_prior(d, "dome", dome_width=invalid)


if __name__ == "__main__":
    unittest.main()
