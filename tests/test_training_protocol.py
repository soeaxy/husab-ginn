from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mineral_deep_models import build_deep_compare_model, build_spatial_ego_graphs
from run_reconstructed_benchmarks import build_command, build_parser
from train_multi_physics_model import (
    OutputConfig,
    fit_preprocessor,
    indices_from_block_partitions,
    load_spatial_split_manifest,
    spatial_block_groups,
    transform_features,
    validation_selection_score,
)


class TrainingProtocolTests(unittest.TestCase):
    def test_preprocessor_is_fit_on_training_rows_only(self) -> None:
        train = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        held_out = np.array([[1000.0, 1001.0]], dtype=np.float32)

        train_scaled, imputer, selector, scaler, names = fit_preprocessor(train, ["a", "b"])
        held_out_scaled = transform_features(held_out, imputer, selector, scaler)

        self.assertEqual(names, ["a", "b"])
        np.testing.assert_allclose(train_scaled.mean(axis=0), np.zeros(2), atol=1e-6)
        self.assertGreater(float(held_out_scaled.min()), 100.0)

    def test_spatial_groups_respect_explicit_bounds(self) -> None:
        coords = np.array([[0.0, 0.0], [9.9, 9.9], [5.1, 0.1]], dtype=np.float64)
        groups = spatial_block_groups(coords, grid_size=2, bounds=(0.0, 0.0, 10.0, 10.0))
        np.testing.assert_array_equal(groups, np.array([0, 3, 2]))

    def test_fixed_block_partitions_are_complete_and_disjoint(self) -> None:
        groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        partitions = {"train": [0], "validation": [1], "test": [2]}

        train_idx, val_idx, test_idx = indices_from_block_partitions(groups, labels, partitions)

        np.testing.assert_array_equal(train_idx, np.array([0, 1]))
        np.testing.assert_array_equal(val_idx, np.array([2, 3]))
        np.testing.assert_array_equal(test_idx, np.array([4, 5]))
        self.assertEqual(set(train_idx) | set(val_idx) | set(test_idx), set(range(6)))
        self.assertFalse(set(train_idx) & set(val_idx))
        self.assertFalse(set(train_idx) & set(test_idx))
        self.assertFalse(set(val_idx) & set(test_idx))

    def test_fixed_block_partitions_reject_missing_blocks(self) -> None:
        groups = np.array([0, 1, 2], dtype=np.int64)
        labels = np.array([0, 1, 0], dtype=np.int64)
        partitions = {"train": [0], "validation": [1], "test": []}
        with self.assertRaisesRegex(ValueError, "assign every populated block"):
            indices_from_block_partitions(groups, labels, partitions)

    def test_gcn_transformer_and_mlp_have_expected_output_shape(self) -> None:
        graph_model, graph_meta = build_deep_compare_model("gcn_transformer", input_dim=3)
        mlp_model, mlp_meta = build_deep_compare_model("mlp", input_dim=3)

        self.assertEqual(graph_model(torch.zeros(2, 9, 3)).shape, (2, 2))
        self.assertEqual(mlp_model(torch.zeros(2, 3)).shape, (2, 2))
        self.assertEqual(graph_meta["neighborhood_size"], 8)
        self.assertEqual(mlp_meta["input_dim"], 3)

    def test_spatial_ego_graph_excludes_query_node_from_neighbors(self) -> None:
        features = np.arange(12, dtype=np.float32).reshape(4, 3)
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        graphs = build_spatial_ego_graphs(
            features,
            coords,
            features,
            coords,
            k_neighbors=2,
            exclude_self=True,
        )
        self.assertEqual(graphs.shape, (4, 3, 3))
        for row in range(graphs.shape[0]):
            self.assertFalse(np.any(np.all(graphs[row, 1:] == features[row], axis=1)))

    def test_torch_models_use_pth_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for algorithm in ("pinn", "mlp", "gcn_transformer", "gnn", "vae", "transunet"):
                cfg = OutputConfig(root, "model", algorithm=algorithm)
                self.assertEqual(cfg.model_path.suffix, ".pth")

    def test_split_manifest_loader_accepts_fixed_protocol(self) -> None:
        payload = {
            "split_mode": "spatial_block",
            "grid_size": 5,
            "bounds": {"min_x": 0, "min_y": 1, "max_x": 10, "max_y": 11},
            "partitions": {"train": [0, 1], "validation": [2], "test": [3]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            grid, bounds, partitions, loaded = load_spatial_split_manifest(path)
        self.assertEqual(grid, 5)
        self.assertEqual(bounds, (0.0, 1.0, 10.0, 11.0))
        self.assertEqual(partitions["test"], [3])
        self.assertEqual(loaded["split_mode"], "spatial_block")

    def test_benchmark_command_adds_priors_only_for_pinn(self) -> None:
        common = {
            "train_script": Path("train.py"),
            "sample_path": Path("samples.shp"),
            "feature_dir": Path("factors"),
            "prior_dir": Path("priors"),
            "output_dir": Path("out"),
            "split_manifest": Path("split.json"),
            "model_seed": 2026,
            "split_seed": 2026,
            "grid_size": 5,
            "deep_epochs": 80,
            "patience": 20,
            "physics_weight": 0.1,
            "selection_metric": "pr_auc",
            "device": "cpu",
        }
        pinn = build_command(algorithm="pinn", **common)
        graph = build_command(algorithm="gcn_transformer", **common)
        self.assertIn("--dome-tif", pinn)
        self.assertNotIn("--dome-tif", graph)
        self.assertIn("--split-manifest", graph)
        self.assertEqual(graph[graph.index("--selection-metric") + 1], "pr_auc")

    def test_formal_benchmark_defaults_to_pr_auc_checkpoint_selection(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.selection_metric, "pr_auc")

    def test_validation_selection_score_uses_requested_metric(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        probabilities = np.array([0.1, 0.8, 0.7, 0.9], dtype=np.float64)

        roc_score, roc_auc, pr_auc = validation_selection_score(labels, probabilities, "roc_auc")
        pr_score, repeated_roc_auc, repeated_pr_auc = validation_selection_score(labels, probabilities, "pr_auc")

        self.assertAlmostEqual(roc_score, roc_auc)
        self.assertAlmostEqual(pr_score, pr_auc)
        self.assertAlmostEqual(repeated_roc_auc, roc_auc)
        self.assertAlmostEqual(repeated_pr_auc, pr_auc)
        self.assertNotAlmostEqual(roc_auc, pr_auc)


if __name__ == "__main__":
    unittest.main()
