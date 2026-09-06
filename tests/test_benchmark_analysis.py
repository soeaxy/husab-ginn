from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_reconstructed_benchmarks import (
    bootstrap_mean_ci,
    compute_topk_metrics,
    collect_artifacts,
    holm_correction,
    load_benchmark_manifest,
    signed_rank_biserial,
    validate_selection_protocol,
)


class BenchmarkAnalysisTests(unittest.TestCase):
    def test_holm_correction_is_monotone_after_sorting(self) -> None:
        adjusted = holm_correction([0.01, 0.03, 0.04, 0.20])
        self.assertEqual(len(adjusted), 4)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        self.assertAlmostEqual(adjusted[0], 0.04, places=8)
        self.assertAlmostEqual(adjusted[1], 0.09, places=8)
        self.assertAlmostEqual(adjusted[2], 0.09, places=8)
        self.assertAlmostEqual(adjusted[3], 0.20, places=8)

    def test_signed_rank_biserial_ignores_zero_differences(self) -> None:
        differences = np.array([0.3, 0.2, -0.1, 0.0], dtype=float)
        effect = signed_rank_biserial(differences)
        self.assertAlmostEqual(effect, 2.0 / 3.0, places=8)

    def test_bootstrap_mean_ci_is_deterministic(self) -> None:
        values = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
        first = bootstrap_mean_ci(values, 2000, 7)
        second = bootstrap_mean_ci(values, 2000, 7)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], np.mean(values))
        self.assertGreaterEqual(first[1], np.mean(values))

    def test_compute_topk_metrics_prefers_high_scores(self) -> None:
        labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=int)
        probabilities = np.array([0.99, 0.10, 0.95, 0.20, 0.30, 0.80, 0.40, 0.50, 0.60, 0.70], dtype=float)
        metrics = compute_topk_metrics(labels, probabilities)
        self.assertAlmostEqual(metrics["top_01pct_capture"], 1.0 / 3.0, places=8)
        self.assertAlmostEqual(metrics["top_01pct_precision"], 1.0, places=8)
        self.assertAlmostEqual(metrics["top_10pct_capture"], 1.0 / 3.0, places=8)
        self.assertAlmostEqual(metrics["top_10pct_precision"], 1.0, places=8)

    def test_validate_selection_protocol_distinguishes_checkpoint_and_single_fit_models(self) -> None:
        deep_protocol = validate_selection_protocol(
            "pinn",
            {"train_config": {"selection_metric": "pr_auc"}},
            {
                "selection_metric": "pr_auc",
                "best_validation_selection_score": 0.42,
                "validation_metrics": {"pr_auc": 0.42, "roc_auc": 0.81},
            },
        )
        tree_protocol = validate_selection_protocol(
            "rf",
            {
                "train_config": {"selection_metric": "pr_auc"},
                "algo_config": {"optimizer": "none"},
            },
            {
                "selection_metric": "pr_auc",
                "best_validation_selection_score": 0.81,
                "validation_metrics": {"pr_auc": 0.30, "roc_auc": 0.81},
            },
        )
        self.assertEqual(deep_protocol, "validation_pr_auc_checkpoint")
        self.assertEqual(tree_protocol, "single_fit_no_checkpoint")

    def test_validate_selection_protocol_rejects_tuned_classical_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a single-fit estimator"):
            validate_selection_protocol(
                "lightgbm",
                {
                    "train_config": {"selection_metric": "pr_auc"},
                    "algo_config": {"optimizer": "bayes"},
                },
                {
                    "selection_metric": "pr_auc",
                    "best_validation_selection_score": 0.81,
                    "validation_metrics": {"pr_auc": 0.42, "roc_auc": 0.81},
                },
            )

    def test_validate_selection_protocol_rejects_deep_model_score_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match validation PR-AUC"):
            validate_selection_protocol(
                "gcn_transformer",
                {"train_config": {"selection_metric": "pr_auc"}},
                {
                    "selection_metric": "pr_auc",
                    "best_validation_selection_score": 0.81,
                    "validation_metrics": {"pr_auc": 0.42, "roc_auc": 0.81},
                },
            )

    def test_collect_artifacts_reports_missing_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"algorithms": ["pinn", "mlp"], "realizations": [0], "model_seed": 2026}
            (root / "benchmark_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            seed_dir = root / "realization_00" / "pinn" / "seed_2026" / "metrics"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "test_metrics": {"pr_auc": 0.4, "roc_auc": 0.8, "mcc": 0.3, "recall": 0.6, "brier_score": 0.2},
                        "validation_metrics": {"pr_auc": 0.3, "roc_auc": 0.7, "mcc": 0.2, "recall": 0.5},
                    }
                ),
                encoding="utf-8",
            )
            (seed_dir / "run_config.json").write_text(json.dumps({"split_mode": "spatial_block"}), encoding="utf-8")
            (seed_dir / "split_assignments.csv").write_text(
                "partition,block_id\ntrain,1\nvalidation,2\ntest,3\n",
                encoding="utf-8",
            )
            (seed_dir / "test_predictions.csv").write_text(
                "label,probability\n0,0.1\n1,0.7\n",
                encoding="utf-8",
            )
            loaded = load_benchmark_manifest(root)
            artifacts, missing, warnings = collect_artifacts(root, loaded)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["algorithm"], "mlp")
        self.assertFalse(warnings)


if __name__ == "__main__":
    unittest.main()
