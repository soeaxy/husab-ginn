"""Opt-in integrity check of the completed real-data OGR experiment artifacts.

Run with OGR_VERIFY_ARTIFACTS=1 after all batches finish. Synthetic fixtures are
never substituted for missing experiment outputs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from run_ogr_revision_experiments import (
    ANALYSIS, ARCHIVE, BASELINE, ROOT, experiment_matrix, manifest_path,
    strict_zero_matrix, write_json,
)
from run_reconstructed_benchmarks import sha256_file


@unittest.skipUnless(os.environ.get("OGR_VERIFY_ARTIFACTS") == "1", "Real-data batch not requested for artifact audit")
class OGRArtifactIntegrityTests(unittest.TestCase):
    def test_all_real_data_runs_match_the_declared_protocol(self) -> None:
        runs = [BASELINE] + experiment_matrix() + strict_zero_matrix()
        self.assertEqual(len(runs), 72)
        audited = []
        for run in runs:
            with self.subTest(run=run.key):
                output = run.output
                completed = json.loads((output / "completed.json").read_text(encoding="utf-8"))
                self.assertEqual(completed["status"], "completed")
                self.assertEqual(completed["exit_code"], 0)
                config = json.loads((output / "metrics" / "run_config.json").read_text(encoding="utf-8"))
                expected_sample = ROOT / "data" / "reconstructed_samples_v2" / f"realization_{run.realization:02d}" / "combined_samples_rebuilt.shp"
                self.assertEqual(Path(config["data_config"]["shapefile_path"]).resolve(), expected_sample.resolve())
                self.assertEqual([Path(path).resolve() for path in config["data_config"]["feature_dirs"]], [(ROOT / "data" / "factors").resolve()])
                train = config["train_config"]
                self.assertEqual(train["seed"], run.seed)
                self.assertEqual(train["split_seed"], 2026)
                self.assertEqual(train["selection_metric"], "pr_auc")
                self.assertEqual(train["physics_weight"], run.physics_weight)
                self.assertEqual(train["epochs"], 1 if run.algorithm == "rf" else 80)
                self.assertEqual(train["patience"], 20)
                self.assertEqual(train["batch_size_cpu"], 2048)
                self.assertEqual(train["fixed_prior_weights"], run.fixed_weights)
                self.assertEqual(config["device"], "cpu")
                self.assertEqual(config["preprocessing_fit_partition"], "train")
                self.assertEqual(config["algo_config"]["algorithm"], run.algorithm)
                for prior in ("dome", "fault", "strata"):
                    self.assertEqual(config["enabled_priors"][prior], run.algorithm == "pinn" and prior != run.omit_prior)
                self.assertFalse(config["enabled_priors"]["singularity"])
                expected_scales = {"dome_width": 1000.0, "fault_rate": 0.002, "strata_width": 800.0}
                if run.scale_argument:
                    expected_scales[run.scale_argument.replace("-", "_")] = run.scale_value
                for key, value in expected_scales.items():
                    self.assertEqual(train[key], value)

                splits = pd.read_csv(output / "metrics" / "split_assignments.csv")
                self.assertFalse(splits.sample_id.duplicated().any())
                self.assertEqual(set(splits.partition), {"train", "validation", "test"})
                self.assertTrue((splits.groupby("block_id").partition.nunique() == 1).all())
                if run.grid == 5:
                    reference = ARCHIVE / f"realization_{run.realization:02d}" / "pinn" / "seed_2026" / "metrics" / "split_assignments.csv"
                    pd.testing.assert_frame_equal(splits, pd.read_csv(reference), check_exact=True)
                metrics = json.loads((output / "metrics" / "metrics.json").read_text(encoding="utf-8"))
                for partition in ("validation", "test"):
                    predictions = pd.read_csv(output / "metrics" / f"{partition}_predictions.csv")
                    selected = splits[splits.partition == partition]
                    self.assertEqual(predictions.row_index.tolist(), selected.row_index.tolist())
                    self.assertEqual(predictions.sample_id.tolist(), selected.sample_id.tolist())
                    self.assertEqual(predictions.label.tolist(), selected.label.tolist())
                    self.assertTrue(np.isfinite(predictions.probability).all())
                    self.assertTrue(predictions.probability.between(0, 1).all())
                    y, p = predictions.label.to_numpy(), predictions.probability.to_numpy(dtype=np.float32)
                    computed = {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p), "brier_score": brier_score_loss(y, p)}
                    for metric, value in computed.items():
                        self.assertAlmostEqual(value, metrics[f"{partition}_metrics"][metric], places=10)
                    self.assertEqual(metrics[f"{partition}_metrics"]["n_samples"], len(predictions))
                if run.algorithm == "pinn":
                    params = json.loads((output / "metrics" / "model_parameters.json").read_text(encoding="utf-8"))
                    self.assertEqual(params["total_params"], 2790)
                    self.assertEqual(params["trainable_params"], 2786 if run.fixed_weights else 2790)
                    if run.fixed_weights or run.physics_weight == 0:
                        self.assertTrue(all(sigma == 1.0 for sigma in params["learned_sigma"].values()))
                    if run.fixed_weights:
                        self.assertTrue(all(not params["per_parameter_stats"][f"log_sigma_{prior}"]["requires_grad"] for prior in ("dome", "fault", "strata", "singularity")))
                if run.algorithm != "rf":
                    history = pd.read_csv(output / "metrics" / "train_history.csv")
                    self.assertLessEqual(history.epoch.max(), 80)
                    self.assertLessEqual(abs(history.val_selection_score.max() - metrics["best_validation_selection_score"]), 1e-6)
                audited.append({"key": run.key, "test_pr_auc": metrics["test_metrics"]["pr_auc"], "n_test": metrics["test_metrics"]["n_samples"], "split_sha256": sha256_file(manifest_path(run.grid)), "prediction_sha256": sha256_file(output / "metrics" / "test_predictions.csv")})
        self.assertEqual(len(audited), len(runs), "Do not publish a passing audit when any run failed a check.")
        for grid in (4, 10):
            selected_runs = [run for run in runs if run.axis == "spatial_grid" and run.grid == grid]
            splits = [pd.read_csv(run.output / "metrics" / "split_assignments.csv") for run in selected_runs]
            for split in splits[1:]:
                pd.testing.assert_frame_equal(splits[0], split, check_exact=True)
        write_json(ANALYSIS / "artifact_integrity_audit.json", {"passed": True, "real_data_runs_audited": len(audited), "checks": ["config", "repeat axes", "prior enablement", "parameter scales", "train-only preprocessing", "spatial partition isolation", "exact reference split", "prediction-label alignment", "recomputed AP ROC-AUC Brier", "fixed-weight parameters", "validation-only checkpoint selection", "matched grid model partitions"], "current_training_sources_sha256": {name: sha256_file(ROOT / name) for name in ("train_multi_physics_model.py", "physics_informed_model.py")}, "runs": audited})


if __name__ == "__main__":
    unittest.main()
