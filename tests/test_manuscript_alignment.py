from __future__ import annotations

import inspect
import json
import re
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import train_multi_physics_model as training
from manuscript_protocol import (
    ACTIVE_GEOLOGICAL_PRIORS,
    BENCHMARK_ALGORITHMS,
    DEEP_BASELINE_LR_SCHEDULER_PATIENCE,
    GCN_TRANSFORMER_CONFIG,
    GINN_ACTIVE_PARAMETER_COUNT,
    GINN_LR_SCHEDULER_PATIENCE,
    GINN_REGISTERED_PARAMETER_COUNT,
    HUSAB_POSITIVE_CANDIDATE_COUNT,
    HUSAB_POSITIVE_SAMPLE_COUNT,
    PSEUDO_ABSENCE_SEEDS,
    SYNTHETIC_EXAMPLE_BOUNDS,
    SPATIAL_GRID_SIZE,
    SPATIAL_PARTITIONS,
    SUPPLEMENTARY_REAL_DATA_RUNS,
    manuscript_model_label,
    normalize_algorithm_key,
)
from mineral_deep_models import build_deep_compare_model
from mineral_prediction_api.schemas import TrainJobRequest
from physics_informed_model import GeologyInformedClassifier, PhysicsInformedClassifier
from plot_manuscript_evidence_layers import (
    EVIDENCE_LAYERS,
    suppress_coordinate_information,
)
from run_reconstructed_benchmarks import DEFAULT_ALGORITHMS

ROOT = Path(__file__).resolve().parents[1]


class ManuscriptAlignmentTests(unittest.TestCase):
    def test_public_ginn_name_preserves_archived_key(self) -> None:
        self.assertEqual(normalize_algorithm_key("ginn"), "pinn")
        self.assertEqual(normalize_algorithm_key("PINN"), "pinn")
        self.assertEqual(manuscript_model_label("pinn"), "GINN")
        self.assertIs(PhysicsInformedClassifier, GeologyInformedClassifier)
        parsed = training.build_parser().parse_args(["--algorithm", "ginn"])
        self.assertEqual(parsed.algorithm, "ginn")
        api_default = TrainJobRequest(shapefile="samples.shp", feature_dirs=["factors"])
        self.assertEqual(api_default.algorithm, "ginn")
        api_legacy = TrainJobRequest(
            shapefile="samples.shp", feature_dirs=["factors"], algorithm="pinn"
        )
        self.assertEqual(api_legacy.algorithm, "pinn")

    def test_ginn_parameter_counts_match_manuscript(self) -> None:
        model = GeologyInformedClassifier(input_dim=9)
        registered = sum(parameter.numel() for parameter in model.parameters())
        active = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(registered, GINN_REGISTERED_PARAMETER_COUNT)
        self.assertEqual(active, GINN_REGISTERED_PARAMETER_COUNT)
        active_without_legacy = active - model.log_sigma_singularity.numel()
        self.assertEqual(active_without_legacy, GINN_ACTIVE_PARAMETER_COUNT)
        self.assertTupleEqual(ACTIVE_GEOLOGICAL_PRIORS, ("dome", "fault", "strata"))

    def test_gcn_transformer_configuration_and_parameter_count(self) -> None:
        model, metadata = build_deep_compare_model("gcn_transformer", input_dim=9)
        self.assertEqual(
            metadata["neighborhood_size"], GCN_TRANSFORMER_CONFIG["neighborhood_size"]
        )
        self.assertEqual(metadata["d_model"], GCN_TRANSFORMER_CONFIG["d_model"])
        self.assertEqual(metadata["nhead"], GCN_TRANSFORMER_CONFIG["nhead"])
        self.assertEqual(metadata["num_layers"], GCN_TRANSFORMER_CONFIG["num_layers"])
        self.assertEqual(
            metadata["dim_feedforward"], GCN_TRANSFORMER_CONFIG["dim_feedforward"]
        )
        self.assertFalse(metadata["neighbor_labels_used"])
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(
            parameter_count, GCN_TRANSFORMER_CONFIG["trainable_parameters"]
        )

    def test_scheduler_direction_matches_executed_code(self) -> None:
        deep_source = inspect.getsource(training.train_deep_compare_model)
        ginn_source = inspect.getsource(training.train_model)
        self.assertIn("DEEP_BASELINE_LR_SCHEDULER_PATIENCE", deep_source)
        self.assertIn("GINN_LR_SCHEDULER_PATIENCE", ginn_source)
        self.assertEqual(DEEP_BASELINE_LR_SCHEDULER_PATIENCE, 10)
        self.assertEqual(GINN_LR_SCHEDULER_PATIENCE, 15)

    def test_benchmark_and_sample_contract(self) -> None:
        self.assertTupleEqual(DEFAULT_ALGORITHMS, BENCHMARK_ALGORITHMS)
        self.assertEqual(len(DEFAULT_ALGORITHMS), 7)
        self.assertEqual(len(PSEUDO_ABSENCE_SEEDS), 10)
        self.assertEqual(PSEUDO_ABSENCE_SEEDS[0], 2025)
        self.assertEqual(PSEUDO_ABSENCE_SEEDS[-1], 11106)
        self.assertEqual(HUSAB_POSITIVE_CANDIDATE_COUNT, 41_980)
        self.assertEqual(HUSAB_POSITIVE_SAMPLE_COUNT, 4_198)
        self.assertEqual(SUPPLEMENTARY_REAL_DATA_RUNS, 76)

    def test_synthetic_split_manifest_executes_complete_disjoint_split(self) -> None:
        manifest = json.loads(
            (ROOT / "configs" / "synthetic_spatial_split.example.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIs(manifest["synthetic"], True)
        self.assertNotIn("crs", manifest)
        self.assertEqual(manifest["grid_size"], SPATIAL_GRID_SIZE)
        bounds = manifest["bounds"]
        self.assertTupleEqual(
            (bounds["min_x"], bounds["min_y"], bounds["max_x"], bounds["max_y"]),
            SYNTHETIC_EXAMPLE_BOUNDS,
        )
        self.assertNotEqual(manifest["partitions"], SPATIAL_PARTITIONS)
        all_blocks = [
            block for blocks in manifest["partitions"].values() for block in blocks
        ]
        self.assertEqual(sorted(all_blocks), list(range(25)))
        coords = np.array(
            [(x + 500.0, y + 500.0) for x in range(0, 5000, 1000)
             for y in range(0, 5000, 1000) for _ in (0, 1)]
        )
        labels = np.tile([0, 1], 25)
        groups = training.spatial_block_groups(
            coords, SPATIAL_GRID_SIZE, SYNTHETIC_EXAMPLE_BOUNDS
        )
        partitions = training.indices_from_block_partitions(
            groups, labels, manifest["partitions"]
        )
        self.assertEqual([len(indices) for indices in partitions], [30, 10, 10])
        self.assertEqual(sorted(np.concatenate(partitions).tolist()), list(range(50)))

    def test_manuscript_figure_has_nine_layers_and_suppresses_coordinates(self) -> None:
        self.assertEqual(len(EVIDENCE_LAYERS), 9)
        figure, axis = plt.subplots()
        axis.set_xlabel("Easting (km)")
        axis.set_ylabel("Northing (km)")
        axis.grid(True)
        suppress_coordinate_information(axis)
        self.assertEqual(axis.get_xlabel(), "")
        self.assertEqual(axis.get_ylabel(), "")
        self.assertFalse(any(line.get_visible() for line in axis.get_xgridlines()))
        self.assertFalse(
            any(tick.tick1line.get_visible() for tick in axis.xaxis.get_major_ticks())
        )
        plt.close(figure)

    def test_readme_states_non_superiority_and_release_boundary(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for token in (
            "geology-informed neural network (GINN)",
            "not a general superiority claim",
            "41,980",
            "4,198",
            "76 additional real-data runs",
            "data have not been authorized for public release",
            "https://github.com/soeaxy/husab-ginn",
        ):
            self.assertIn(token, readme)

    def test_code_release_sources_have_no_local_absolute_paths(self) -> None:
        excluded = {
            "build_field_verification_points.py",
            "build_green_red_probability_map.py",
            "build_rectangular_probability_maps.py",
            "analyze_evidence_guided_ensemble.py",
        }
        paths = [path for path in ROOT.glob("*.py") if path.name not in excluded]
        paths.extend((ROOT / "src").rglob("*.py"))
        paths.extend((ROOT / "docs").rglob("*.md"))
        windows_absolute = re.compile(r"(?i)(?:^|[\"'`\s(])[a-z]:[\\/]")
        offenders = []
        for path in paths:
            text = path.read_text(encoding="utf-8-sig")
            if windows_absolute.search(text):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertListEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
