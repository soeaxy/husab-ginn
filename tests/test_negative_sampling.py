import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

import rebuild_negative_samples as rns
from train_multi_physics_model import (
    DataConfig,
    grouped_holdout_indices,
    load_samples as train_load_samples,
    spatial_block_groups,
)


class NegativeSamplingTests(unittest.TestCase):
    def test_compute_quotas_returns_expected_counts(self) -> None:
        config = rns.SamplingConfig(
            hard_fraction=0.70,
            transition_fraction=0.05,
            background_fraction=0.25,
        )

        quotas = rns.compute_quotas(20, config)

        self.assertEqual(quotas, {"hard": 14, "transition": 1, "background": 5})

    def test_assign_strata_prioritizes_hard_then_transition_then_background(self) -> None:
        config = rns.SamplingConfig(
            hard_prior_threshold=0.50,
            transition_prior_threshold=0.20,
        )

        strata = rns.assign_strata(
            geo_score=np.array([0.10, 0.30, 0.60, 0.05, 0.80]),
            in_structure=np.array([False, False, False, True, False]),
            in_lithology=np.array([False, False, False, False, True]),
            eligible=np.array([True, True, True, True, False]),
            config=config,
        )

        self.assertListEqual(
            strata.tolist(),
            ["background", "transition", "hard", "hard", "ineligible"],
        )

    def test_select_spatially_balanced_is_deterministic_for_fixed_seed(self) -> None:
        candidate_indices = np.arange(12, dtype=np.int64)
        thin_ids = np.arange(12, dtype=np.int64)
        block_ids = np.repeat(np.arange(4, dtype=np.int64), 3)

        first = rns.select_spatially_balanced(
            candidate_indices,
            thin_ids,
            block_ids,
            quota=6,
            rng=np.random.default_rng(2025),
        )
        second = rns.select_spatially_balanced(
            candidate_indices,
            thin_ids,
            block_ids,
            quota=6,
            rng=np.random.default_rng(2025),
        )

        np.testing.assert_array_equal(first, second)

    def test_select_spatially_balanced_changes_selection_for_different_seed(self) -> None:
        candidate_indices = np.arange(12, dtype=np.int64)
        thin_ids = np.arange(12, dtype=np.int64)
        block_ids = np.repeat(np.arange(4, dtype=np.int64), 3)

        first = rns.select_spatially_balanced(
            candidate_indices,
            thin_ids,
            block_ids,
            quota=6,
            rng=np.random.default_rng(2025),
        )
        second = rns.select_spatially_balanced(
            candidate_indices,
            thin_ids,
            block_ids,
            quota=6,
            rng=np.random.default_rng(2026),
        )

        self.assertFalse(np.array_equal(first, second))

    def test_select_spatially_balanced_keeps_unique_thinning_cells(self) -> None:
        candidate_indices = np.arange(8, dtype=np.int64)
        thin_ids = np.array([10, 10, 20, 20, 30, 30, 40, 40], dtype=np.int64)
        block_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)

        selected = rns.select_spatially_balanced(
            candidate_indices,
            thin_ids,
            block_ids,
            quota=4,
            rng=np.random.default_rng(2025),
        )

        self.assertEqual(np.unique(thin_ids[selected]).size, selected.size)

    def test_validate_training_samples_rejects_negative_inside_positive_buffer(self) -> None:
        samples = self._make_training_samples(
            positive_points=[(1000.0, 1000.0)],
            negative_points=[(1005.0, 1005.0)],
        )
        positive_buffer = Point(1000.0, 1000.0).buffer(20.0)

        with self.assertRaisesRegex(ValueError, "positive exclusion buffer"):
            rns.validate_training_samples(
                samples,
                expected_positive=1,
                expected_negative=1,
                positive_buffer=positive_buffer,
            )

    def test_validate_training_samples_enforces_exact_mine_distance(self) -> None:
        samples = self._make_training_samples(
            positive_points=[(1000.0, 1000.0)],
            negative_points=[(2000.0, 1000.0)],
        )
        samples["mine_dist"] = [0.0, 999.999]

        with self.assertRaisesRegex(ValueError, "minimum mine-distance"):
            rns.validate_training_samples(
                samples,
                expected_positive=1,
                expected_negative=1,
                minimum_negative_distance_m=1000.0,
            )

    def test_validate_and_train_loader_accept_training_schema(self) -> None:
        samples = self._make_training_samples(
            positive_points=[(1000.0, 1000.0), (1200.0, 1200.0)],
            negative_points=[(2000.0, 2000.0), (2200.0, 2200.0)],
        )
        positive_buffer = Point(1000.0, 1000.0).buffer(50.0).union(
            Point(1200.0, 1200.0).buffer(50.0)
        )

        validation = rns.validate_training_samples(
            samples,
            expected_positive=2,
            expected_negative=2,
            positive_buffer=positive_buffer,
        )

        self.assertEqual(validation["rows"], 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = Path(tmpdir) / "combined_samples_rebuilt.shp"
            rns.write_shapefile(samples, shp_path)
            cfg = DataConfig(
                shapefile_path=shp_path,
                feature_dirs=[],
                dome_tif_path=None,
                fault_tif_path=None,
                strata_tif_path=None,
                singularity_tif_path=None,
                label_col="Class",
            )

            loaded = train_load_samples(cfg)

        self.assertEqual(len(loaded), 4)
        self.assertSetEqual(set(loaded["Class"].astype(int).unique()), {0, 1})
        self.assertTrue((loaded.geom_type == "Point").all())

    def test_spatial_block_split_retains_both_classes(self) -> None:
        samples = self._make_training_samples(
            positive_points=[
                (1000.0, 1000.0),
                (1200.0, 1200.0),
                (5000.0, 5000.0),
                (5200.0, 5200.0),
            ],
            negative_points=[
                (2000.0, 2000.0),
                (2200.0, 2200.0),
                (7000.0, 7000.0),
                (7200.0, 7200.0),
            ],
        )
        coords = np.column_stack([samples.geometry.x.to_numpy(), samples.geometry.y.to_numpy()])
        labels = samples["Class"].to_numpy(dtype=int)
        groups = spatial_block_groups(coords, grid_size=2)

        train_idx, test_idx = grouped_holdout_indices(
            indices=np.arange(len(samples), dtype=np.int64),
            labels=labels,
            groups=groups,
            test_size=0.25,
            seed=2025,
            trials=32,
        )

        self.assertGreaterEqual(np.unique(labels[train_idx]).size, 2)
        self.assertGreaterEqual(np.unique(labels[test_idx]).size, 2)
        self.assertEqual(len(set(train_idx) & set(test_idx)), 0)

    def test_clear_generated_output_refuses_unknown_files_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "reconstructed"
            output_dir.mkdir()
            unknown = output_dir / "keep_me.txt"
            unknown.write_text("user data", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not owned by this generator"):
                rns.clear_generated_output(output_dir)

            self.assertTrue(unknown.exists())

    def test_clear_generated_output_removes_only_recognized_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "reconstructed"
            realization_dir = output_dir / "realization_00"
            realization_dir.mkdir(parents=True)
            (output_dir / "protocol.json").write_text("{}", encoding="utf-8")
            (output_dir / "realization_summary.csv").write_text(
                "realization\n0\n", encoding="utf-8"
            )
            (realization_dir / "manifest.json").write_text("{}", encoding="utf-8")
            (realization_dir / "negative_samples_rebuilt.shp").write_bytes(b"shp")
            (realization_dir / "negative_samples_rebuilt.dbf").write_bytes(b"dbf")

            rns.clear_generated_output(output_dir)

            self.assertTrue(output_dir.exists())
            self.assertEqual(list(output_dir.iterdir()), [])

    def _make_training_samples(
        self,
        positive_points: list[tuple[float, float]],
        negative_points: list[tuple[float, float]],
    ) -> gpd.GeoDataFrame:
        rows: list[dict[str, object]] = []
        for idx, (x, y) in enumerate(positive_points):
            rows.append(
                {
                    "sample_id": f"P{idx:07d}",
                    "Class": 1,
                    "source": "known_zone",
                    "stratum": "positive",
                    "geometry": Point(x, y),
                }
            )
        for idx, (x, y) in enumerate(negative_points):
            rows.append(
                {
                    "sample_id": f"N00H{idx:06d}",
                    "Class": 0,
                    "source": "pseudo_abs",
                    "stratum": "hard",
                    "geometry": Point(x, y),
                }
            )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32733")


if __name__ == "__main__":
    unittest.main()
