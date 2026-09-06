import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon, box

import rebuild_positive_samples as rps

CRS = "EPSG:32733"


class PositiveSamplingTests(unittest.TestCase):
    def test_grid_is_min_bound_anchored_x_outer_y_inner(self) -> None:
        study = self._frame([box(100.0, 200.0, 161.0, 261.0)])
        known = self._frame([box(99.0, 199.0, 162.0, 262.0)])

        candidates, audit = rps.build_positive_candidates(study, known, spacing_m=20.0)

        coordinates = [(point.x, point.y) for point in candidates.geometry]
        self.assertListEqual(
            coordinates,
            [
                (120.0, 220.0),
                (120.0, 240.0),
                (120.0, 260.0),
                (140.0, 220.0),
                (140.0, 240.0),
                (140.0, 260.0),
                (160.0, 220.0),
                (160.0, 240.0),
                (160.0, 260.0),
            ],
        )
        self.assertEqual(audit["full_bounding_grid_count"], 16)
        self.assertTrue((candidates["Class"] == 1).all())

    def test_strict_within_excludes_study_and_known_polygon_boundaries(self) -> None:
        study = self._frame([box(0.0, 0.0, 61.0, 61.0)])
        known = self._frame([box(20.0, 20.0, 60.0, 60.0)])

        candidates, _ = rps.build_positive_candidates(study, known, spacing_m=20.0)

        self.assertListEqual(
            [(point.x, point.y) for point in candidates.geometry],
            [(40.0, 40.0)],
        )

    def test_sampling_matches_pandas_and_is_deterministic(self) -> None:
        candidates = self._candidate_frame(30)

        first, first_audit = rps.sample_positive_candidates(candidates)
        second, second_audit = rps.sample_positive_candidates(candidates)
        expected = candidates.sample(n=3, random_state=1, replace=False)

        self.assertListEqual(first["cand_id"].tolist(), expected["cand_id"].tolist())
        self.assertListEqual(first["cand_id"].tolist(), second["cand_id"].tolist())
        self.assertDictEqual(first_audit, second_audit)

    def test_sampling_is_floor_division_without_replacement(self) -> None:
        candidates = self._candidate_frame(29)

        sampled, audit = rps.sample_positive_candidates(candidates)

        self.assertEqual(len(sampled), 2)
        self.assertEqual(audit["sample_size_before_tailings"], 2)
        self.assertEqual(sampled["cand_id"].nunique(), 2)
        self.assertTrue(sampled["cand_id"].isin(candidates["cand_id"]).all())

    def test_tailings_exclusion_occurs_after_sampling_and_includes_boundary(
        self,
    ) -> None:
        candidates = self._candidate_frame(20)
        presample = candidates.sample(n=2, random_state=1, replace=False)
        selected_point = presample.geometry.iloc[0]
        tailings = self._frame(
            [
                Polygon(
                    [
                        (selected_point.x, selected_point.y),
                        (selected_point.x + 0.5, selected_point.y),
                        (selected_point.x + 0.5, selected_point.y + 0.5),
                    ]
                )
            ]
        )

        sampled, audit = rps.sample_positive_candidates(candidates, tailings)

        self.assertEqual(audit["sample_size_before_tailings"], 2)
        self.assertEqual(audit["tailings_intersection_excluded_count"], 1)
        self.assertNotIn(presample["cand_id"].iloc[0], sampled["cand_id"].tolist())
        self.assertIn(presample["cand_id"].iloc[1], sampled["cand_id"].tolist())

    def test_cli_run_writes_hash_manifest_and_preserves_unknown_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            study_path = root / "study.geojson"
            known_path = root / "known.geojson"
            tailings_path = root / "tailings.geojson"
            output_dir = root / "outputs"
            output_dir.mkdir()
            unknown = output_dir / "keep_me.txt"
            unknown.write_text("not owned by the generator", encoding="utf-8")
            self._frame([box(0.0, 0.0, 221.0, 221.0)]).to_file(
                study_path, driver="GeoJSON"
            )
            self._frame([box(0.0, 0.0, 221.0, 221.0)]).to_file(
                known_path, driver="GeoJSON"
            )
            self._frame([box(500.0, 500.0, 501.0, 501.0)]).to_file(
                tailings_path, driver="GeoJSON"
            )
            args = rps.parse_args(
                [
                    "--study-area",
                    str(study_path),
                    "--known-polygons",
                    str(known_path),
                    "--tailings",
                    str(tailings_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            manifest = rps.run(args)
            manifest_path = output_dir / "positive_sampling_manifest.json"
            on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertTrue(unknown.exists())
            self.assertEqual(on_disk["method"]["loop_order"], "x_outer_y_inner")
            self.assertFalse(on_disk["method"]["replace"])
            self.assertEqual(on_disk["parameters"]["random_state"], 1)
            self.assertEqual(on_disk["counts"]["positive_candidate_count"], 121)
            self.assertEqual(on_disk["counts"]["final_positive_sample_count"], 12)
            self.assertEqual(manifest["counts"], on_disk["counts"])
            for input_item in on_disk["inputs"].values():
                if input_item is not None:
                    self.assertTrue(input_item["sha256"])
            for output_item in on_disk["outputs"].values():
                self.assertTrue(output_item["sha256"])
                output_path = Path(output_item["path"])
                for filename, digest in output_item["sha256"].items():
                    self.assertEqual(
                        rps.file_sha256(output_path.parent / filename), digest
                    )

            rerun_args = rps.parse_args(
                [
                    "--study-area",
                    str(study_path),
                    "--known-polygons",
                    str(known_path),
                    "--tailings",
                    str(tailings_path),
                    "--output-dir",
                    str(output_dir),
                    "--overwrite",
                ]
            )
            rps.run(rerun_args)
            self.assertTrue(unknown.exists())

    @staticmethod
    def _frame(polygons: list[Polygon]) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(geometry=polygons, crs=CRS)

    @staticmethod
    def _candidate_frame(count: int) -> gpd.GeoDataFrame:
        frame = gpd.GeoDataFrame(
            {
                "cand_id": [f"C{index:08d}" for index in range(count)],
                "grid_ord": list(range(count)),
                "x_idx": list(range(count)),
                "y_idx": [0] * count,
                "Class": [1] * count,
            },
            geometry=gpd.points_from_xy(range(count), [0] * count),
            crs=CRS,
        )
        return frame


if __name__ == "__main__":
    unittest.main()
