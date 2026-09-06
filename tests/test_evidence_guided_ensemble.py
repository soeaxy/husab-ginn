from __future__ import annotations

import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from analyze_evidence_guided_ensemble import (
    merge_component_frames,
    rank_normalize,
    select_weights,
    simplex_weight_grid,
)


class EvidenceGuidedEnsembleTests(unittest.TestCase):
    def test_rank_normalize_maps_highest_score_to_one(self) -> None:
        values = np.array([0.2, 0.8, 0.5], dtype=float)
        normalized = rank_normalize(values)
        self.assertAlmostEqual(float(normalized[1]), 1.0, places=8)
        self.assertAlmostEqual(float(normalized[0]), 0.0, places=8)
        self.assertAlmostEqual(float(normalized[2]), 0.5, places=8)

    def test_simplex_weight_grid_covers_expected_half_step_simplex(self) -> None:
        grid = simplex_weight_grid(3, 0.5)
        self.assertEqual(len(grid), 6)
        self.assertIn((0.0, 0.5, 0.5), grid)
        self.assertIn((1.0, 0.0, 0.0), grid)
        for weights in grid:
            self.assertAlmostEqual(sum(weights), 1.0, places=8)

    def test_select_weights_breaks_ties_by_lower_geo_weight_then_simplicity(self) -> None:
        frame = pd.DataFrame(
            {
                "label": [1, 0, 1, 0],
                "rank_rf": [1.0, 0.0, 1.0, 0.0],
                "rank_gcn_transformer": [1.0, 0.0, 1.0, 0.0],
                "rank_pinn": [1.0, 0.0, 1.0, 0.0],
                "rank_geo_score": [1.0, 0.0, 1.0, 0.0],
            }
        )
        weights, metrics = select_weights(frame, ("rf", "gcn_transformer", "pinn", "geo_score"), 0.5)
        self.assertEqual(weights, (1.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(metrics["pr_auc"], 1.0, places=8)

    def test_merge_component_frames_rejects_coordinate_mismatch(self) -> None:
        rf = pd.DataFrame(
            {
                "sample_id": ["A"],
                "partition": ["validation"],
                "label": [1],
                "probability": [0.9],
                "prediction": [1],
                "x": [100.0],
                "y": [200.0],
                "block_id": [1],
                "row_index": [0],
            }
        )
        pinn = rf.copy()
        pinn["x"] = [101.0]
        geo = gpd.GeoDataFrame(
            {
                "sample_id": ["A"],
                "Class": [1],
                "geo_score": [0.7],
                "geometry": [Point(100.0, 200.0)],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        with self.assertRaisesRegex(ValueError, "x-coordinate mismatch"):
            merge_component_frames({"rf": rf, "pinn": pinn}, geo, "validation")


if __name__ == "__main__":
    unittest.main()
