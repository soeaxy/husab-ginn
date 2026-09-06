from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rebuild_negative_samples import SamplingConfig, compute_quotas
from run_ogr_sampling_design import ANALYSIS, OUTPUT, select_uniform_cells


class OGRSamplingDesignTests(unittest.TestCase):
    def test_uniform_sampling_is_deterministic_and_unique_per_thinning_cell(self) -> None:
        pool = pd.DataFrame({"thin_id": np.repeat(np.arange(50), 3), "stratum": ["hard"] * 150})
        first = select_uniform_cells(pool, 25)
        repeated = select_uniform_cells(pool, 25)
        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(pool.thin_id.iloc[first].nunique(), 25)
        self.assertFalse(np.array_equal(first, select_uniform_cells(pool, 25, seed=2027)))

    def test_alternative_quota_preserves_actual_training_negative_count(self) -> None:
        quotas = compute_quotas(27121, SamplingConfig(hard_fraction=0.5, transition_fraction=0.05, background_fraction=0.45))
        self.assertEqual(quotas, {"hard": 13560, "transition": 1356, "background": 12205})
        self.assertEqual(sum(quotas.values()), 27121)
        feasible = compute_quotas(27121, SamplingConfig(hard_fraction=0.65, transition_fraction=0.05, background_fraction=0.30))
        self.assertEqual(feasible, {"hard": 17629, "transition": 1356, "background": 8136})

    def test_outputs_do_not_overwrite_locked_experiment_axis(self) -> None:
        self.assertEqual(OUTPUT.name, "ogr_revision_sampling_design_20260905")
        self.assertEqual(ANALYSIS.name, "ogr_revision_sampling_design_20260905")


if __name__ == "__main__":
    unittest.main()
