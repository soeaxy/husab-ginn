from __future__ import annotations

import unittest

import numpy as np

from predict_reconstructed_pinn_ensemble import update_online_moments


class PinnEnsembleTests(unittest.TestCase):
    def test_online_moments_match_sample_statistics(self) -> None:
        members = np.array(
            [
                [0.1, 0.8, 0.4],
                [0.2, 0.7, 0.6],
                [0.3, 0.9, 0.5],
            ],
            dtype=np.float64,
        )
        mean = None
        m2 = None
        for count, member in enumerate(members, start=1):
            mean, m2 = update_online_moments(mean, m2, member, count)

        assert mean is not None and m2 is not None
        np.testing.assert_allclose(mean, members.mean(axis=0))
        np.testing.assert_allclose(np.sqrt(m2 / (members.shape[0] - 1)), members.std(axis=0, ddof=1))


if __name__ == "__main__":
    unittest.main()
