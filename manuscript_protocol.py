"""Machine-readable contract for the ORGEO-D-26-00764 experiments.

The archived experiment folders use ``pinn`` as the algorithm key.  The
manuscript calls the same implementation a geology-informed neural network
(GINN), because it uses heuristic geological priors rather than governing
physical equations.  Do not rename archived folders or result keys: use the
normalization and display helpers below at public interfaces.
"""

from __future__ import annotations

from typing import Final

MANUSCRIPT_ID: Final = "ORGEO-D-26-00764"
MANUSCRIPT_TITLE: Final = (
    "Geology-informed neural networks for uranium prospectivity mapping in "
    "the Rössing–Husab district, Namibia: a spatially blocked benchmark"
)

GINN_INTERNAL_KEY: Final = "pinn"
GINN_PUBLIC_KEY: Final = "ginn"
GINN_DISPLAY_NAME: Final = "GINN"
ALGORITHM_ALIASES: Final = {GINN_PUBLIC_KEY: GINN_INTERNAL_KEY}
MODEL_LABELS: Final = {
    "pinn": "GINN",
    "ginn": "GINN",
    "mlp": "MLP",
    "gcn_transformer": "GCN–Transformer",
    "rf": "RF",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}

BENCHMARK_ALGORITHMS: Final = (
    "pinn",
    "mlp",
    "gcn_transformer",
    "rf",
    "lightgbm",
    "xgboost",
    "catboost",
)
MODEL_SEED: Final = 2026
PSEUDO_ABSENCE_BASE_SEED: Final = 2025
PSEUDO_ABSENCE_SEED_STEP: Final = 1009
PSEUDO_ABSENCE_SEEDS: Final = tuple(
    PSEUDO_ABSENCE_BASE_SEED + PSEUDO_ABSENCE_SEED_STEP * index for index in range(10)
)

POSITIVE_GRID_SPACING_M: Final = 20.0
POSITIVE_SUBSAMPLE_FRACTION: Final = 0.10
POSITIVE_SUBSAMPLE_RANDOM_STATE: Final = 1
HUSAB_POSITIVE_CANDIDATE_COUNT: Final = 41_980
HUSAB_POSITIVE_SAMPLE_COUNT: Final = 4_198
HUSAB_PSEUDO_ABSENCE_COUNT: Final = 41_980

SPATIAL_GRID_SIZE: Final = 5
# Real spatial bounds are controlled data and are not distributed. This local
# Cartesian example has no geographic CRS and cannot reproduce the study area.
SYNTHETIC_EXAMPLE_BOUNDS: Final = (0.0, 0.0, 5000.0, 5000.0)
SPATIAL_PARTITIONS: Final = {
    "train": (6, 7, 10, 12, 14, 15, 16, 19, 20, 21, 23, 24),
    "validation": (0, 1, 11, 13),
    "test": (5, 17, 18, 22),
}

ACTIVE_GEOLOGICAL_PRIORS: Final = ("dome", "fault", "strata")
LEGACY_OPTIONAL_PRIORS: Final = ("singularity",)
PRIOR_KERNELS: Final = {
    "dome": {"kind": "gaussian", "width_m": 1000.0},
    "fault": {"kind": "exponential", "rate_per_m": 0.002},
    "strata": {"kind": "gaussian", "width_m": 800.0},
}
GINN_HIDDEN_DIMS: Final = (64, 32)
GINN_ACTIVE_PARAMETER_COUNT: Final = 2_789
GINN_REGISTERED_PARAMETER_COUNT: Final = 2_790
GINN_LR_SCHEDULER_PATIENCE: Final = 15
DEEP_BASELINE_LR_SCHEDULER_PATIENCE: Final = 10

GCN_TRANSFORMER_CONFIG: Final = {
    "neighborhood_size": 8,
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.1,
    "position_tokens": 9,
    "trainable_parameters": 76_610,
}

PRIMARY_BENCHMARK_RUNS: Final = 70
SUPPLEMENTARY_REAL_DATA_RUNS: Final = 76


def normalize_algorithm_key(value: str) -> str:
    """Return the stable internal key while accepting the public GINN name."""

    key = str(value).strip().lower()
    return ALGORITHM_ALIASES.get(key, key)


def manuscript_model_label(value: str) -> str:
    """Return the manuscript-facing model label for a code/result key."""

    key = str(value).strip().lower()
    return MODEL_LABELS.get(key, key.upper())


def scheduler_patience_for(value: str) -> int | None:
    """Return the fixed learning-rate scheduler patience used in archived runs."""

    key = normalize_algorithm_key(value)
    if key == GINN_INTERNAL_KEY:
        return GINN_LR_SCHEDULER_PATIENCE
    if key in {"mlp", "gcn_transformer", "gnn", "vae", "transunet"}:
        return DEEP_BASELINE_LR_SCHEDULER_PATIENCE
    return None
