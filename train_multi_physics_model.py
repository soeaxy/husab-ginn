# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyproj import Transformer
from rasterio.transform import rowcol
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from manuscript_protocol import (
    DEEP_BASELINE_LR_SCHEDULER_PATIENCE,
    GINN_DISPLAY_NAME,
    GINN_INTERNAL_KEY,
    GINN_LR_SCHEDULER_PATIENCE,
    GINN_PUBLIC_KEY,
    manuscript_model_label,
    normalize_algorithm_key,
    scheduler_patience_for,
)
from mineral_deep_models import (
    DEEP_COMPARE_ALGOS,
    augment_with_knn_graph_features,
    build_deep_compare_model,
    build_spatial_ego_graphs,
    resolve_torch_device,
)
from mineral_deep_models import (
    predict_probabilities_deep_compare as predict_proba_deep_compare,
)
from mineral_deep_models import (
    predict_probabilities_estimator as predict_proba_estimator,
)
from physics_informed_model import GeologyInformedClassifier

warnings.filterwarnings("ignore")
plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.unicode_minus": False,
    }
)
JOURNAL_COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "magenta": "#CC79A7",
    "gray": "#4D4D4D",
}


@dataclass
class DataConfig:
    shapefile_path: Path
    feature_dirs: list[Path]
    dome_tif_path: Path | None
    fault_tif_path: Path | None
    strata_tif_path: Path | None
    singularity_tif_path: Path | None
    label_col: str = "Class"


@dataclass
class TrainConfig:
    seed: int = 2025
    test_size: float = 0.2
    val_size: float = 0.2
    epochs: int = 800
    patience: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    physics_weight: float = 1.0
    grad_clip_norm: float = 2.0
    batch_size_cuda: int = 4096
    batch_size_cpu: int = 512
    num_workers: int = 0
    threshold: float = 0.5
    split_mode: str = "random"
    spatial_block_grid: int = 5
    split_seed: int | None = None
    split_manifest_path: Path | None = None
    selection_metric: str = "roc_auc"
    fixed_prior_weights: bool = False
    dome_width: float = 1000.0
    fault_rate: float = 0.002
    strata_width: float = 800.0
    torch_threads: int | None = None


@dataclass
class ExplainConfig:
    enable_shap: bool = True
    shap_background_size: int = 64
    shap_eval_size: int = 256
    shap_nsamples: int = 200


@dataclass
class AlgoConfig:
    algorithm: str = "pinn"
    optimizer: str = "none"
    bayes_trials: int = 25


CLASSICAL_ALGOS = {"rf", "lightgbm", "catboost", "xgboost"}
GEOLOGY_INFORMED_ALGOS = {GINN_INTERNAL_KEY}
PRIOR_NAMES = ("dome", "fault", "strata", "singularity")


@dataclass
class OutputConfig:
    output_root: Path
    model_name: str
    algorithm: str = "pinn"

    @property
    def model_dir(self) -> Path:
        return self.output_root / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.output_root / "metrics"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def explain_dir(self) -> Path:
        return self.output_root / "explainability"

    @property
    def model_path(self) -> Path:
        suffix = ".pth" if self.algorithm == GINN_INTERNAL_KEY or self.algorithm in DEEP_COMPARE_ALGOS else ".joblib"
        return self.model_dir / f"{self.model_name}{suffix}"

    @property
    def preproc_path(self) -> Path:
        return self.model_dir / "preproc.joblib"

    @property
    def train_history_path(self) -> Path:
        return self.metrics_dir / "train_history.csv"

    @property
    def metrics_path(self) -> Path:
        return self.metrics_dir / "metrics.json"

    @property
    def roc_curve_data_path(self) -> Path:
        return self.metrics_dir / "roc_curve_test.csv"

    @property
    def pr_curve_data_path(self) -> Path:
        return self.metrics_dir / "pr_curve_test.csv"

    @property
    def report_text_path(self) -> Path:
        return self.metrics_dir / "classification_report.txt"

    @property
    def model_params_path(self) -> Path:
        return self.metrics_dir / "model_parameters.json"

    @property
    def library_versions_path(self) -> Path:
        return self.metrics_dir / "library_versions.json"

    @property
    def config_path(self) -> Path:
        return self.metrics_dir / "run_config.json"

    @property
    def split_assignments_path(self) -> Path:
        return self.metrics_dir / "split_assignments.csv"

    @property
    def validation_predictions_path(self) -> Path:
        return self.metrics_dir / "validation_predictions.csv"

    @property
    def test_predictions_path(self) -> Path:
        return self.metrics_dir / "test_predictions.csv"

    @property
    def roc_fig_path(self) -> Path:
        return self.figures_dir / "roc_curve_test.png"

    @property
    def pr_fig_path(self) -> Path:
        return self.figures_dir / "pr_curve_test.png"

    @property
    def shap_csv_path(self) -> Path:
        return self.explain_dir / "shap_feature_importance.csv"

    @property
    def shap_plot_path(self) -> Path:
        return self.explain_dir / "shap_summary.png"

    @property
    def shap_values_path(self) -> Path:
        return self.explain_dir / "shap_values.npy"

    @property
    def shap_bar_path(self) -> Path:
        return self.explain_dir / "shap_bar.png"

    @property
    def loss_fig_path(self) -> Path:
        return self.figures_dir / "loss_curve.png"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(out_cfg: OutputConfig) -> None:
    for path in [out_cfg.output_root, out_cfg.model_dir, out_cfg.metrics_dir, out_cfg.figures_dir, out_cfg.explain_dir]:
        path.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Unsupported JSON type: {type(value)}")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)


def extract_learned_sigma(model: nn.Module) -> dict[str, float]:
    learned_sigma: dict[str, float] = {}
    for prior_name in PRIOR_NAMES:
        attr_name = f"log_sigma_{prior_name}"
        if hasattr(model, attr_name):
            sigma = torch.exp(getattr(model, attr_name)).detach().cpu().item()
            learned_sigma[prior_name] = float(sigma)
    return learned_sigma


def collect_library_versions() -> dict[str, Any]:
    libs = [
        "numpy",
        "pandas",
        "geopandas",
        "rasterio",
        "pyproj",
        "scikit-learn",
        "torch",
        "matplotlib",
        "joblib",
        "shap",
        "lightgbm",
        "catboost",
        "xgboost",
        "optuna",
    ]
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        versions["cuda_device_name"] = torch.cuda.get_device_name(0)

    for lib in libs:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "not_installed"
    return versions


def discover_tif_files(feature_dirs: list[Path]) -> list[Path]:
    tif_paths: list[Path] = []
    for folder in feature_dirs:
        if not folder.exists():
            logging.warning("Feature directory does not exist: %s", folder)
            continue
        tif_paths.extend(sorted(folder.glob("*.tif")))
    unique_paths = sorted({path.resolve() for path in tif_paths})
    if not unique_paths:
        raise FileNotFoundError("No .tif feature files found in --feature-dir")
    return unique_paths


def load_samples(data_cfg: DataConfig) -> gpd.GeoDataFrame:
    samples = gpd.read_file(data_cfg.shapefile_path)
    if data_cfg.label_col not in samples.columns:
        raise ValueError(f"Label column '{data_cfg.label_col}' not found in shapefile.")
    if samples.empty:
        raise ValueError("Shapefile has no rows.")
    if samples.geometry.is_empty.any():
        raise ValueError("Shapefile contains empty geometry.")
    samples[data_cfg.label_col] = samples[data_cfg.label_col].astype(int)
    return samples


def extract_features_at_points(samples: gpd.GeoDataFrame, tif_files: list[Path]) -> np.ndarray:
    xs = samples.geometry.x.to_numpy()
    ys = samples.geometry.y.to_numpy()
    features: list[np.ndarray] = []

    for tif in tif_files:
        with rasterio.open(tif) as src:
            if samples.crs != src.crs:
                transformer = Transformer.from_crs(samples.crs, src.crs, always_xy=True)
                x_r, y_r = transformer.transform(xs, ys)
            else:
                x_r, y_r = xs, ys

            rr, cc = rowcol(src.transform, x_r, y_r, op=np.floor)
            rr = rr.astype(int)
            cc = cc.astype(int)
            in_bounds = (rr >= 0) & (cc >= 0) & (rr < src.height) & (cc < src.width)

            band = src.read(1)
            values = np.full(xs.shape, np.nan, dtype=np.float32)
            values[in_bounds] = band[rr[in_bounds], cc[in_bounds]].astype(np.float32)
            if src.nodata is not None:
                values[np.isclose(values, src.nodata)] = np.nan
            features.append(values)

    return np.vstack(features).T


def clean_distance(dist_arr: np.ndarray, reference_indices: np.ndarray | None = None) -> np.ndarray:
    dist_arr = np.where(np.isfinite(dist_arr), dist_arr, np.nan)
    if np.all(~np.isfinite(dist_arr)):
        raise ValueError("Distance raster extraction is all NaN.")
    reference = dist_arr if reference_indices is None else dist_arr[np.asarray(reference_indices, dtype=np.int64)]
    if np.all(~np.isfinite(reference)):
        raise ValueError("Distance raster has no finite training values.")
    lo, hi = np.nanpercentile(reference, [1, 99])
    return np.clip(dist_arr, lo, hi)


def make_geological_prior(
    raw_values: np.ndarray,
    feature_type: str,
    *,
    dome_width: float = 1000.0,
    fault_rate: float = 0.002,
    strata_width: float = 800.0,
) -> np.ndarray:
    if any(not np.isfinite(value) or value <= 0 for value in (dome_width, fault_rate, strata_width)):
        raise ValueError("Geological prior widths and decay rate must be finite and positive.")
    valid_mask = np.isfinite(raw_values)
    d = raw_values.astype(np.float32, copy=True)
    d[~valid_mask] = np.nanmax(d[valid_mask]) if np.any(valid_mask) else 10000.0

    if feature_type == "dome":
        sigma = dome_width
        prior = np.exp(-((d**2) / (2 * sigma**2)))
    elif feature_type == "fault":
        decay_rate = fault_rate
        prior = np.exp(-decay_rate * d)
    elif feature_type == "strata":
        sigma = strata_width
        prior = np.exp(-((d**2) / (2 * sigma**2)))
    elif feature_type == "singularity":
        if not np.any(valid_mask):
            raise ValueError("Singularity raster extraction is all NaN.")
        lo, hi = np.nanpercentile(d[valid_mask], [1, 99])
        d = np.clip(d, lo, hi)
        if hi > lo:
            prior = 1.0 - (d - lo) / (hi - lo)
        else:
            prior = np.ones_like(d, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    prior = np.clip(prior, 0.0, 1.0).astype(np.float32)
    prior[~valid_mask] = 0.0
    return prior


def fit_preprocessor(
    X_raw: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, SimpleImputer, VarianceThreshold, StandardScaler, list[str]]:
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    var_selector = VarianceThreshold(threshold=1e-20)
    X_var = var_selector.fit_transform(X_imp)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_var)

    support = var_selector.get_support()
    kept_feature_names = list(np.asarray(feature_names)[support])
    return X_scaled.astype(np.float32), imputer, var_selector, scaler, kept_feature_names


def transform_features(
    X_raw: np.ndarray,
    imputer: SimpleImputer,
    var_selector: VarianceThreshold,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply a training-fitted preprocessing pipeline to held-out rows."""
    X_imp = imputer.transform(X_raw)
    X_var = var_selector.transform(X_imp)
    return scaler.transform(X_var).astype(np.float32)


def spatial_block_groups(
    coords: np.ndarray,
    grid_size: int,
    bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Assign projected coordinates to a fixed spatial-block grid."""
    if grid_size < 2:
        raise ValueError("spatial_block_grid must be at least 2.")
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        raise ValueError("coords must have shape [n_samples, 2] with at least one row.")
    if bounds is None:
        lower = coords.min(axis=0)
        upper = coords.max(axis=0)
    else:
        min_x, min_y, max_x, max_y = (float(value) for value in bounds)
        lower = np.array([min_x, min_y], dtype=np.float64)
        upper = np.array([max_x, max_y], dtype=np.float64)
        if np.any(upper <= lower):
            raise ValueError("Spatial split bounds must have positive width and height.")
        tolerance = np.finfo(np.float64).eps * np.maximum(1.0, np.abs(upper)) * 16
        if np.any(coords < lower - tolerance) or np.any(coords > upper + tolerance):
            raise ValueError("Sample coordinates fall outside the fixed spatial split bounds.")
    span = np.maximum(upper - lower, np.finfo(np.float64).eps)
    bins = np.floor((coords - lower) / span * grid_size).astype(int)
    bins = np.clip(bins, 0, grid_size - 1)
    return bins[:, 0] * grid_size + bins[:, 1]


def indices_from_block_partitions(
    groups: np.ndarray,
    labels: np.ndarray,
    partitions: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve fixed train/validation/test block assignments into row indices."""
    groups = np.asarray(groups, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if groups.ndim != 1 or labels.shape != groups.shape:
        raise ValueError("groups and labels must be one-dimensional arrays of equal length.")
    required = ("train", "validation", "test")
    if any(name not in partitions for name in required):
        raise ValueError("Block partitions must define train, validation, and test lists.")

    block_sets = {name: {int(value) for value in partitions[name]} for name in required}
    if any(not block_sets[name] for name in required):
        raise ValueError("Block partitions must assign every populated block and keep every partition non-empty.")
    if block_sets["train"] & block_sets["validation"] or block_sets["train"] & block_sets["test"] or block_sets["validation"] & block_sets["test"]:
        raise ValueError("Block partitions must be disjoint.")

    populated = {int(value) for value in np.unique(groups)}
    assigned = set().union(*block_sets.values())
    if not populated.issubset(assigned):
        raise ValueError("Block partitions must assign every populated block exactly once.")

    indices = {
        name: np.flatnonzero(np.isin(groups, sorted(block_sets[name])))
        for name in required
    }
    for name in required:
        if np.unique(labels[indices[name]]).size < 2:
            raise ValueError(f"Fixed spatial {name} partition must contain both classes.")
    return indices["train"], indices["validation"], indices["test"]


def load_spatial_split_manifest(
    path: Path,
) -> tuple[int, tuple[float, float, float, float], dict[str, list[int]], dict[str, Any]]:
    """Load and validate a fixed spatial-block split manifest."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("split_mode") != "spatial_block":
        raise ValueError("Split manifest must declare split_mode='spatial_block'.")
    grid_size = int(payload.get("grid_size", 0))
    raw_bounds = payload.get("bounds")
    if isinstance(raw_bounds, dict):
        bounds = (
            float(raw_bounds["min_x"]),
            float(raw_bounds["min_y"]),
            float(raw_bounds["max_x"]),
            float(raw_bounds["max_y"]),
        )
    elif isinstance(raw_bounds, list) and len(raw_bounds) == 4:
        bounds = tuple(float(value) for value in raw_bounds)
    else:
        raise ValueError("Split manifest bounds must be a four-value list or named mapping.")
    raw_partitions = payload.get("partitions")
    if not isinstance(raw_partitions, dict):
        raise ValueError("Split manifest must contain a partitions mapping.")
    partitions = {
        name: [int(value) for value in raw_partitions.get(name, [])]
        for name in ("train", "validation", "test")
    }
    return grid_size, bounds, partitions, payload


def grouped_holdout_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
    trials: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic spatial holdout with a feasible class balance."""
    target_n = max(1, int(round(indices.size * test_size)))
    target_pos = max(1, int(round(labels.sum() * test_size)))
    splitter = GroupShuffleSplit(n_splits=trials, test_size=test_size, random_state=seed)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for train_local, test_local in splitter.split(indices, labels, groups):
        y_train, y_test = labels[train_local], labels[test_local]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        score = abs(test_local.size - target_n) / target_n + abs(int(y_test.sum()) - target_pos) / target_pos
        if best is None or score < best[0]:
            best = (score, indices[train_local], indices[test_local])
    if best is None:
        raise ValueError("Could not create a spatial block split that retains both classes in every subset.")
    return best[1], best[2]


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def validation_selection_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str,
) -> tuple[float, float, float]:
    """Return the configured selection score plus both validation AUC diagnostics."""
    roc_auc = safe_roc_auc(y_true, y_prob)
    pr_auc = safe_pr_auc(y_true, y_prob)
    if metric == "roc_auc":
        score = roc_auc
    elif metric == "pr_auc":
        score = pr_auc
    else:
        raise ValueError(f"Unsupported validation selection metric: {metric}")
    return score, roc_auc, pr_auc


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "n_samples": int(y_true.size),
        "positive_ratio": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0),
    }

    if len(np.unique(y_true)) >= 2:
        clipped_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
        metrics["log_loss"] = float(log_loss(y_true, clipped_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["brier_score"] = None
        metrics["log_loss"] = None

    return metrics


def plot_loss_curve(history_df: pd.DataFrame, save_path: Path) -> None:
    if history_df.empty:
        return
    required_cols = {"epoch", "train_total_loss", "train_ce_loss", "train_physics_loss"}
    if not required_cols.issubset(set(history_df.columns)):
        logging.info("Skip loss curve: training history does not contain physics-model loss columns.")
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor="white")
    epochs = history_df["epoch"].values
    ax.plot(epochs, history_df["train_total_loss"], label="Total Loss", color=JOURNAL_COLORS["blue"], lw=2.2)
    ax.plot(epochs, history_df["train_ce_loss"], label="CE Loss", color=JOURNAL_COLORS["orange"], lw=1.8, linestyle="--")
    ax.plot(epochs, history_df["train_physics_loss"], label="Geological-prior loss", color=JOURNAL_COLORS["green"], lw=1.8, linestyle=":")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Curve", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6, color="#B7C1CC")
    ax.legend(frameon=False, fontsize=11)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path, data_path: Path | None = None) -> None:
    if len(np.unique(y_true)) < 2:
        logging.warning("Skip ROC plot: test set has only one class.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6.4, 5.3), facecolor="white")
    ax.plot(fpr, tpr, color=JOURNAL_COLORS["blue"], lw=2.4, label=f"Model (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], color=JOURNAL_COLORS["gray"], lw=1.4, linestyle="--", label="Random baseline")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Test ROC Curve", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6, color="#B7C1CC")
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    if data_path is not None:
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(data_path, index=False, encoding="utf-8-sig")
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path, data_path: Path | None = None) -> None:
    if len(np.unique(y_true)) < 2:
        logging.warning("Skip PR plot: test set has only one class.")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_value = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6.4, 5.3), facecolor="white")
    ax.plot(recall, precision, color=JOURNAL_COLORS["green"], lw=2.4, label=f"Model (AP = {ap_value:.4f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Test Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6, color="#B7C1CC")
    ax.legend(loc="lower left", frameon=False, fontsize=11)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    if data_path is not None:
        pd.DataFrame({"recall": recall, "precision": precision}).to_csv(data_path, index=False, encoding="utf-8-sig")
    plt.close()


def predict_proba(
    model: GeologyInformedClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_prob: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device, non_blocking=True)
            yb = batch[-1].detach().cpu().numpy()
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_prob.append(prob)
            all_y.append(yb)
    return np.concatenate(all_prob), np.concatenate(all_y)


def summarize_model_parameters(model: GeologyInformedClassifier) -> dict[str, Any]:
    parameter_stats: dict[str, Any] = {}
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        data = param.detach().cpu().float().numpy()
        numel = int(param.numel())
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
        parameter_stats[name] = {
            "shape": list(param.shape),
            "numel": numel,
            "requires_grad": bool(param.requires_grad),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
        }

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "learned_sigma": extract_learned_sigma(model),
        "per_parameter_stats": parameter_stats,
    }


def train_deep_compare_model(
    *,
    algorithm: str,
    train_cfg: TrainConfig,
    device: torch.device,
    X_tr: np.ndarray,
    pd_tr: np.ndarray,
    pf_tr: np.ndarray,
    ps_tr: np.ndarray,
    psi_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight_tensor: torch.Tensor,
    use_dome_prior: bool = True,
    use_fault_prior: bool = True,
    use_strata_prior: bool = True,
    use_singularity_prior: bool = True,
) -> tuple[nn.Module, pd.DataFrame, float, dict[str, Any]]:
    model, model_meta = build_deep_compare_model(algorithm, input_dim=int(X_tr.shape[-1]))
    model = model.to(device)
    batch_size = train_cfg.batch_size_cuda if device.type == "cuda" else train_cfg.batch_size_cpu
    pin_memory = device.type == "cuda"
    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(pd_tr, dtype=torch.float32),
        torch.tensor(pf_tr, dtype=torch.float32),
        torch.tensor(ps_tr, dtype=torch.float32),
        torch.tensor(psi_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=DEEP_BASELINE_LR_SCHEDULER_PATIENCE,
        min_lr=1e-6,
    )
    amp_enabled = device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    best_score = -np.inf
    bad_rounds = 0
    best_state: dict[str, torch.Tensor] | None = None
    history_rows: list[dict[str, Any]] = []
    algo = algorithm.lower()

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        seen = 0
        run_total = 0.0
        run_ce = 0.0
        run_phy = 0.0
        run_aux = 0.0

        for xb, p_dome_b, p_fault_b, p_strata_b, p_singularity_b, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            p_dome_b = p_dome_b.to(device, non_blocking=True) if use_dome_prior else None
            p_fault_b = p_fault_b.to(device, non_blocking=True) if use_fault_prior else None
            p_strata_b = p_strata_b.to(device, non_blocking=True) if use_strata_prior else None
            p_singularity_b = p_singularity_b.to(device, non_blocking=True) if use_singularity_prior else None
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                if algo == "vae":
                    logits, recon, mu, logvar = model(xb)
                    ce_loss = F.cross_entropy(logits, yb, weight=class_weight_tensor)
                    phy_loss = torch.tensor(0.0, device=device)
                    recon_loss = F.mse_loss(recon, xb)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    aux_loss = recon_loss + 0.1 * kl_loss
                    total_loss = ce_loss + 0.2 * aux_loss
                else:
                    logits = model(xb)
                    ce_loss = F.cross_entropy(logits, yb, weight=class_weight_tensor)
                    phy_loss = torch.tensor(0.0, device=device)
                    aux_loss = torch.tensor(0.0, device=device)
                    total_loss = ce_loss

            if not torch.isfinite(total_loss):
                continue
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            bsz = int(xb.size(0))
            seen += bsz
            run_total += float(total_loss.item()) * bsz
            run_ce += float(ce_loss.item()) * bsz
            run_phy += float(phy_loss.item()) * bsz
            run_aux += float(aux_loss.item()) * bsz

        avg_total = run_total / max(1, seen)
        avg_ce = run_ce / max(1, seen)
        avg_phy = run_phy / max(1, seen)
        avg_aux = run_aux / max(1, seen)
        val_prob = predict_proba_deep_compare(model, X_val, batch_size=batch_size, device=device, algorithm=algo)
        selection_score, val_roc_auc, val_pr_auc = validation_selection_score(
            y_val, val_prob, train_cfg.selection_metric
        )
        if np.isfinite(selection_score):
            scheduler.step(selection_score)
        if np.isfinite(selection_score) and selection_score > best_score + 1e-6:
            best_score = float(selection_score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_rounds = 0
        else:
            bad_rounds += 1

        history_rows.append(
            {
                "epoch": float(epoch),
                "train_total_loss": float(avg_total),
                "train_ce_loss": float(avg_ce),
                "train_physics_loss": float(avg_phy),
                "train_aux_loss": float(avg_aux),
                "val_auc": float(val_roc_auc) if np.isfinite(val_roc_auc) else np.nan,
                "val_roc_auc": float(val_roc_auc) if np.isfinite(val_roc_auc) else np.nan,
                "val_pr_auc": float(val_pr_auc) if np.isfinite(val_pr_auc) else np.nan,
                "val_selection_score": float(selection_score) if np.isfinite(selection_score) else np.nan,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if epoch == 1 or epoch % 10 == 0:
            logging.info(
                "[%s] Epoch %04d | loss=%.5f | ce=%.5f | phy=%.5f | aux=%.5f | "
                "val_roc_auc=%.5f | val_pr_auc=%.5f | select(%s)=%.5f",
                algo.upper(),
                epoch,
                avg_total,
                avg_ce,
                avg_phy,
                avg_aux,
                val_roc_auc,
                val_pr_auc,
                train_cfg.selection_metric,
                selection_score,
            )
        if bad_rounds >= train_cfg.patience:
            logging.info(
                "[%s] Early stopping at epoch %d (best validation %s=%.5f).",
                algo.upper(),
                epoch,
                train_cfg.selection_metric,
                best_score,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history_df = pd.DataFrame(history_rows)
    return model, history_df, best_score, model_meta


def summarize_torch_model_parameters(model: nn.Module, algorithm: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    parameter_stats: dict[str, Any] = {}
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        data = param.detach().cpu().float().numpy()
        numel = int(param.numel())
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
        parameter_stats[name] = {
            "shape": list(param.shape),
            "numel": numel,
            "requires_grad": bool(param.requires_grad),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
        }
    payload: dict[str, Any] = {
        "algorithm": algorithm,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "per_parameter_stats": parameter_stats,
    }
    learned_sigma = extract_learned_sigma(model)
    if learned_sigma:
        payload["learned_sigma"] = learned_sigma
    if extra:
        payload["extra"] = dict(extra)
    return payload


def build_classical_model(algorithm: str, seed: int, params: dict[str, Any]) -> Any:
    algo = algorithm.lower()
    if algo == "rf":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=params.get("max_depth", None),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            bootstrap=bool(params.get("bootstrap", True)),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if algo == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            raise ImportError("lightgbm is not installed. Please install lightgbm>=4.") from exc
        return LGBMClassifier(
            objective="binary",
            n_estimators=int(params.get("n_estimators", 500)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 63)),
            max_depth=int(params.get("max_depth", -1)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if algo == "catboost":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            raise ImportError("catboost is not installed. Please install catboost>=1.2.") from exc
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=int(params.get("iterations", 500)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            depth=int(params.get("depth", 7)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
            random_strength=float(params.get("random_strength", 1.0)),
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            auto_class_weights="Balanced",
            thread_count=-1,
        )
    if algo == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise ImportError("xgboost is not installed. Please install xgboost>=2.0.") from exc
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=int(params.get("n_estimators", 500)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 6)),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def suggest_classical_params(trial: Any, algorithm: str) -> dict[str, Any]:
    algo = algorithm.lower()
    if algo == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 900),
            "max_depth": trial.suggest_categorical("max_depth", [None, 6, 8, 10, 12, 16, 22]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    if algo == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
    if algo == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 150, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 8.0, log=True),
        }
    if algo == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 12.0),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def train_classical_model(
    *,
    algorithm: str,
    optimizer_name: str,
    bayes_trials: int,
    seed: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Any, pd.DataFrame, float, dict[str, Any]]:
    algo = algorithm.lower()
    optimizer_name = optimizer_name.lower()
    history_rows: list[dict[str, Any]] = []
    imbalance_params: dict[str, Any] = {}
    if algo == "xgboost":
        positives = max(1, int(np.sum(y_tr == 1)))
        negatives = int(np.sum(y_tr == 0))
        imbalance_params["scale_pos_weight"] = float(negatives / positives)
    best_params: dict[str, Any] = dict(imbalance_params)

    if optimizer_name == "bayes":
        try:
            import optuna
        except Exception as exc:
            raise ImportError("optuna is not installed. Please install optuna>=3.6.") from exc

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: Any) -> float:
            params = {**suggest_classical_params(trial, algo), **imbalance_params}
            model = build_classical_model(algo, seed, params)
            model.fit(X_tr, y_tr)
            val_prob = predict_proba_estimator(model, X_val)
            val_auc = safe_roc_auc(y_val, val_prob)
            score = float(val_auc) if np.isfinite(val_auc) else 0.0
            history_rows.append(
                {
                    "trial": int(trial.number),
                    "val_auc": score,
                    **{f"param_{k}": v for k, v in params.items()},
                }
            )
            return score

        logging.info("Bayesian optimization started: algorithm=%s, trials=%d", algo, bayes_trials)
        study.optimize(objective, n_trials=bayes_trials, show_progress_bar=False)
        best_params = {**dict(study.best_params), **imbalance_params}
        best_val_auc = float(study.best_value) if study.best_trial is not None else float("nan")
    else:
        model = build_classical_model(algo, seed, imbalance_params)
        model.fit(X_tr, y_tr)
        val_prob = predict_proba_estimator(model, X_val)
        best_val_auc = safe_roc_auc(y_val, val_prob)
        history_rows.append({"trial": 0, "val_auc": float(best_val_auc) if np.isfinite(best_val_auc) else np.nan})
        best_params = dict(imbalance_params)

    model = build_classical_model(algo, seed, best_params)
    model.fit(X_tr, y_tr)
    history_df = pd.DataFrame(history_rows)
    return model, history_df, best_val_auc, best_params


def summarize_estimator_parameters(model: Any, algorithm: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "algorithm": algorithm,
        "model_class": type(model).__name__,
    }
    if hasattr(model, "get_params"):
        payload["params"] = model.get_params(deep=False)
    if hasattr(model, "n_features_in_"):
        payload["n_features_in"] = int(model.n_features_in_)
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=np.float64)
        payload["feature_importances"] = importance.tolist()
    return payload


def train_model(
    model: GeologyInformedClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weight_tensor: torch.Tensor,
    train_cfg: TrainConfig,
    device: torch.device,
    use_dome_prior: bool = True,
    use_fault_prior: bool = True,
    use_strata_prior: bool = True,
    use_singularity_prior: bool = True,
) -> tuple[GeologyInformedClassifier, pd.DataFrame, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=GINN_LR_SCHEDULER_PATIENCE,
        min_lr=1e-6,
    )
    amp_enabled = device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    best_score = -np.inf
    bad_rounds = 0
    best_state: dict[str, torch.Tensor] | None = None
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running_total = 0.0
        running_ce = 0.0
        running_phy = 0.0
        seen = 0

        for xb, p_dome_b, p_fault_b, p_strata_b, p_singularity_b, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            p_dome_b = p_dome_b.to(device, non_blocking=True) if use_dome_prior else None
            p_fault_b = p_fault_b.to(device, non_blocking=True) if use_fault_prior else None
            p_strata_b = p_strata_b.to(device, non_blocking=True) if use_strata_prior else None
            p_singularity_b = p_singularity_b.to(device, non_blocking=True) if use_singularity_prior else None
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                logits = model(xb)
                ce_loss = F.cross_entropy(logits, yb, weight=class_weight_tensor)
                phy_loss = model.geological_prior_loss(
                    logits,
                    p_dome_b,
                    p_fault_b,
                    p_strata_b,
                    p_singularity_b,
                )
                total_loss = ce_loss + train_cfg.physics_weight * phy_loss

            if not torch.isfinite(total_loss):
                continue

            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            batch_size = xb.size(0)
            running_total += total_loss.item() * batch_size
            running_ce += ce_loss.item() * batch_size
            running_phy += phy_loss.item() * batch_size
            seen += batch_size

        avg_total_loss = running_total / max(1, seen)
        avg_ce_loss = running_ce / max(1, seen)
        avg_phy_loss = running_phy / max(1, seen)

        val_prob, val_true = predict_proba(model, val_loader, device)
        selection_score, val_roc_auc, val_pr_auc = validation_selection_score(
            val_true, val_prob, train_cfg.selection_metric
        )

        if np.isfinite(selection_score):
            scheduler.step(selection_score)

        if np.isfinite(selection_score) and selection_score > best_score + 1e-6:
            best_score = float(selection_score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_rounds = 0
        else:
            bad_rounds += 1

        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": float(epoch),
                "train_total_loss": float(avg_total_loss),
                "train_ce_loss": float(avg_ce_loss),
                "train_physics_loss": float(avg_phy_loss),
                "train_geology_loss": float(avg_phy_loss),
                "val_auc": float(val_roc_auc) if np.isfinite(val_roc_auc) else np.nan,
                "val_roc_auc": float(val_roc_auc) if np.isfinite(val_roc_auc) else np.nan,
                "val_pr_auc": float(val_pr_auc) if np.isfinite(val_pr_auc) else np.nan,
                "val_selection_score": float(selection_score) if np.isfinite(selection_score) else np.nan,
                "lr": current_lr,
            }
        )

        if epoch == 1 or epoch % 10 == 0:
            logging.info(
                "Epoch %04d | loss=%.5f | ce=%.5f | phy=%.5f | val_roc_auc=%.5f | "
                "val_pr_auc=%.5f | select(%s)=%.5f | sigma=%s",
                epoch,
                avg_total_loss,
                avg_ce_loss,
                avg_phy_loss,
                val_roc_auc,
                val_pr_auc,
                train_cfg.selection_metric,
                selection_score,
                ", ".join(f"{k}:{v:.3f}" for k, v in extract_learned_sigma(model).items()),
            )

        if bad_rounds >= train_cfg.patience:
            logging.info(
                "Early stopping at epoch %d (best validation %s=%.5f).",
                epoch,
                train_cfg.selection_metric,
                best_score,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history_df = pd.DataFrame(history_rows)
    return model, history_df, best_score


def compute_shap_importance(
    model: Any,
    algorithm: str,
    X_background: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list[str],
    explain_cfg: ExplainConfig,
    out_cfg: OutputConfig,
    device: torch.device | None = None,
) -> pd.DataFrame | None:
    if not explain_cfg.enable_shap:
        logging.info("SHAP disabled by configuration.")
        return None

    try:
        import shap
    except ImportError:
        logging.warning("SHAP is not installed. Skip SHAP analysis.")
        return None

    if X_background.size == 0 or X_eval.size == 0:
        logging.warning("Empty arrays for SHAP. Skip SHAP analysis.")
        return None

    background = X_background[: explain_cfg.shap_background_size]
    eval_data = X_eval[: explain_cfg.shap_eval_size]
    if background.shape[0] < 2 or eval_data.shape[0] < 1:
        logging.warning("Not enough samples for SHAP. Skip SHAP analysis.")
        return None

    algo = algorithm.lower()

    def predict_class1(x: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x, dtype=np.float32)
        if algo == GINN_INTERNAL_KEY or algo in DEEP_COMPARE_ALGOS:
            if device is None:
                raise ValueError("device is required for torch-model SHAP.")
            if algo == GINN_INTERNAL_KEY:
                model.eval()
                x_tensor = torch.as_tensor(x_np, device=device)
                with torch.no_grad():
                    logits = model(x_tensor)
                    probs = torch.softmax(logits, dim=1)[:, 1]
                return probs.detach().cpu().numpy()
            return predict_proba_deep_compare(model, x_np, batch_size=8192, device=device, algorithm=algo)
        return predict_proba_estimator(model, x_np)

    logging.info(
        "Running SHAP analysis: algorithm=%s, background=%d, eval=%d, nsamples=%d",
        algo,
        background.shape[0],
        eval_data.shape[0],
        explain_cfg.shap_nsamples,
    )
    if algo in CLASSICAL_ALGOS:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(eval_data)
    else:
        explainer = shap.KernelExplainer(predict_class1, background)
        shap_values = explainer.shap_values(eval_data, nsamples=explain_cfg.shap_nsamples)

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0], dtype=np.float64)
    else:
        values = np.asarray(shap_values, dtype=np.float64)

    if values.ndim == 3:
        # Support both (n_samples, n_features, n_classes) and (n_classes, n_samples, n_features)
        if values.shape[0] in {1, 2} and values.shape[1] == eval_data.shape[0]:
            values = values[1 if values.shape[0] > 1 else 0, :, :]
        elif values.shape[2] in {1, 2} and values.shape[0] == eval_data.shape[0]:
            values = values[:, :, 1 if values.shape[2] > 1 else 0]

    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[0] != eval_data.shape[0]:
        values = values.reshape(eval_data.shape[0], -1)
    if values.shape[1] != len(feature_names):
        raise ValueError("SHAP values shape does not match feature count.")

    np.save(out_cfg.shap_values_path, values)
    importance = np.mean(np.abs(values), axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df.to_csv(out_cfg.shap_csv_path, index=False, encoding="utf-8-sig")

    shap.summary_plot(values, eval_data, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_cfg.shap_plot_path, dpi=600, bbox_inches="tight")
    plt.close()

    # Generate SHAP Bar Plot for Global Interpretability
    shap.summary_plot(values, eval_data, feature_names=feature_names, show=False, max_display=20, plot_type="bar")
    plt.tight_layout()
    plt.savefig(out_cfg.shap_bar_path, dpi=600, bbox_inches="tight")
    plt.close()

    logging.info("SHAP artifacts saved to %s", out_cfg.explain_dir)
    return importance_df


def positive_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Value must be finite and positive.")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return number


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train a geology-informed mineral prospectivity model and comparison baselines."
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=project_root / "data" / "combined_samples.shp",
    )
    parser.add_argument(
        "--feature-dir",
        dest="feature_dirs",
        action="append",
        type=Path,
        default=None,
        help="Can be passed multiple times.",
    )
    parser.add_argument(
        "--dome-tif",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--fault-tif",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--strata-tif",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--singularity-tif",
        type=Path,
        default=None,
    )
    parser.add_argument("--label-col", type=str, default="Class")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--model-name", type=str, default="mineral_model_multi_phy_final")
    parser.add_argument(
        "--algorithm",
        type=str,
        default=GINN_PUBLIC_KEY,
        choices=[GINN_PUBLIC_KEY, GINN_INTERNAL_KEY, "rf", "lightgbm", "catboost", "xgboost", "mlp", "gnn", "vae", "transunet", "gcn_transformer"],
        help="Use 'ginn' for the manuscript model; legacy key 'pinn' remains accepted for archived workflows.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="none",
        choices=["none", "bayes"],
    )
    parser.add_argument("--bayes-trials", type=int, default=25)

    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Optional split-only seed. Defaults to --seed when omitted.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--geology-weight",
        "--physics-weight",
        dest="physics_weight",
        type=float,
        default=1.0,
        help="Weight applied to the geological-prior loss; --physics-weight is a legacy alias.",
    )
    parser.add_argument("--fixed-prior-weights", action="store_true", help="Freeze log_sigma at zero (equal 0.5 MSE coefficients).")
    parser.add_argument("--dome-width", type=positive_float, default=1000.0)
    parser.add_argument("--fault-rate", type=positive_float, default=0.002)
    parser.add_argument("--strata-width", type=positive_float, default=800.0)
    parser.add_argument("--torch-threads", type=positive_int, default=None, help="Optional PyTorch CPU thread limit; omitted preserves the runtime default.")
    parser.add_argument("--grad-clip-norm", type=float, default=2.0)
    parser.add_argument("--batch-size-cuda", type=int, default=4096)
    parser.add_argument("--batch-size-cpu", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--selection-metric",
        choices=["roc_auc", "pr_auc"],
        default="roc_auc",
        help="Validation metric used for deep-model scheduling, early stopping, and checkpoint selection.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "spatial_block"],
        default="random",
        help="Random stratified split or spatially disjoint block split.",
    )
    parser.add_argument(
        "--spatial-block-grid",
        type=int,
        default=5,
        help="Number of blocks per map axis when --split-mode spatial_block is selected.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest that fixes spatial bounds and block assignments across runs.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--disable-shap", action="store_true")
    parser.add_argument("--shap-background-size", type=int, default=64)
    parser.add_argument("--shap-eval-size", type=int, default=256)
    parser.add_argument("--shap-nsamples", type=int, default=200)
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    device = resolve_torch_device(args.device)
    logging.info("Using device: %s", device)

    feature_dirs = args.feature_dirs or [Path(__file__).resolve().parent / "data" / "factors"]

    data_cfg = DataConfig(
        shapefile_path=args.shapefile,
        feature_dirs=feature_dirs,
        dome_tif_path=args.dome_tif,
        fault_tif_path=args.fault_tif,
        strata_tif_path=args.strata_tif,
        singularity_tif_path=args.singularity_tif,
        label_col=args.label_col,
    )
    train_cfg = TrainConfig(
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        physics_weight=args.physics_weight,
        grad_clip_norm=args.grad_clip_norm,
        batch_size_cuda=args.batch_size_cuda,
        batch_size_cpu=args.batch_size_cpu,
        num_workers=args.num_workers,
        threshold=args.threshold,
        split_mode=args.split_mode,
        spatial_block_grid=args.spatial_block_grid,
        split_seed=args.split_seed,
        split_manifest_path=args.split_manifest,
        selection_metric=args.selection_metric,
        fixed_prior_weights=args.fixed_prior_weights,
        dome_width=args.dome_width,
        fault_rate=args.fault_rate,
        strata_width=args.strata_width,
        torch_threads=args.torch_threads,
    )
    explain_cfg = ExplainConfig(
        enable_shap=(not args.disable_shap),
        shap_background_size=args.shap_background_size,
        shap_eval_size=args.shap_eval_size,
        shap_nsamples=args.shap_nsamples,
    )
    requested_algorithm = args.algorithm
    algo_cfg = AlgoConfig(
        algorithm=normalize_algorithm_key(requested_algorithm),
        optimizer=args.optimizer,
        bayes_trials=max(1, int(args.bayes_trials)),
    )
    out_cfg = OutputConfig(output_root=args.output_root, model_name=args.model_name, algorithm=algo_cfg.algorithm)
    ensure_dirs(out_cfg)

    set_global_seed(train_cfg.seed)

    save_json(
        out_cfg.config_path,
        {
            "data_config": data_cfg.__dict__,
            "enabled_priors": {
                "dome": data_cfg.dome_tif_path is not None,
                "fault": data_cfg.fault_tif_path is not None,
                "strata": data_cfg.strata_tif_path is not None,
                "singularity": data_cfg.singularity_tif_path is not None,
            },
            "train_config": train_cfg.__dict__,
            "algo_config": algo_cfg.__dict__,
            "explain_config": explain_cfg.__dict__,
            "output_config": out_cfg.__dict__,
            "device": str(device),
            "manuscript_context": {
                "model_label": manuscript_model_label(requested_algorithm),
                "requested_algorithm": requested_algorithm,
                "internal_algorithm_key": algo_cfg.algorithm,
                "legacy_pinn_key_retained": algo_cfg.algorithm == GINN_INTERNAL_KEY,
                "lr_scheduler_patience": scheduler_patience_for(algo_cfg.algorithm),
            },
        },
    )
    save_json(out_cfg.library_versions_path, collect_library_versions())

    samples = load_samples(data_cfg)
    feature_tifs = discover_tif_files(data_cfg.feature_dirs)
    feature_names = [path.stem for path in feature_tifs]
    logging.info("Loaded %d samples and %d feature rasters.", samples.shape[0], len(feature_tifs))

    X_feature_raw = extract_features_at_points(samples, feature_tifs)
    X_raw = np.where(np.isfinite(X_feature_raw), X_feature_raw, np.nan).astype(np.float32)
    y = samples[data_cfg.label_col].to_numpy().astype(np.int64)
    coords = np.column_stack([samples.geometry.x.to_numpy(), samples.geometry.y.to_numpy()]).astype(np.float64)

    n_samples = y.shape[0]
    all_indices = np.arange(n_samples)
    split_seed = train_cfg.seed if train_cfg.split_seed is None else int(train_cfg.split_seed)
    all_groups = np.full(n_samples, -1, dtype=np.int64)
    split_metadata: dict[str, Any] = {
        "mode": train_cfg.split_mode,
        "split_seed": split_seed,
    }
    if train_cfg.split_manifest_path is not None and train_cfg.split_mode != "spatial_block":
        raise ValueError("--split-manifest can only be used with --split-mode spatial_block.")

    if train_cfg.split_mode == "random":
        train_indices, test_indices = train_test_split(
            all_indices, test_size=train_cfg.test_size, random_state=split_seed, stratify=y
        )
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=train_cfg.val_size,
            random_state=split_seed + 1,
            stratify=y[train_indices],
        )
    else:
        if train_cfg.split_manifest_path is not None:
            manifest_grid, manifest_bounds, block_partitions, manifest_payload = load_spatial_split_manifest(
                train_cfg.split_manifest_path
            )
            if manifest_grid != train_cfg.spatial_block_grid:
                raise ValueError(
                    "--spatial-block-grid does not match the fixed split manifest "
                    f"({train_cfg.spatial_block_grid} != {manifest_grid})."
                )
            all_groups = spatial_block_groups(coords, manifest_grid, bounds=manifest_bounds)
            train_indices, val_indices, test_indices = indices_from_block_partitions(
                all_groups, y, block_partitions
            )
            split_metadata.update(
                {
                    "manifest_path": str(train_cfg.split_manifest_path.resolve()),
                    "manifest": manifest_payload,
                }
            )
        else:
            all_groups = spatial_block_groups(coords, train_cfg.spatial_block_grid)
            train_indices, test_indices = grouped_holdout_indices(
                all_indices, y, all_groups, train_cfg.test_size, split_seed
            )
            train_indices, val_indices = grouped_holdout_indices(
                train_indices,
                y[train_indices],
                all_groups[train_indices],
                train_cfg.val_size,
                split_seed + 1,
            )
            split_metadata["generated_bounds"] = [
                float(coords[:, 0].min()),
                float(coords[:, 1].min()),
                float(coords[:, 0].max()),
                float(coords[:, 1].max()),
            ]
            split_metadata["partitions"] = {
                "train": sorted(int(value) for value in np.unique(all_groups[train_indices])),
                "validation": sorted(int(value) for value in np.unique(all_groups[val_indices])),
                "test": sorted(int(value) for value in np.unique(all_groups[test_indices])),
            }
        logging.info(
            "Spatial block split: grid=%dx%d | train=%d | val=%d | test=%d",
            train_cfg.spatial_block_grid,
            train_cfg.spatial_block_grid,
            train_indices.size,
            val_indices.size,
            test_indices.size,
        )

    X_tr, imputer, var_selector, scaler, kept_feature_names = fit_preprocessor(
        X_raw[train_indices], feature_names
    )
    X_val = transform_features(X_raw[val_indices], imputer, var_selector, scaler)
    X_test = transform_features(X_raw[test_indices], imputer, var_selector, scaler)
    logging.info(
        "Training-only preprocessing done: %d -> %d features after variance filtering.",
        len(feature_names),
        len(kept_feature_names),
    )

    use_dome_prior = data_cfg.dome_tif_path is not None
    use_fault_prior = data_cfg.fault_tif_path is not None
    use_strata_prior = data_cfg.strata_tif_path is not None
    use_singularity_prior = data_cfg.singularity_tif_path is not None
    if algo_cfg.algorithm in GEOLOGY_INFORMED_ALGOS and not (
        use_dome_prior or use_fault_prior or use_strata_prior or use_singularity_prior
    ):
        raise ValueError(
            f"{GINN_DISPLAY_NAME} requires at least one prior tif (dome/fault/strata; singularity is legacy optional input)."
        )
    logging.info("Selected algorithm: %s | optimizer: %s", algo_cfg.algorithm, algo_cfg.optimizer)
    if algo_cfg.algorithm == GINN_INTERNAL_KEY and algo_cfg.optimizer == "bayes":
        logging.warning(
            "Bayesian optimizer is currently applied to classical models only; GINN uses standard training."
        )
    logging.info(
        "Prior files present: dome=%s, fault=%s, strata=%s, singularity=%s",
        use_dome_prior,
        use_fault_prior,
        use_strata_prior,
        use_singularity_prior,
    )
    p_dome = np.zeros((n_samples, 1), dtype=np.float32)
    p_fault = np.zeros((n_samples, 1), dtype=np.float32)
    p_strata = np.zeros((n_samples, 1), dtype=np.float32)
    p_singularity = np.zeros((n_samples, 1), dtype=np.float32)
    if algo_cfg.algorithm in GEOLOGY_INFORMED_ALGOS and use_dome_prior:
        dome_distance = clean_distance(
            extract_features_at_points(samples, [data_cfg.dome_tif_path])[:, 0], train_indices
        )
        p_dome = make_geological_prior(dome_distance, "dome", dome_width=train_cfg.dome_width).reshape(-1, 1)
    if algo_cfg.algorithm in GEOLOGY_INFORMED_ALGOS and use_fault_prior:
        fault_distance = clean_distance(
            extract_features_at_points(samples, [data_cfg.fault_tif_path])[:, 0], train_indices
        )
        p_fault = make_geological_prior(fault_distance, "fault", fault_rate=train_cfg.fault_rate).reshape(-1, 1)
    if algo_cfg.algorithm in GEOLOGY_INFORMED_ALGOS and use_strata_prior:
        strata_distance = clean_distance(
            extract_features_at_points(samples, [data_cfg.strata_tif_path])[:, 0], train_indices
        )
        p_strata = make_geological_prior(strata_distance, "strata", strata_width=train_cfg.strata_width).reshape(-1, 1)
    if algo_cfg.algorithm in GEOLOGY_INFORMED_ALGOS and use_singularity_prior:
        singularity_values = extract_features_at_points(samples, [data_cfg.singularity_tif_path])[:, 0]
        p_singularity = make_geological_prior(singularity_values, "singularity").reshape(-1, 1)

    y_tr, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
    pd_tr, pd_val = p_dome[train_indices], p_dome[val_indices]
    pf_tr, pf_val = p_fault[train_indices], p_fault[val_indices]
    ps_tr, ps_val = p_strata[train_indices], p_strata[val_indices]
    psi_tr, psi_val = p_singularity[train_indices], p_singularity[val_indices]
    coords_tr, coords_val, coords_test = coords[train_indices], coords[val_indices], coords[test_indices]

    split_labels = np.empty(n_samples, dtype=object)
    split_labels[train_indices] = "train"
    split_labels[val_indices] = "validation"
    split_labels[test_indices] = "test"
    sample_keys = (
        samples["sample_id"].astype(str).to_numpy()
        if "sample_id" in samples.columns
        else np.asarray([str(value) for value in all_indices])
    )
    pd.DataFrame(
        {
            "row_index": all_indices,
            "sample_id": sample_keys,
            "partition": split_labels,
            "block_id": all_groups,
            "label": y,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    ).to_csv(out_cfg.split_assignments_path, index=False, encoding="utf-8-sig")
    split_metadata["counts"] = {
        "train": int(train_indices.size),
        "validation": int(val_indices.size),
        "test": int(test_indices.size),
    }
    split_metadata["positive_counts"] = {
        "train": int(y_tr.sum()),
        "validation": int(y_val.sum()),
        "test": int(y_test.sum()),
    }
    run_config_payload = json.loads(out_cfg.config_path.read_text(encoding="utf-8"))
    run_config_payload["split_protocol"] = split_metadata
    run_config_payload["preprocessing_fit_partition"] = "train"
    save_json(out_cfg.config_path, run_config_payload)

    best_params: dict[str, Any] = {}
    preproc_extra: dict[str, Any] = {}
    shap_X_background = X_tr
    shap_X_eval = X_test
    shap_feature_names = kept_feature_names
    if algo_cfg.algorithm == GINN_INTERNAL_KEY:
        batch_size = train_cfg.batch_size_cuda if device.type == "cuda" else train_cfg.batch_size_cpu
        pin_memory = device.type == "cuda"
        train_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(pd_tr, dtype=torch.float32),
            torch.tensor(pf_tr, dtype=torch.float32),
            torch.tensor(ps_tr, dtype=torch.float32),
            torch.tensor(psi_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(pd_val, dtype=torch.float32),
            torch.tensor(pf_val, dtype=torch.float32),
            torch.tensor(ps_val, dtype=torch.float32),
            torch.tensor(psi_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=train_cfg.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=pin_memory,
        )

        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_tr), y=y_tr)
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

        model = GeologyInformedClassifier(
            input_dim=X_tr.shape[1], fixed_prior_weights=train_cfg.fixed_prior_weights
        ).to(device)
        model, history_df, best_val_auc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weight_tensor=class_weight_tensor,
            train_cfg=train_cfg,
            device=device,
            use_dome_prior=use_dome_prior,
            use_fault_prior=use_fault_prior,
            use_strata_prior=use_strata_prior,
            use_singularity_prior=use_singularity_prior,
        )

        torch.save(model.state_dict(), out_cfg.model_path)
        val_prob, val_true = predict_proba(model, val_loader, device)
        test_prob, test_true = predict_proba(model, test_loader, device)
        model_param_payload = summarize_model_parameters(model)
    elif algo_cfg.algorithm in CLASSICAL_ALGOS:
        if algo_cfg.optimizer == "bayes":
            model, history_df, best_val_auc, best_params = train_classical_model(
                algorithm=algo_cfg.algorithm,
                optimizer_name=algo_cfg.optimizer,
                bayes_trials=algo_cfg.bayes_trials,
                seed=train_cfg.seed,
                X_tr=X_tr,
                y_tr=y_tr,
                X_val=X_val,
                y_val=y_val,
            )
        else:
            model, history_df, best_val_auc, best_params = train_classical_model(
                algorithm=algo_cfg.algorithm,
                optimizer_name="none",
                bayes_trials=1,
                seed=train_cfg.seed,
                X_tr=X_tr,
                y_tr=y_tr,
                X_val=X_val,
                y_val=y_val,
            )
        joblib.dump(model, out_cfg.model_path)
        val_prob = predict_proba_estimator(model, X_val)
        val_true = y_val
        test_prob = predict_proba_estimator(model, X_test)
        test_true = y_test
        model_param_payload = summarize_estimator_parameters(model, algo_cfg.algorithm)
    elif algo_cfg.algorithm in DEEP_COMPARE_ALGOS:
        if algo_cfg.optimizer == "bayes":
            logging.warning("Bayesian optimizer is currently applied to classical models only; %s uses standard training.", algo_cfg.algorithm)

        X_tr_work = X_tr
        X_val_work = X_val
        X_test_work = X_test
        if algo_cfg.algorithm == "gnn":
            gnn_k = 8
            X_tr_work = augment_with_knn_graph_features(X_tr, X_tr, k_neighbors=gnn_k)
            X_val_work = augment_with_knn_graph_features(X_tr, X_val, k_neighbors=gnn_k)
            X_test_work = augment_with_knn_graph_features(X_tr, X_test, k_neighbors=gnn_k)
            preproc_extra["gnn_ref_features"] = X_tr.astype(np.float32)
            preproc_extra["gnn_k"] = int(gnn_k)
            shap_feature_names = list(kept_feature_names) + [f"{n}_邻域均值" for n in kept_feature_names]
        elif algo_cfg.algorithm == "gcn_transformer":
            graph_k = 8
            X_tr_work = build_spatial_ego_graphs(
                X_tr, coords_tr, X_tr, coords_tr, k_neighbors=graph_k, exclude_self=True
            )
            X_val_work = build_spatial_ego_graphs(X_tr, coords_tr, X_val, coords_val, k_neighbors=graph_k)
            X_test_work = build_spatial_ego_graphs(X_tr, coords_tr, X_test, coords_test, k_neighbors=graph_k)
            preproc_extra["gcn_transformer_ref_features"] = X_tr.astype(np.float32)
            preproc_extra["gcn_transformer_ref_coords"] = coords_tr.astype(np.float64)
            preproc_extra["gcn_transformer_k"] = int(graph_k)

        deep_class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_tr), y=y_tr)
        deep_class_weight_tensor = torch.tensor(deep_class_weights, dtype=torch.float32, device=device)
        model, history_df, best_val_auc, model_meta = train_deep_compare_model(
            algorithm=algo_cfg.algorithm,
            train_cfg=train_cfg,
            device=device,
            X_tr=X_tr_work,
            pd_tr=pd_tr,
            pf_tr=pf_tr,
            ps_tr=ps_tr,
            psi_tr=psi_tr,
            y_tr=y_tr,
            X_val=X_val_work,
            y_val=y_val,
            class_weight_tensor=deep_class_weight_tensor,
            use_dome_prior=False,
            use_fault_prior=False,
            use_strata_prior=False,
            use_singularity_prior=False,
        )
        torch.save(
            {
                "algorithm": algo_cfg.algorithm,
                "model_state": model.state_dict(),
                "model_meta": model_meta,
            },
            out_cfg.model_path,
        )
        batch_size_eval = train_cfg.batch_size_cuda if device.type == "cuda" else train_cfg.batch_size_cpu
        val_prob = predict_proba_deep_compare(model, X_val_work, batch_size_eval, device, algo_cfg.algorithm)
        val_true = y_val
        test_prob = predict_proba_deep_compare(model, X_test_work, batch_size_eval, device, algo_cfg.algorithm)
        test_true = y_test
        model_param_payload = summarize_torch_model_parameters(model, algo_cfg.algorithm, extra=model_meta)
        shap_X_background = X_tr_work
        shap_X_eval = X_test_work
    else:
        raise ValueError(f"Unsupported algorithm: {algo_cfg.algorithm}")

    history_df.to_csv(out_cfg.train_history_path, index=False, encoding="utf-8-sig")
    plot_loss_curve(history_df, out_cfg.loss_fig_path)
    logging.info("Model saved: %s", out_cfg.model_path)

    joblib.dump(
        {
            "imputer": imputer,
            "variance_selector": var_selector,
            "scaler": scaler,
            "all_feature_names": feature_names,
            "kept_feature_names": kept_feature_names,
            "algorithm": algo_cfg.algorithm,
            "manuscript_model_label": manuscript_model_label(algo_cfg.algorithm),
            **preproc_extra,
        },
        out_cfg.preproc_path,
    )
    logging.info("Preprocessor saved: %s", out_cfg.preproc_path)

    val_metrics = compute_metrics(val_true, val_prob, threshold=train_cfg.threshold)
    test_metrics = compute_metrics(test_true, test_prob, threshold=train_cfg.threshold)
    for output_path, subset_indices, probabilities, truth, partition_name in (
        (out_cfg.validation_predictions_path, val_indices, val_prob, val_true, "validation"),
        (out_cfg.test_predictions_path, test_indices, test_prob, test_true, "test"),
    ):
        pd.DataFrame(
            {
                "row_index": subset_indices,
                "sample_id": sample_keys[subset_indices],
                "partition": partition_name,
                "block_id": all_groups[subset_indices],
                "label": truth,
                "probability": probabilities,
                "prediction": (probabilities >= train_cfg.threshold).astype(np.int64),
                "x": coords[subset_indices, 0],
                "y": coords[subset_indices, 1],
            }
        ).to_csv(output_path, index=False, encoding="utf-8-sig")
    plot_roc_curve(test_true, test_prob, out_cfg.roc_fig_path, data_path=out_cfg.roc_curve_data_path)
    plot_pr_curve(test_true, test_prob, out_cfg.pr_fig_path, data_path=out_cfg.pr_curve_data_path)

    metrics_payload = {
        "algorithm": algo_cfg.algorithm,
        "manuscript_model_label": manuscript_model_label(algo_cfg.algorithm),
        "optimizer": algo_cfg.optimizer,
        "optimized_params": best_params,
        "selection_metric": train_cfg.selection_metric,
        "best_validation_selection_score": best_val_auc if np.isfinite(best_val_auc) else None,
        "best_val_auc_during_training": (
            best_val_auc if train_cfg.selection_metric == "roc_auc" and np.isfinite(best_val_auc) else None
        ),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(out_cfg.metrics_path, metrics_payload)

    report_text = [
        "Validation Classification Report",
        classification_report(val_true, (val_prob >= train_cfg.threshold).astype(int), digits=4, zero_division=0),
        "",
        "Test Classification Report",
        classification_report(test_true, (test_prob >= train_cfg.threshold).astype(int), digits=4, zero_division=0),
    ]
    out_cfg.report_text_path.write_text("\n".join(report_text), encoding="utf-8")

    save_json(out_cfg.model_params_path, model_param_payload)

    try:
        compute_shap_importance(
            model=model,
            algorithm=algo_cfg.algorithm,
            X_background=shap_X_background,
            X_eval=shap_X_eval,
            feature_names=shap_feature_names,
            explain_cfg=explain_cfg,
            out_cfg=out_cfg,
            device=(device if algo_cfg.algorithm == GINN_INTERNAL_KEY or algo_cfg.algorithm in DEEP_COMPARE_ALGOS else None),
        )
    except Exception as exc:
        logging.warning("SHAP analysis failed and was skipped: %s", exc)

    logging.info("Training completed.")
    logging.info("Best validation %s: %.5f", train_cfg.selection_metric, best_val_auc)
    logging.info("Test ROC-AUC: %s", test_metrics.get("roc_auc"))
    logging.info("Artifacts root: %s", out_cfg.output_root.resolve())


if __name__ == "__main__":
    main()
