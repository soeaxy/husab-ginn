from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


PRIMARY_COMPONENTS = ("rf", "gcn_transformer", "pinn")
PRIMARY_LABELS = {
    "rf": "RF",
    "gcn_transformer": "GCN-Transformer",
    "pinn": "GINN",
    "model_only_ensemble": "Model-only ensemble",
    "geo_augmented_ensemble": "Geology-augmented ensemble",
}
TOPK_FRACTIONS = (0.01, 0.05, 0.10)
DEFAULT_GRID_STEP = 0.05
DEFAULT_BOOTSTRAP = 10000
DEFAULT_SEED = 2026
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.6,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
CONTRASTS = (
    ("model_only_ensemble", "rf"),
    ("geo_augmented_ensemble", "rf"),
    ("geo_augmented_ensemble", "model_only_ensemble"),
)
PAIRWISE_METRICS = ("pr_auc", "top_05pct_capture")
WARNING_TEXT = (
    "Circularity warning: dome, fault, and strata priors contributed to the rebuilt pseudo-negative "
    "protocol. The geology-augmented ensemble is therefore diagnostic for internal ranking behavior, "
    "not an independent external validation result."
)


@dataclass(frozen=True)
class AnalysisConfig:
    experiment_root: Path
    output_dir: Path
    figures_dir: Path
    grid_step: float
    bootstrap_iterations: int
    seed: int


def parse_args() -> AnalysisConfig:
    project_root = Path(__file__).resolve().parent
    experiment_root = project_root / "experiments" / "reconstructed_v2_spatial_prselected"
    output_dir = project_root / "analysis_output" / "reconstructed_v2_evidence_guided_ensemble"
    parser = argparse.ArgumentParser(
        description=(
            "Build validation-selected rank ensembles over RF, GCN-Transformer, and GINN "
            "for the reconstructed_v2_spatial_prselected benchmark."
        )
    )
    parser.add_argument("--experiment-root", type=Path, default=experiment_root)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--grid-step", type=float, default=DEFAULT_GRID_STEP)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    return AnalysisConfig(
        experiment_root=args.experiment_root,
        output_dir=args.output_dir,
        figures_dir=args.output_dir / "figures",
        grid_step=float(args.grid_step),
        bootstrap_iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )


def simplex_weight_grid(dimensions: int, step: float = DEFAULT_GRID_STEP) -> list[tuple[float, ...]]:
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    units = round(1.0 / step)
    if not math.isclose(units * step, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"grid step {step} does not evenly tile the simplex")
    combos: list[tuple[float, ...]] = []

    def recurse(remaining_dims: int, remaining_units: int, prefix: list[int]) -> None:
        if remaining_dims == 1:
            combos.append(tuple((prefix + [remaining_units])[idx] * step for idx in range(dimensions)))
            return
        for current in range(remaining_units + 1):
            recurse(remaining_dims - 1, remaining_units - current, prefix + [current])

    recurse(dimensions, units, [])
    return combos


def rank_normalize(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty 1D sequence")
    if np.isnan(values).any():
        raise ValueError("scores contain NaN values")
    if values.size == 1:
        return np.array([1.0], dtype=float)
    descending_ranks = stats.rankdata(-values, method="average")
    normalized = (values.size - descending_ranks) / (values.size - 1)
    return normalized.astype(float)


def compute_topk_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    total_positive = max(1, int(sorted_labels.sum()))
    total_samples = sorted_labels.size
    metrics: dict[str, float] = {}
    for fraction in TOPK_FRACTIONS:
        k = max(1, int(math.ceil(total_samples * fraction)))
        positives = int(sorted_labels[:k].sum())
        prefix = f"top_{int(fraction * 100):02d}pct"
        metrics[f"{prefix}_capture"] = positives / total_positive
        metrics[f"{prefix}_precision"] = positives / k
    return metrics


def compute_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if labels_array.ndim != 1 or scores_array.ndim != 1 or labels_array.size != scores_array.size:
        raise ValueError("labels and scores must be aligned 1D sequences")
    metrics = {
        "pr_auc": float(average_precision_score(labels_array, scores_array)),
        "roc_auc": float(roc_auc_score(labels_array, scores_array)),
    }
    metrics.update(compute_topk_metrics(labels_array, scores_array))
    return metrics


def bootstrap_mean_ci(values: Sequence[float], n_boot: int, seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("cannot bootstrap empty values")
    rng = np.random.default_rng(seed)
    draws = rng.choice(data, size=(n_boot, data.size), replace=True)
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def signed_rank_biserial(differences: Sequence[float]) -> float:
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[np.abs(diffs) > 0]
    if diffs.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diffs))
    positive = float(ranks[diffs > 0].sum())
    negative = float(ranks[diffs < 0].sum())
    denom = positive + negative
    return 0.0 if denom == 0.0 else (positive - negative) / denom


def holm_correction(p_values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(float(value) for value in p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(indexed)
    running_max = 0.0
    total = len(indexed)
    for rank, (original_idx, p_val) in enumerate(indexed, start=1):
        corrected = min(1.0, (total - rank + 1) * p_val)
        running_max = max(running_max, corrected)
        adjusted[original_idx] = running_max
    return adjusted


def merge_component_frames(
    model_frames: dict[str, pd.DataFrame],
    geo_frame: pd.DataFrame,
    partition: str,
) -> pd.DataFrame:
    if not model_frames:
        raise ValueError("model_frames must not be empty")
    required_pred_columns = {"sample_id", "label", "x", "y", "probability"}
    merged: pd.DataFrame | None = None
    for model_name, raw_frame in model_frames.items():
        frame = raw_frame.copy()
        missing = required_pred_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{model_name} is missing required prediction columns: {sorted(missing)}")
        frame = frame.loc[frame["partition"].astype(str) == partition].copy()
        if frame.empty:
            raise ValueError(f"{model_name} has no rows for partition={partition!r}")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{model_name} contains duplicated sample_id values in partition={partition!r}")
        frame = frame.rename(
            columns={
                "label": f"label_{model_name}",
                "x": f"x_{model_name}",
                "y": f"y_{model_name}",
                "probability": f"probability_{model_name}",
                "block_id": f"block_id_{model_name}",
                "prediction": f"prediction_{model_name}",
                "partition": f"partition_{model_name}",
                "row_index": f"row_index_{model_name}",
            }
        )
        selected = frame[
            [
                "sample_id",
                f"label_{model_name}",
                f"x_{model_name}",
                f"y_{model_name}",
                f"probability_{model_name}",
                f"block_id_{model_name}",
                f"prediction_{model_name}",
                f"partition_{model_name}",
            ]
        ]
        merged = selected if merged is None else merged.merge(selected, on="sample_id", how="inner", validate="one_to_one")

    assert merged is not None
    merged = merged.merge(geo_frame, on="sample_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(f"no aligned rows after merging partition={partition!r}")

    base_model = next(iter(model_frames.keys()))
    merged["label"] = pd.to_numeric(merged[f"label_{base_model}"], errors="raise").astype(int)
    merged["x"] = pd.to_numeric(merged[f"x_{base_model}"], errors="raise")
    merged["y"] = pd.to_numeric(merged[f"y_{base_model}"], errors="raise")
    for model_name in model_frames:
        label_series = pd.to_numeric(merged[f"label_{model_name}"], errors="raise").astype(int)
        if not np.array_equal(label_series.to_numpy(), merged["label"].to_numpy()):
            raise ValueError(f"label mismatch detected for {model_name} in partition={partition!r}")
        x_series = pd.to_numeric(merged[f"x_{model_name}"], errors="raise")
        y_series = pd.to_numeric(merged[f"y_{model_name}"], errors="raise")
        if not np.allclose(x_series.to_numpy(dtype=float), merged["x"].to_numpy(dtype=float), atol=1e-6):
            raise ValueError(f"x-coordinate mismatch detected for {model_name} in partition={partition!r}")
        if not np.allclose(y_series.to_numpy(dtype=float), merged["y"].to_numpy(dtype=float), atol=1e-6):
            raise ValueError(f"y-coordinate mismatch detected for {model_name} in partition={partition!r}")
        if not (merged[f"partition_{model_name}"].astype(str) == partition).all():
            raise ValueError(f"partition mismatch detected for {model_name} in partition={partition!r}")

    geo_labels = pd.to_numeric(merged["Class"], errors="raise").astype(int)
    if not np.array_equal(geo_labels.to_numpy(), merged["label"].to_numpy()):
        raise ValueError(f"shapefile label mismatch detected in partition={partition!r}")
    geom_x = merged["geometry"].map(lambda geom: float(geom.x)).to_numpy(dtype=float)
    geom_y = merged["geometry"].map(lambda geom: float(geom.y)).to_numpy(dtype=float)
    if not np.allclose(geom_x, merged["x"].to_numpy(dtype=float), atol=1e-6):
        raise ValueError(f"geometry.x mismatch detected in partition={partition!r}")
    if not np.allclose(geom_y, merged["y"].to_numpy(dtype=float), atol=1e-6):
        raise ValueError(f"geometry.y mismatch detected in partition={partition!r}")
    return merged


def select_weights(
    frame: pd.DataFrame,
    component_names: Sequence[str],
    grid_step: float,
) -> tuple[tuple[float, ...], dict[str, float]]:
    weight_grid = simplex_weight_grid(len(component_names), grid_step)
    component_matrix = np.column_stack([frame[f"rank_{name}"].to_numpy(dtype=float) for name in component_names])
    labels = frame["label"].to_numpy(dtype=int)
    best_weights: tuple[float, ...] | None = None
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, ...] | None = None
    geo_index = component_names.index("geo_score") if "geo_score" in component_names else None

    for weights in weight_grid:
        score = component_matrix @ np.asarray(weights, dtype=float)
        metrics = compute_metrics(labels, score)
        geo_weight = weights[geo_index] if geo_index is not None else 0.0
        nonzero_count = sum(value > 0.0 for value in weights)
        key = (
            metrics["pr_auc"],
            metrics["top_05pct_capture"],
            -geo_weight,
            -float(nonzero_count),
            *weights,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_weights = weights
            best_metrics = metrics
    assert best_weights is not None and best_metrics is not None
    return best_weights, best_metrics


def build_ensemble_score(frame: pd.DataFrame, component_names: Sequence[str], weights: Sequence[float]) -> np.ndarray:
    matrix = np.column_stack([frame[f"rank_{name}"].to_numpy(dtype=float) for name in component_names])
    return matrix @ np.asarray(weights, dtype=float)


def load_geo_frame(shapefile_path: Path) -> pd.DataFrame:
    gdf = gpd.read_file(shapefile_path)
    required = {"sample_id", "Class", "geo_score", "geometry"}
    missing = required.difference(gdf.columns)
    if missing:
        raise ValueError(f"{shapefile_path} is missing required shapefile columns: {sorted(missing)}")
    if gdf["sample_id"].duplicated().any():
        raise ValueError(f"{shapefile_path} contains duplicated sample_id values")
    return gdf[["sample_id", "Class", "geo_score", "geometry"]].copy()


def load_prediction_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if "sample_id" not in frame.columns:
        raise ValueError(f"{csv_path} is missing sample_id")
    return frame


def load_realization_frames(experiment_root: Path, realization: int) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    realization_root = experiment_root / f"realization_{realization:02d}"
    if not realization_root.exists():
        raise FileNotFoundError(realization_root)
    model_frames_validation: dict[str, pd.DataFrame] = {}
    model_frames_test: dict[str, pd.DataFrame] = {}
    for component in PRIMARY_COMPONENTS:
        metrics_dir = realization_root / component / f"seed_{DEFAULT_SEED}" / "metrics"
        validation_path = metrics_dir / "validation_predictions.csv"
        test_path = metrics_dir / "test_predictions.csv"
        if not validation_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"missing prediction file for realization={realization:02d}, model={component}")
        model_frames_validation[component] = load_prediction_frame(validation_path)
        model_frames_test[component] = load_prediction_frame(test_path)
    shapefile_path = experiment_root.parent.parent / "data" / "reconstructed_samples_v2" / f"realization_{realization:02d}" / "combined_samples_rebuilt.shp"
    geo_frame = load_geo_frame(shapefile_path)
    return model_frames_validation, model_frames_test, geo_frame


def prepare_partition(frame: pd.DataFrame, component_names: Sequence[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for component in component_names:
        if component == "geo_score":
            values = pd.to_numeric(prepared["geo_score"], errors="raise").to_numpy(dtype=float)
        else:
            values = pd.to_numeric(prepared[f"probability_{component}"], errors="raise").to_numpy(dtype=float)
        prepared[f"rank_{component}"] = rank_normalize(values)
    return prepared


def summarize_by_method(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_columns = [
        "pr_auc",
        "roc_auc",
        "top_01pct_capture",
        "top_01pct_precision",
        "top_05pct_capture",
        "top_05pct_precision",
        "top_10pct_capture",
        "top_10pct_precision",
    ]
    for method, subset in metrics_frame.groupby("method", sort=False):
        row: dict[str, object] = {
            "method": method,
            "label": PRIMARY_LABELS[method],
            "n_realizations": int(len(subset)),
        }
        for metric in metric_columns:
            values = subset[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            ci_low, ci_high = bootstrap_mean_ci(values, DEFAULT_BOOTSTRAP, DEFAULT_SEED + len(rows))
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def frame_to_markdown(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if frame.empty:
        return "_No data available._"
    headers = [str(column) for column in frame.columns]
    rows: list[list[str]] = []
    for row in frame.itertuples(index=False, name=None):
        rendered: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append(format(float(value), floatfmt))
            else:
                rendered.append(str(value))
        rows.append(rendered)
    matrix = [headers, ["---"] * len(headers), *rows]
    widths = [max(len(str(row[idx])) for row in matrix) for idx in range(len(headers))]

    def render_row(values: Sequence[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    return "\n".join(render_row(row) for row in matrix)


def paired_wilcoxon_tests(metrics_frame: pd.DataFrame, bootstrap_iterations: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pivot = metrics_frame.pivot(index="realization", columns="method")
    for metric in PAIRWISE_METRICS:
        for left, right in CONTRASTS:
            left_values = pivot[(metric, left)].to_numpy(dtype=float)
            right_values = pivot[(metric, right)].to_numpy(dtype=float)
            differences = left_values - right_values
            if np.allclose(differences, 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                wilcoxon = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided", mode="exact")
                statistic = float(wilcoxon.statistic)
                p_value = float(wilcoxon.pvalue)
            ci_low, ci_high = bootstrap_mean_ci(differences, bootstrap_iterations, seed + len(rows))
            rows.append(
                {
                    "metric": metric,
                    "left_method": left,
                    "left_label": PRIMARY_LABELS[left],
                    "right_method": right,
                    "right_label": PRIMARY_LABELS[right],
                    "n_realizations": int(differences.size),
                    "mean_diff": float(differences.mean()),
                    "median_diff": float(np.median(differences)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_value,
                    "rank_biserial": float(signed_rank_biserial(differences)),
                }
            )
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm_correction(frame["p_raw"].tolist())
    frame["significant_at_0_05"] = frame["p_holm"] < 0.05
    return frame


def plot_metric_distributions(metrics_frame: pd.DataFrame, metric: str, output_stem: Path) -> None:
    method_order = ["rf", "model_only_ensemble", "geo_augmented_ensemble"]
    fig, ax = plt.subplots(figsize=(6.9, 3.8), layout="constrained")
    data = [metrics_frame.loc[metrics_frame["method"] == method, metric].to_numpy(dtype=float) for method in method_order]
    parts = ax.violinplot(data, positions=np.arange(1, len(method_order) + 1), showmeans=True, showextrema=False, widths=0.8)
    colors = ["#CC79A7", "#0072B2", "#009E73"]
    for body, color in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
    parts["cmeans"].set_color("#333333")
    parts["cmeans"].set_linewidth(1.0)
    for idx, (method, color) in enumerate(zip(method_order, colors, strict=True), start=1):
        values = metrics_frame.loc[metrics_frame["method"] == method, metric].to_numpy(dtype=float)
        jitter = np.linspace(-0.08, 0.08, values.size)
        ax.scatter(np.full(values.shape, idx) + jitter, values, color=color, edgecolors="white", linewidths=0.4, s=18, zorder=3)
    ax.set_xticks(np.arange(1, len(method_order) + 1))
    ax.set_xticklabels([PRIMARY_LABELS[method] for method in method_order])
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"{metric.replace('_', ' ').upper()} distribution across 10 realizations")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def write_reports(
    cfg: AnalysisConfig,
    metrics_frame: pd.DataFrame,
    weights_frame: pd.DataFrame,
    paired_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    verification: dict[str, object],
) -> None:
    metrics_frame.to_csv(cfg.output_dir / "per_realization_metrics.csv", index=False, encoding="utf-8-sig")
    weights_frame.to_csv(cfg.output_dir / "selected_weights.csv", index=False, encoding="utf-8-sig")
    paired_frame.to_csv(cfg.output_dir / "paired_tests.csv", index=False, encoding="utf-8-sig")
    summary_frame.to_csv(cfg.output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    with (cfg.output_dir / "verification.json").open("w", encoding="utf-8") as stream:
        json.dump(verification, stream, ensure_ascii=False, indent=2)

    summary_cols = [
        "label",
        "pr_auc_mean",
        "pr_auc_std",
        "pr_auc_ci95_low",
        "pr_auc_ci95_high",
        "roc_auc_mean",
        "top_05pct_capture_mean",
        "top_05pct_precision_mean",
    ]
    paired_cols = [
        "metric",
        "left_label",
        "right_label",
        "mean_diff",
        "ci95_low",
        "ci95_high",
        "p_raw",
        "p_holm",
        "rank_biserial",
    ]
    top_weights = (
        weights_frame.groupby("variant")[["weight_rf", "weight_gcn_transformer", "weight_pinn", "weight_geo_score"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    analysis_lines = [
        "# Evidence-guided ensemble analysis",
        "",
        "## What this script did",
        "",
        "- Compared RF, GCN-Transformer, GINN, a validation-selected model-only rank ensemble, and a validation-selected geology-augmented rank ensemble.",
        "- Selected nonnegative simplex weights on the validation partition only (grid step 0.05), then applied those fixed weights to the untouched test partition.",
        "- Rank-normalized every component within partition before blending; the resulting score is a ranking score, not a calibrated probability.",
        "",
        "## Circularity warning",
        "",
        f"- {WARNING_TEXT}",
        "",
        "## Mean test performance",
        "",
        frame_to_markdown(summary_frame[summary_cols], floatfmt=".4f"),
        "",
        "## Mean selected weights across realizations",
        "",
        frame_to_markdown(top_weights, floatfmt=".4f"),
        "",
        "## Paired tests reported here",
        "",
        frame_to_markdown(paired_frame[paired_cols], floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- If the geology-augmented ensemble beats the model-only ensemble on PR-AUC or top-5% capture, that means the geological prior still carries ranking signal under the current reconstructed-negative protocol.",
        "- Because `geo_score` partially informed the rebuilt pseudo-negative pool, any gain from the geology-augmented ensemble should be treated as an internal-diagnostic result rather than an independent validation claim.",
        "- The model-only ensemble is the fairer comparison for manuscript claims about predictive complementarity among RF, GCN-Transformer, and GINN.",
    ]
    (cfg.output_dir / "analysis-report.md").write_text("\n".join(analysis_lines), encoding="utf-8")

    stats_lines = [
        "# Statistical appendix",
        "",
        "## Summary statistics",
        "",
        frame_to_markdown(summary_frame, floatfmt=".4f"),
        "",
        "## Per-contrast paired tests",
        "",
        frame_to_markdown(paired_frame, floatfmt=".4f"),
        "",
        "## Notes",
        "",
        "- Wilcoxon signed-rank tests are paired over the 10 negative-sample realizations.",
        "- Rank-biserial > 0 favors the left method.",
        "- Confidence intervals are bootstrap 95% intervals on the mean paired difference.",
        "- Holm correction is applied over all planned paired tests reported in `paired_tests.csv`.",
    ]
    (cfg.output_dir / "stats-appendix.md").write_text("\n".join(stats_lines), encoding="utf-8")

    figure_lines = [
        "# Figure catalog",
        "",
        "| Figure | Files | Purpose | Key readout |",
        "| --- | --- | --- | --- |",
        "| Exploratory Figure 1 | `figures/figure_01_pr_auc_distribution.png`, `figures/figure_01_pr_auc_distribution.pdf` | Compare PR-AUC distributions for RF, the model-only ensemble, and the geology-augmented ensemble. | Whether ensemble blending improves the primary ranking metric over single-model RF. |",
        "| Exploratory Figure 2 | `figures/figure_02_top5_capture_distribution.png`, `figures/figure_02_top5_capture_distribution.pdf` | Compare top-5% positive capture distributions for RF, the model-only ensemble, and the geology-augmented ensemble. | Whether blending improves early target capture, which is often more decision-relevant than a fixed threshold. |",
    ]
    (cfg.output_dir / "figure-catalog.md").write_text("\n".join(figure_lines), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    verification: dict[str, object] = {
        "experiment_root": str(cfg.experiment_root),
        "grid_step": cfg.grid_step,
        "bootstrap_iterations": cfg.bootstrap_iterations,
        "seed": cfg.seed,
        "warning": WARNING_TEXT,
        "realizations": [],
    }

    for realization in range(10):
        validation_frames, test_frames, geo_frame = load_realization_frames(cfg.experiment_root, realization)
        validation_merged = merge_component_frames(validation_frames, geo_frame, "validation")
        test_merged = merge_component_frames(test_frames, geo_frame, "test")
        validation_prepared = prepare_partition(validation_merged, (*PRIMARY_COMPONENTS, "geo_score"))
        test_prepared = prepare_partition(test_merged, (*PRIMARY_COMPONENTS, "geo_score"))

        rf_validation_metrics = compute_metrics(validation_prepared["label"], validation_prepared["rank_rf"])
        rf_test_metrics = compute_metrics(test_prepared["label"], test_prepared["rank_rf"])
        gcn_test_metrics = compute_metrics(test_prepared["label"], test_prepared["rank_gcn_transformer"])
        pinn_test_metrics = compute_metrics(test_prepared["label"], test_prepared["rank_pinn"])

        model_only_weights, model_only_validation_metrics = select_weights(validation_prepared, PRIMARY_COMPONENTS, cfg.grid_step)
        geo_augmented_weights, geo_augmented_validation_metrics = select_weights(
            validation_prepared,
            (*PRIMARY_COMPONENTS, "geo_score"),
            cfg.grid_step,
        )
        model_only_test_scores = build_ensemble_score(test_prepared, PRIMARY_COMPONENTS, model_only_weights)
        geo_augmented_test_scores = build_ensemble_score(
            test_prepared, (*PRIMARY_COMPONENTS, "geo_score"), geo_augmented_weights
        )
        model_only_test_metrics = compute_metrics(test_prepared["label"], model_only_test_scores)
        geo_augmented_test_metrics = compute_metrics(test_prepared["label"], geo_augmented_test_scores)

        for method, test_metrics, validation_metrics in [
            ("rf", rf_test_metrics, rf_validation_metrics),
            ("gcn_transformer", gcn_test_metrics, compute_metrics(validation_prepared["label"], validation_prepared["rank_gcn_transformer"])),
            ("pinn", pinn_test_metrics, compute_metrics(validation_prepared["label"], validation_prepared["rank_pinn"])),
            ("model_only_ensemble", model_only_test_metrics, model_only_validation_metrics),
            ("geo_augmented_ensemble", geo_augmented_test_metrics, geo_augmented_validation_metrics),
        ]:
            metrics_rows.append(
                {
                    "realization": realization,
                    "method": method,
                    "label": PRIMARY_LABELS[method],
                    "partition": "test",
                    **test_metrics,
                    "validation_pr_auc": float(validation_metrics["pr_auc"]),
                    "validation_top_05pct_capture": float(validation_metrics["top_05pct_capture"]),
                }
            )

        weight_rows.extend(
            [
                {
                    "realization": realization,
                    "variant": "model_only_ensemble",
                    "weight_rf": float(model_only_weights[0]),
                    "weight_gcn_transformer": float(model_only_weights[1]),
                    "weight_pinn": float(model_only_weights[2]),
                    "weight_geo_score": 0.0,
                    "validation_pr_auc": float(model_only_validation_metrics["pr_auc"]),
                    "validation_top_05pct_capture": float(model_only_validation_metrics["top_05pct_capture"]),
                    "test_pr_auc": float(model_only_test_metrics["pr_auc"]),
                    "test_top_05pct_capture": float(model_only_test_metrics["top_05pct_capture"]),
                },
                {
                    "realization": realization,
                    "variant": "geo_augmented_ensemble",
                    "weight_rf": float(geo_augmented_weights[0]),
                    "weight_gcn_transformer": float(geo_augmented_weights[1]),
                    "weight_pinn": float(geo_augmented_weights[2]),
                    "weight_geo_score": float(geo_augmented_weights[3]),
                    "validation_pr_auc": float(geo_augmented_validation_metrics["pr_auc"]),
                    "validation_top_05pct_capture": float(geo_augmented_validation_metrics["top_05pct_capture"]),
                    "test_pr_auc": float(geo_augmented_test_metrics["pr_auc"]),
                    "test_top_05pct_capture": float(geo_augmented_test_metrics["top_05pct_capture"]),
                },
            ]
        )

        verification["realizations"].append(
            {
                "realization": realization,
                "validation_n": int(len(validation_prepared)),
                "test_n": int(len(test_prepared)),
                "model_only_weights": [float(value) for value in model_only_weights],
                "geo_augmented_weights": [float(value) for value in geo_augmented_weights],
            }
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    weights_frame = pd.DataFrame(weight_rows)
    paired_frame = paired_wilcoxon_tests(metrics_frame, cfg.bootstrap_iterations, cfg.seed)
    summary_frame = summarize_by_method(metrics_frame)

    with plt.rc_context(PLOT_STYLE):
        plot_metric_distributions(metrics_frame, "pr_auc", cfg.figures_dir / "figure_01_pr_auc_distribution")
        plot_metric_distributions(metrics_frame, "top_05pct_capture", cfg.figures_dir / "figure_02_top5_capture_distribution")

    verification["output_files"] = [
        "analysis-report.md",
        "stats-appendix.md",
        "figure-catalog.md",
        "per_realization_metrics.csv",
        "selected_weights.csv",
        "paired_tests.csv",
        "summary_metrics.csv",
        "verification.json",
    ]
    write_reports(cfg, metrics_frame, weights_frame, paired_frame, summary_frame, verification)


if __name__ == "__main__":
    main()
