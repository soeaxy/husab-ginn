from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


MODEL_ORDER = [
    "pinn",
    "mlp",
    "gcn_transformer",
    "rf",
    "lightgbm",
    "xgboost",
    "catboost",
]

DEEP_MODEL_ALGORITHMS = frozenset({"pinn", "mlp", "gcn_transformer"})
CLASSICAL_MODEL_ALGORITHMS = frozenset({"rf", "lightgbm", "xgboost", "catboost"})

MODEL_LABELS = {
    "pinn": "GINN",
    "mlp": "MLP (no physics)",
    "gcn_transformer": "GCN-Transformer",
    "rf": "RF",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}

MODEL_COLORS = {
    "pinn": "#0072B2",
    "mlp": "#009E73",
    "gcn_transformer": "#D55E00",
    "rf": "#CC79A7",
    "lightgbm": "#56B4E9",
    "xgboost": "#E69F00",
    "catboost": "#000000",
}

MODEL_MARKERS = {
    "pinn": "o",
    "mlp": "s",
    "gcn_transformer": "^",
    "rf": "D",
    "lightgbm": "P",
    "xgboost": "X",
    "catboost": "v",
}

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.2,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.4,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.3,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

PRIMARY_METRIC = "pr_auc"
SECONDARY_METRICS = ["roc_auc", "mcc", "recall", "brier_score"]
ALL_METRICS = [PRIMARY_METRIC, *SECONDARY_METRICS]
BOOTSTRAP_ITERATIONS = 10000
SEED = 2026
TOPK_FRACTIONS = (0.01, 0.05, 0.10)
REQUIRED_ARTIFACT_FILES = (
    "metrics.json",
    "run_config.json",
    "split_assignments.csv",
    "test_predictions.csv",
)


@dataclass(frozen=True)
class AnalysisConfig:
    benchmark_root: Path
    output_dir: Path
    split_manifest: Path
    ledger_path: Path
    figures_dir: Path
    summary_csv: Path
    metrics_long_csv: Path
    paired_tests_csv: Path
    assumption_checks_csv: Path
    verification_json: Path
    allow_incomplete: bool


def parse_args() -> AnalysisConfig:
    project_root = Path(__file__).resolve().parent
    benchmark_root = project_root / "experiments" / "reconstructed_v2_spatial"
    output_dir = project_root / "analysis_output" / "reconstructed_v2_benchmark"
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the reconstructed-negative benchmark matrix and export "
            "publication-grade statistics and figures."
        )
    )
    parser.add_argument("--benchmark-root", "--experiment-root", dest="benchmark_root", type=Path, default=benchmark_root)
    parser.add_argument("--output-dir", "--output-root", dest="output_dir", type=Path, default=output_dir)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir
    return AnalysisConfig(
        benchmark_root=args.benchmark_root,
        output_dir=output_dir,
        split_manifest=args.benchmark_root / "fixed_spatial_split.json",
        ledger_path=args.benchmark_root / "run_ledger.csv",
        figures_dir=output_dir / "figures",
        summary_csv=output_dir / "summary_metrics.csv",
        metrics_long_csv=output_dir / "metrics_long.csv",
        paired_tests_csv=output_dir / "paired_tests.csv",
        assumption_checks_csv=output_dir / "assumption_checks.csv",
        verification_json=output_dir / "benchmark_verification.json",
        allow_incomplete=bool(args.allow_incomplete),
    )


def ensure_exists(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_ITERATIONS, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True)
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, (idx, p_val) in enumerate(indexed, start=1):
        adj = min(1.0, (m - rank + 1) * p_val)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def markdown_table(frame: pd.DataFrame, columns: list[str], floatfmt: str = ".4f") -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, divider]
    for _, row in frame[columns].iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(format(float(value), floatfmt))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def holm_correction(p_values: list[float]) -> list[float]:
    return holm_adjust(p_values)


def rank_biserial_from_differences(differences: np.ndarray) -> float:
    diffs = differences[np.abs(differences) > 0]
    if diffs.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diffs))
    positive = float(ranks[diffs > 0].sum())
    negative = float(ranks[diffs < 0].sum())
    denom = positive + negative
    if denom == 0:
        return 0.0
    return (positive - negative) / denom


def signed_rank_biserial(differences: np.ndarray) -> float:
    return rank_biserial_from_differences(differences)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_curve(curve_path: Path, x_col: str, y_col: str, grid: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(curve_path)
    frame = frame[[x_col, y_col]].dropna()
    if frame.empty:
        raise ValueError(f"Curve file is empty: {curve_path}")
    frame = frame.sort_values(x_col)
    grouped = frame.groupby(x_col, as_index=False)[y_col].max()
    xs = grouped[x_col].to_numpy(dtype=float)
    ys = grouped[y_col].to_numpy(dtype=float)
    return np.interp(grid, xs, ys, left=ys[0], right=ys[-1])


def load_benchmark_manifest(benchmark_root: Path) -> dict[str, object]:
    manifest_path = benchmark_root / "benchmark_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    realizations = sorted(
        {
            int(path.name.split("_")[-1])
            for path in benchmark_root.glob("realization_*")
            if path.is_dir() and path.name.split("_")[-1].isdigit()
        }
    )
    return {"algorithms": MODEL_ORDER, "realizations": realizations, "model_seed": SEED}


def validate_selection_protocol(
    algorithm: str,
    run_config: dict[str, object],
    metrics: dict[str, object],
) -> str:
    if algorithm in CLASSICAL_MODEL_ALGORITHMS:
        optimizer = run_config.get("algo_config", {}).get("optimizer")
        if optimizer != "none":
            raise ValueError(
                f"{algorithm} is not a single-fit estimator in this run: optimizer={optimizer!r}"
            )
        return "single_fit_no_checkpoint"
    if algorithm not in DEEP_MODEL_ALGORITHMS:
        raise ValueError(f"Unknown benchmark algorithm: {algorithm}")

    configured_metric = run_config.get("train_config", {}).get("selection_metric")
    recorded_metric = metrics.get("selection_metric")
    if configured_metric != "pr_auc" or recorded_metric != "pr_auc":
        raise ValueError(f"{algorithm} did not record validation PR-AUC checkpoint selection.")

    validation_pr_auc = metrics.get("validation_metrics", {}).get("pr_auc")
    selected_score = metrics.get("best_validation_selection_score")
    if validation_pr_auc is None or selected_score is None:
        raise ValueError(f"{algorithm} is missing validation PR-AUC selection evidence.")
    if not math.isclose(float(selected_score), float(validation_pr_auc), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"{algorithm} selected score does not match validation PR-AUC: "
            f"{selected_score} != {validation_pr_auc}"
        )
    return "validation_pr_auc_checkpoint"


def collect_artifacts(
    benchmark_root: Path, manifest: dict[str, object]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    artifacts: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    warnings: list[str] = []
    model_seed = int(manifest.get("model_seed", SEED) or SEED)
    for realization in [int(value) for value in manifest.get("realizations", [])]:
        for algorithm in [str(value) for value in manifest.get("algorithms", MODEL_ORDER)]:
            metrics_dir = benchmark_root / f"realization_{realization:02d}" / algorithm / f"seed_{model_seed}" / "metrics"
            if not metrics_dir.exists():
                missing.append({"realization": realization, "algorithm": algorithm, "status": "missing_directory"})
                continue
            absent = [name for name in REQUIRED_ARTIFACT_FILES if not (metrics_dir / name).exists()]
            if absent:
                missing.append(
                    {
                        "realization": realization,
                        "algorithm": algorithm,
                        "status": "missing_required_files",
                        "missing_files": ";".join(absent),
                    }
                )
                continue
            artifacts.append(
                {
                    "realization": realization,
                    "algorithm": algorithm,
                    "metrics_dir": str(metrics_dir),
                }
            )
    return artifacts, missing, warnings


def frame_to_markdown(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if frame.empty:
        return "_No data available._"
    formatter = "{:" + floatfmt + "}"
    headers = [str(column) for column in frame.columns]
    rows: list[list[str]] = []
    for row in frame.itertuples(index=False, name=None):
        rendered: list[str] = []
        for value in row:
            if isinstance(value, float):
                rendered.append(formatter.format(value))
            else:
                rendered.append(str(value))
        rows.append(rendered)
    matrix = [headers, ["---"] * len(headers), *rows]
    widths = [max(len(str(row[idx])) for row in matrix) for idx in range(len(headers))]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    return "\n".join(render_row([str(value) for value in row]) for row in matrix)


def compute_topk_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    order = np.argsort(probabilities)[::-1]
    labels_sorted = labels[order]
    total_positive = max(1, int(labels_sorted.sum()))
    total_samples = labels_sorted.size
    results: dict[str, float] = {}
    for fraction in TOPK_FRACTIONS:
        k = max(1, int(math.ceil(total_samples * fraction)))
        positives = int(labels_sorted[:k].sum())
        prefix = f"top_{int(fraction * 100):02d}pct"
        results[f"{prefix}_capture"] = positives / total_positive
        results[f"{prefix}_precision"] = positives / k
    return results


def cumulative_gain_curve(labels: np.ndarray, probabilities: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(probabilities)[::-1]
    labels_sorted = labels[order]
    total_positive = max(1, int(labels_sorted.sum()))
    cumulative_capture = np.cumsum(labels_sorted) / total_positive
    sample_share = np.arange(1, labels_sorted.size + 1, dtype=float) / labels_sorted.size
    sample_share = np.concatenate([[0.0], sample_share])
    cumulative_capture = np.concatenate([[0.0], cumulative_capture])
    return np.interp(grid, sample_share, cumulative_capture)


def collect_metrics(cfg: AnalysisConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_exists([cfg.benchmark_root, cfg.ledger_path, cfg.split_manifest])
    ledger = pd.read_csv(cfg.ledger_path)
    if ledger.empty:
        raise ValueError("run_ledger.csv is empty.")
    manifest = load_benchmark_manifest(cfg.benchmark_root)
    if manifest.get("selection_metric") != "pr_auc":
        raise ValueError("Formal benchmark must configure validation PR-AUC for deep-model checkpoint selection.")
    if manifest.get("primary_metric") != "test_pr_auc":
        raise ValueError("Formal benchmark must declare test PR-AUC as the primary endpoint.")
    expected = len(manifest.get("realizations", [])) * len(manifest.get("algorithms", MODEL_ORDER))
    if not cfg.allow_incomplete and len(ledger) != expected:
        raise ValueError(f"Expected {expected} ledger rows, found {len(ledger)}.")
    accepted_statuses = {"completed", "skipped_complete"}
    completed_rows = int(ledger["status"].isin(accepted_statuses).sum())
    incomplete = ledger.loc[~ledger["status"].isin(accepted_statuses)]
    if not cfg.allow_incomplete and not incomplete.empty:
        raise ValueError(f"Incomplete benchmark rows remain:\n{incomplete.to_string(index=False)}")
    if cfg.allow_incomplete:
        ledger = ledger.loc[ledger["status"].isin(accepted_statuses)].copy()
        if ledger.empty:
            raise ValueError("No completed benchmark rows available for provisional analysis.")

    rows: list[dict[str, object]] = []
    verification: dict[str, object] = {
        "expected_rows": expected,
        "completed_rows": completed_rows,
        "allow_incomplete": bool(cfg.allow_incomplete),
        "split_hashes": {},
        "manifest_paths": {},
        "selection_protocols": {},
        "checks": {
            "all_status_completed": bool(not cfg.allow_incomplete and incomplete.empty),
            "per_realization_hash_consistent": True,
            "per_model_manifest_consistent": True,
            "metrics_files_present": True,
            "deep_model_validation_pr_auc_checkpoint_selection": True,
            "classical_baselines_single_fit_no_checkpoint": True,
            "preprocessing_fit_on_train_only": True,
        },
    }
    split_manifest_ref = json.loads(cfg.split_manifest.read_text(encoding="utf-8"))
    verification["fixed_split_manifest"] = split_manifest_ref

    for row in ledger.itertuples(index=False):
        output_dir = Path(row.output_dir)
        metrics_dir = output_dir / "metrics"
        ensure_exists(
            [
                metrics_dir / "metrics.json",
                metrics_dir / "run_config.json",
                metrics_dir / "split_assignments.csv",
                metrics_dir / "test_predictions.csv",
                metrics_dir / "validation_predictions.csv",
                metrics_dir / "pr_curve_test.csv",
                metrics_dir / "roc_curve_test.csv",
            ]
        )
        metrics = json.loads((metrics_dir / "metrics.json").read_text(encoding="utf-8"))
        run_config = json.loads((metrics_dir / "run_config.json").read_text(encoding="utf-8"))
        split_hash = sha256_file(metrics_dir / "split_assignments.csv")
        realization_key = f"realization_{int(row.realization):02d}"
        verification["split_hashes"].setdefault(realization_key, []).append(
            {"algorithm": row.algorithm, "hash": split_hash}
        )
        verification["manifest_paths"].setdefault(realization_key, []).append(
            str(run_config["split_protocol"]["manifest_path"])
        )
        split_protocol = run_config["split_protocol"]
        if split_protocol["mode"] != "spatial_block":
            raise ValueError(f"{output_dir} is not using spatial_block split.")
        if split_protocol["split_seed"] != SEED:
            raise ValueError(f"{output_dir} split seed drifted from {SEED}.")
        try:
            selection_protocol = validate_selection_protocol(row.algorithm, run_config, metrics)
        except ValueError:
            check_name = (
                "deep_model_validation_pr_auc_checkpoint_selection"
                if row.algorithm in DEEP_MODEL_ALGORITHMS
                else "classical_baselines_single_fit_no_checkpoint"
            )
            verification["checks"][check_name] = False
            raise
        verification["selection_protocols"][row.algorithm] = selection_protocol
        if run_config.get("preprocessing_fit_partition") != "train":
            verification["checks"]["preprocessing_fit_on_train_only"] = False
            raise ValueError(f"{output_dir} preprocessing was not fitted on the training partition only.")

        test_metrics = metrics["test_metrics"]
        test_predictions_path = metrics_dir / "test_predictions.csv"
        predictions = pd.read_csv(test_predictions_path)
        topk_metrics = compute_topk_metrics(
            predictions["label"].to_numpy(dtype=int),
            predictions["probability"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "realization": int(row.realization),
                "realization_name": realization_key,
                "algorithm": row.algorithm,
                "model_label": MODEL_LABELS[row.algorithm],
                "duration_seconds": (
                    math.nan if row.status == "skipped_complete" else float(row.duration_seconds)
                ),
                "split_hash": split_hash,
                "split_mode": split_protocol["mode"],
                "split_seed": int(split_protocol["split_seed"]),
                "val_roc_auc": float(metrics["validation_metrics"]["roc_auc"]),
                "val_pr_auc": float(metrics["validation_metrics"]["pr_auc"]),
                "test_n": int(test_metrics["n_samples"]),
                "positive_ratio": float(test_metrics["positive_ratio"]),
                "accuracy": float(test_metrics["accuracy"]),
                "precision": float(test_metrics["precision"]),
                "recall": float(test_metrics["recall"]),
                "f1": float(test_metrics["f1"]),
                "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
                "mcc": float(test_metrics["mcc"]),
                "roc_auc": float(test_metrics["roc_auc"]),
                "pr_auc": float(test_metrics["pr_auc"]),
                "brier_score": float(test_metrics["brier_score"]),
                "log_loss": float(test_metrics["log_loss"]),
                "output_dir": str(output_dir),
                "test_predictions_path": str(test_predictions_path),
                "pr_curve_path": str(metrics_dir / "pr_curve_test.csv"),
                "roc_curve_path": str(metrics_dir / "roc_curve_test.csv"),
                **topk_metrics,
            }
        )

    for realization_key, items in verification["split_hashes"].items():
        unique_hashes = sorted({item["hash"] for item in items})
        if len(unique_hashes) != 1:
            verification["checks"]["per_realization_hash_consistent"] = False
            raise ValueError(f"Split assignment mismatch within {realization_key}: {unique_hashes}")
        verification["split_hashes"][realization_key] = unique_hashes[0]

    for realization_key, manifest_paths in verification["manifest_paths"].items():
        unique_paths = sorted(set(manifest_paths))
        if len(unique_paths) != 1:
            verification["checks"]["per_model_manifest_consistent"] = False
            raise ValueError(f"Manifest path mismatch within {realization_key}: {unique_paths}")
        verification["manifest_paths"][realization_key] = unique_paths[0]

    frame = pd.DataFrame(rows).sort_values(["realization", "algorithm"]).reset_index(drop=True)
    return frame, verification


def summarize_metrics(metrics_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    topk_columns = [
        f"top_{int(fraction * 100):02d}pct_{suffix}"
        for fraction in TOPK_FRACTIONS
        for suffix in ("capture", "precision")
    ]
    for algorithm in MODEL_ORDER:
        subset = metrics_long.loc[metrics_long["algorithm"] == algorithm]
        if subset.empty:
            continue
        row: dict[str, object] = {
            "algorithm": algorithm,
            "model_label": MODEL_LABELS[algorithm],
            "n_realizations": int(len(subset)),
            "mean_duration_seconds": float(subset["duration_seconds"].mean()),
        }
        for metric in ALL_METRICS:
            values = subset[metric].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_mean_ci(values, seed=SEED + len(rows) * 17 + len(metric))
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q1"] = float(np.quantile(values, 0.25))
            row[f"{metric}_q3"] = float(np.quantile(values, 0.75))
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        for metric in topk_columns:
            values = subset[metric].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_mean_ci(values, seed=SEED + len(rows) * 23 + len(metric))
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q1"] = float(np.quantile(values, 0.25))
            row[f"{metric}_q3"] = float(np.quantile(values, 0.75))
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def assumption_checks(metrics_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for algorithm in MODEL_ORDER:
        subset = metrics_long.loc[metrics_long["algorithm"] == algorithm]
        if subset.empty:
            continue
        values = subset[PRIMARY_METRIC].to_numpy(dtype=float)
        if values.size < 3:
            shapiro_stat, shapiro_p = math.nan, math.nan
        else:
            shapiro_stat, shapiro_p = stats.shapiro(values)
        rows.append(
            {
                "algorithm": algorithm,
                "model_label": MODEL_LABELS[algorithm],
                "metric": PRIMARY_METRIC,
                "n": int(values.size),
                "shapiro_w": float(shapiro_stat),
                "shapiro_p": float(shapiro_p),
            }
        )
    return pd.DataFrame(rows)


def paired_tests(metrics_long: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    pivot = metrics_long.pivot(index="realization", columns="algorithm", values=PRIMARY_METRIC)
    available_models = [algorithm for algorithm in MODEL_ORDER if algorithm in pivot.columns]
    if "pinn" not in available_models or len(available_models) < 2:
        return pd.DataFrame(), {
            "friedman_statistic": math.nan,
            "friedman_p": math.nan,
            "n_models": int(len(available_models)),
            "n_realizations": int(pivot.shape[0]),
        }
    matched_pivot = pivot[available_models].dropna(axis=0, how="any")
    if len(available_models) >= 3 and matched_pivot.shape[0] >= 2:
        friedman_stat, friedman_p = stats.friedmanchisquare(
            *[matched_pivot[col].to_numpy(dtype=float) for col in matched_pivot.columns]
        )
    else:
        friedman_stat, friedman_p = math.nan, math.nan

    baseline_algorithms = [algo for algo in available_models if algo != "pinn"]
    raw_rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for algorithm in baseline_algorithms:
        pair = pivot.loc[:, ["pinn", algorithm]].dropna(axis=0, how="any")
        diff = pair["pinn"].to_numpy(dtype=float) - pair[algorithm].to_numpy(dtype=float)
        if diff.size == 0:
            statistic = math.nan
            p_value = math.nan
        elif np.allclose(diff, 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            test = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        raw_p_values.append(p_value)
        raw_rows.append(
            {
                "target": "GINN",
                "baseline": MODEL_LABELS[algorithm],
                "baseline_algorithm": algorithm,
                "metric": PRIMARY_METRIC,
                "n": int(diff.size),
                "mean_diff": float(diff.mean()) if diff.size else math.nan,
                "median_diff": float(np.median(diff)) if diff.size else math.nan,
                "q1_diff": float(np.quantile(diff, 0.25)) if diff.size else math.nan,
                "q3_diff": float(np.quantile(diff, 0.75)) if diff.size else math.nan,
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
                "rank_biserial": float(rank_biserial_from_differences(diff)) if diff.size else math.nan,
            }
        )
    finite_p_values = [value for value in raw_p_values if np.isfinite(value)]
    adjusted_lookup: dict[int, float] = {}
    if finite_p_values:
        adjusted_values = holm_adjust(finite_p_values)
        adjusted_iter = iter(adjusted_values)
        for idx, value in enumerate(raw_p_values):
            if np.isfinite(value):
                adjusted_lookup[idx] = float(next(adjusted_iter))
    for idx, row in enumerate(raw_rows):
        adjusted_p = adjusted_lookup.get(idx, math.nan)
        row["p_holm"] = adjusted_p
        row["significant_at_0_05"] = bool(np.isfinite(adjusted_p) and adjusted_p < 0.05)
    return pd.DataFrame(raw_rows), {
        "friedman_statistic": float(friedman_stat),
        "friedman_p": float(friedman_p),
        "n_models": int(len(available_models)),
        "n_realizations": int(matched_pivot.shape[0]),
    }


def plot_metric_distributions(metrics_long: pd.DataFrame, output_stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.95), layout="constrained")
    rng = np.random.default_rng(SEED)
    metrics = [("pr_auc", "Test PR-AUC"), ("roc_auc", "Test ROC-AUC")]
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        axis.set_title(label)
        data = [metrics_long.loc[metrics_long["algorithm"] == algo, metric].to_numpy(dtype=float) for algo in MODEL_ORDER]
        box = axis.boxplot(
            data,
            positions=np.arange(1, len(MODEL_ORDER) + 1),
            widths=0.55,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.1},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        for patch, algo in zip(box["boxes"], MODEL_ORDER, strict=True):
            patch.set(facecolor=MODEL_COLORS[algo], alpha=0.85, edgecolor=MODEL_COLORS[algo])
        for idx, algo in enumerate(MODEL_ORDER, start=1):
            values = metrics_long.loc[metrics_long["algorithm"] == algo, metric].to_numpy(dtype=float)
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            axis.scatter(
                np.full(values.shape, idx, dtype=float) + jitter,
                values,
                s=18,
                marker=MODEL_MARKERS[algo],
                c="white",
                edgecolors=MODEL_COLORS[algo],
                linewidths=0.8,
                zorder=3,
            )
        axis.set_xticks(np.arange(1, len(MODEL_ORDER) + 1))
        axis.set_xticklabels([MODEL_LABELS[algo] for algo in MODEL_ORDER], rotation=32, ha="right")
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Metric value")
    axes[0].set_ylim(0.0, max(0.9, float(metrics_long["pr_auc"].max()) * 1.08))
    axes[1].set_ylim(0.5, 1.0)
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_pinn_forest(metrics_long: pd.DataFrame, paired_df: pd.DataFrame, output_stem: Path) -> None:
    if paired_df.empty or "baseline_algorithm" not in paired_df.columns:
        return
    pivot = metrics_long.pivot(index="realization", columns="algorithm", values=PRIMARY_METRIC)
    baselines = paired_df["baseline_algorithm"].tolist()
    fig, ax = plt.subplots(figsize=(7.48, 3.7), layout="constrained")
    y_positions = np.arange(len(baselines))[::-1]
    for y_pos, algorithm in zip(y_positions, baselines, strict=True):
        diff = pivot["pinn"].to_numpy(dtype=float) - pivot[algorithm].to_numpy(dtype=float)
        rng = np.random.default_rng(SEED + y_pos)
        boot = rng.choice(diff, size=(BOOTSTRAP_ITERATIONS, diff.size), replace=True).mean(axis=1)
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        ax.hlines(y_pos, ci_low, ci_high, color=MODEL_COLORS[algorithm], linewidth=2.0)
        ax.scatter(diff, np.full(diff.shape, y_pos, dtype=float), color=MODEL_COLORS[algorithm], s=16, alpha=0.55, marker=MODEL_MARKERS[algorithm])
        ax.scatter(diff.mean(), y_pos, color="white", edgecolors=MODEL_COLORS[algorithm], s=42, linewidths=1.2, zorder=4)
    ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_yticks(y_positions)
    labels = []
    for algorithm in baselines:
        row = paired_df.loc[paired_df["baseline_algorithm"] == algorithm].iloc[0]
        labels.append(f"{MODEL_LABELS[algorithm]}  (pHolm={row['p_holm']:.3f})")
    ax.set_yticklabels(labels)
    ax.set_xlabel("GINN - baseline PR-AUC")
    ax.set_title("Pairwise realization-level PR-AUC differences against GINN")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_mean_curves(metrics_long: pd.DataFrame, output_stem: Path) -> None:
    pr_grid = np.linspace(0.0, 1.0, 301)
    roc_grid = np.linspace(0.0, 1.0, 301)
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.95), layout="constrained")

    for algorithm in MODEL_ORDER:
        subset = metrics_long.loc[metrics_long["algorithm"] == algorithm]
        if subset.empty:
            continue
        pr_curves = np.vstack(
            [
                read_curve(Path(path), x_col="recall", y_col="precision", grid=pr_grid)
                for path in subset["pr_curve_path"]
            ]
        )
        roc_curves = np.vstack(
            [
                read_curve(Path(path), x_col="fpr", y_col="tpr", grid=roc_grid)
                for path in subset["roc_curve_path"]
            ]
        )
        pr_mean = pr_curves.mean(axis=0)
        pr_std = pr_curves.std(axis=0, ddof=1) if pr_curves.shape[0] > 1 else np.zeros_like(pr_mean)
        roc_mean = roc_curves.mean(axis=0)
        roc_std = roc_curves.std(axis=0, ddof=1) if roc_curves.shape[0] > 1 else np.zeros_like(roc_mean)
        style = {
            "color": MODEL_COLORS[algorithm],
            "linewidth": 1.6 if algorithm == "pinn" else 1.3,
            "label": MODEL_LABELS[algorithm],
        }
        axes[0].plot(pr_grid, pr_mean, **style)
        axes[0].fill_between(
            pr_grid,
            np.clip(pr_mean - pr_std, 0.0, 1.0),
            np.clip(pr_mean + pr_std, 0.0, 1.0),
            color=MODEL_COLORS[algorithm],
            alpha=0.10,
        )
        axes[1].plot(roc_grid, roc_mean, **style)
        axes[1].fill_between(
            roc_grid,
            np.clip(roc_mean - roc_std, 0.0, 1.0),
            np.clip(roc_mean + roc_std, 0.0, 1.0),
            color=MODEL_COLORS[algorithm],
            alpha=0.10,
        )

    axes[0].set_title("Mean test PR curves across realizations")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25, linewidth=0.5)

    axes[1].set_title("Mean test ROC curves across realizations")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25, linewidth=0.5)
    axes[1].plot([0, 1], [0, 1], color="#666666", linewidth=0.9, linestyle="--")
    axes[1].legend(loc="lower right", frameon=False, ncol=1)
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_cumulative_gains(metrics_long: pd.DataFrame, output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.48, 3.7), layout="constrained")
    grid = np.linspace(0.0, 1.0, 101)
    for algorithm in MODEL_ORDER:
        subset = metrics_long.loc[metrics_long["algorithm"] == algorithm]
        if subset.empty:
            continue
        curves: list[np.ndarray] = []
        for path in subset["test_predictions_path"]:
            frame = pd.read_csv(path)
            curves.append(
                cumulative_gain_curve(
                    frame["label"].to_numpy(dtype=int),
                    frame["probability"].to_numpy(dtype=float),
                    grid,
                )
            )
        matrix = np.vstack(curves)
        mean_curve = matrix.mean(axis=0)
        std_curve = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(mean_curve)
        ax.plot(grid, mean_curve, color=MODEL_COLORS[algorithm], linewidth=1.6, label=MODEL_LABELS[algorithm])
        ax.fill_between(
            grid,
            np.clip(mean_curve - std_curve, 0.0, 1.0),
            np.clip(mean_curve + std_curve, 0.0, 1.0),
            color=MODEL_COLORS[algorithm],
            alpha=0.10,
            linewidth=0.0,
        )
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#777777", linestyle="--", linewidth=0.9, label="Random ranking")
    ax.set_xlabel("Cumulative fraction of sampled test cells")
    ax.set_ylabel("Cumulative positive capture rate")
    ax.set_title("Success-rate / cumulative gains curves")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right", ncol=2, frameon=False)
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_pinn_mlp_ablation(metrics_long: pd.DataFrame, output_stem: Path) -> None:
    paired = metrics_long.loc[metrics_long["algorithm"].isin(["pinn", "mlp"])].copy()
    if paired["algorithm"].nunique() < 2:
        return
    pivot = paired.pivot(index="realization", columns="algorithm", values=["pr_auc", "brier_score"])
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.55), layout="constrained")
    for axis, metric, ylabel, better in [
        (axes[0], "pr_auc", "Test PR-AUC", "Higher is better"),
        (axes[1], "brier_score", "Test Brier score", "Lower is better"),
    ]:
        for realization in pivot.index:
            x = [0, 1]
            y = [
                float(pivot.loc[realization, (metric, "mlp")]),
                float(pivot.loc[realization, (metric, "pinn")]),
            ]
            axis.plot(x, y, color="#BBBBBB", linewidth=0.8, zorder=1)
            axis.scatter(
                x,
                y,
                s=28,
                c=[MODEL_COLORS["mlp"], MODEL_COLORS["pinn"]],
                marker="o",
                edgecolors="white",
                linewidths=0.6,
                zorder=2,
            )
        means = [
            float(pivot[(metric, "mlp")].mean()),
            float(pivot[(metric, "pinn")].mean()),
        ]
        axis.scatter([0, 1], means, s=70, c=["white", "white"], edgecolors=[MODEL_COLORS["mlp"], MODEL_COLORS["pinn"]], linewidths=1.6, zorder=3)
        axis.set_xticks([0, 1])
        axis.set_xticklabels([MODEL_LABELS["mlp"], MODEL_LABELS["pinn"]])
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel}\n{better}")
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    save_figure(fig, output_stem)
    plt.close(fig)


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white", metadata={"Title": output_stem.name})
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, facecolor="white")


def write_analysis_report(
    cfg: AnalysisConfig,
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    assumption_df: pd.DataFrame,
    omnibus: dict[str, object],
    verification: dict[str, object],
) -> None:
    best = summary_df.sort_values(f"{PRIMARY_METRIC}_mean", ascending=False).reset_index(drop=True)
    top = best.iloc[0]
    second = best.iloc[1] if len(best) > 1 else None

    def metric_gap(target: str, baseline: str) -> float | None:
        left = summary_df.loc[summary_df["algorithm"] == target]
        right = summary_df.loc[summary_df["algorithm"] == baseline]
        if left.empty or right.empty:
            return None
        return float(left.iloc[0][f"{PRIMARY_METRIC}_mean"] - right.iloc[0][f"{PRIMARY_METRIC}_mean"])

    pr_gap_vs_mlp = metric_gap("pinn", "mlp")
    pr_gap_vs_gcn = metric_gap("pinn", "gcn_transformer")
    shapiro_series = assumption_df["shapiro_p"].dropna()
    all_shapiro_ok = bool((shapiro_series > 0.05).all()) if not shapiro_series.empty else False
    top5 = summary_df.sort_values("top_05pct_capture_mean", ascending=False).reset_index(drop=True)
    lines = [
        "# Reconstructed-negative benchmark analysis",
        "",
        "## Analysis question",
        "",
        "Do the unified 5x5 spatial-block experiments over 10 reconstructed negative-sample realizations support the proposed GINN over the new GCN-Transformer comparison and the remaining non-graph baselines?",
        "",
        "## Protocol locked before analysis",
        "",
        "- Unit of repetition: negative-sample realization (n = 10).",
        "- Shared split: fixed 5x5 spatial blocks, identical within each realization across all models.",
        "- Shared preprocessing: imputer, variance filter, scaler, and distance clipping fitted on training data only.",
        "- Compared models: GINN, MLP ablation, GCN-Transformer, RF, LightGBM, XGBoost, CatBoost.",
        "- Primary metric: test PR-AUC. Secondary metrics are exploratory.",
        "- GINN, MLP, and GCN-Transformer use validation PR-AUC for scheduling, early stopping, and checkpoint selection; the four tree ensembles are single-fit estimators without checkpoint selection.",
        "",
        "## Key findings",
        "",
        f"- Best mean PR-AUC: {top['model_label']} = {top['pr_auc_mean']:.4f} ± {top['pr_auc_std']:.4f} [95% bootstrap CI {top['pr_auc_ci95_low']:.4f}, {top['pr_auc_ci95_high']:.4f}].",
        (
            f"- Runner-up mean PR-AUC: {second['model_label']} = {second['pr_auc_mean']:.4f} ± {second['pr_auc_std']:.4f}."
            if second is not None
            else "- Runner-up mean PR-AUC: unavailable because fewer than two models have completed runs."
        ),
        (
            f"- GINN vs MLP ablation: ΔPR-AUC = {pr_gap_vs_mlp:+.4f}; this compares the geological-prior loss against the same backbone family."
            if pr_gap_vs_mlp is not None
            else "- GINN vs MLP ablation: pending until both models complete the current matrix."
        ),
        (
            f"- GINN vs GCN-Transformer: ΔPR-AUC = {pr_gap_vs_gcn:+.4f}; this directly answers the AE request for an advanced graph baseline."
            if pr_gap_vs_gcn is not None
            else "- GINN vs GCN-Transformer: pending until both models complete the current matrix."
        ),
        (
            f"- Omnibus PR-AUC difference across all available models: Friedman χ² = {omnibus['friedman_statistic']:.4f}, p = {omnibus['friedman_p']:.6f}."
            if not math.isnan(float(omnibus["friedman_statistic"]))
            else "- Omnibus PR-AUC test: pending because fewer than three models with matched realizations are currently complete."
        ),
        (
            f"- Shapiro checks for per-model PR-AUC distributions all non-significant: {'yes' if all_shapiro_ok else 'no'}; Wilcoxon/Friedman were retained because repeated realizations remain paired and small-sample."
            if not assumption_df.empty
            else "- Shapiro checks: pending because fewer than one model has enough completed realizations."
        ),
        (
            f"- Top-5% sampled-cell capture: {top5.iloc[0]['model_label']} captures {top5.iloc[0]['top_05pct_capture_mean']:.4f} of positives on average within the highest-scoring 5% of sampled test cells."
            if not top5.empty
            else "- Top-5% sampled-cell capture: pending."
        ),
        "",
        "## Verification highlights",
        "",
        f"- Completed matrix: {verification['completed_rows']} / {verification['expected_rows']} runs.",
        f"- Split hashes: {len(set(verification['split_hashes'].values()))} unique hashes across realizations, with exactly one hash per realization shared by all seven models.",
        "- Every analyzed run includes metrics, prediction files, PR/ROC curve CSVs, and a locked spatial-split manifest path.",
        "",
        "## Interpretation",
        "",
        "- This benchmark isolates uncertainty from negative-sample reconstruction while holding the spatial split and model seed fixed.",
        "- The MLP ablation is the cleanest archived test of whether the geological-prior loss adds value beyond network capacity alone, subject to the recorded scheduler difference.",
        "- The GCN-Transformer result should be interpreted as an inductive local spatial graph baseline, not as a full transductive regional graph trained on test nodes.",
        "",
        "## Limitations",
        "",
        "- All runs use one model seed so the repeated-measure unit is realization, not seed.",
        "- Secondary metrics are provided for context and should not replace PR-AUC as the main ranking criterion under class imbalance.",
        "- Threshold-free capture diagnostics are more MPM-relevant than the fixed 0.5 threshold, but they still inherit this study's reconstructed-negative protocol.",
        "- Statistical significance does not erase geological external-validity questions; manuscript claims must stay tied to this study area and this reconstruction protocol.",
    ]
    (cfg.output_dir / "analysis-report.md").write_text("\n".join(lines), encoding="utf-8")


def write_stats_appendix(
    cfg: AnalysisConfig,
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    assumption_df: pd.DataFrame,
    omnibus: dict[str, object],
) -> None:
    summary_cols = [
        "model_label",
        "n_realizations",
        "pr_auc_mean",
        "pr_auc_std",
        "pr_auc_median",
        "pr_auc_q1",
        "pr_auc_q3",
        "pr_auc_ci95_low",
        "pr_auc_ci95_high",
        "roc_auc_mean",
        "mcc_mean",
        "recall_mean",
        "brier_score_mean",
        "top_05pct_capture_mean",
        "top_05pct_precision_mean",
    ]
    paired_cols = [
        "baseline",
        "mean_diff",
        "median_diff",
        "q1_diff",
        "q3_diff",
        "wilcoxon_statistic",
        "p_raw",
        "p_holm",
        "rank_biserial",
    ]
    assumption_cols = ["model_label", "n", "shapiro_w", "shapiro_p"]
    lines = [
        "# Benchmark statistics appendix",
        "",
        "## Descriptive statistics by model",
        "",
        markdown_table(summary_df, summary_cols),
        "",
        "## Omnibus test on PR-AUC",
        "",
        f"- Friedman chi-square = {omnibus['friedman_statistic']:.6f}",
        f"- p-value = {omnibus['friedman_p']:.6f}",
        f"- Models = {omnibus['n_models']}, realizations = {omnibus['n_realizations']}",
        "",
        "## Pairwise Wilcoxon tests against GINN (primary metric only)",
        "",
        markdown_table(paired_df, paired_cols),
        "",
        "## PR-AUC assumption checks",
        "",
        markdown_table(assumption_df, assumption_cols),
        "",
        "## Interpretation notes",
        "",
        "- Positive mean_diff means GINN outperformed the baseline on PR-AUC.",
        "- Holm-adjusted p-values control family-wise error across the six planned GINN-vs-baseline contrasts.",
        "- Rank-biserial > 0 favors GINN; magnitude reflects paired dominance rather than raw score scale.",
    ]
    (cfg.output_dir / "stats-appendix.md").write_text("\n".join(lines), encoding="utf-8")


def write_figure_catalog(cfg: AnalysisConfig) -> None:
    lines = [
        "# Figure catalog",
        "",
        "| Figure | Files | Purpose | What the reader should notice |",
        "| --- | --- | --- | --- |",
        "| QA Figure 1 | `figures/figure_01_metric_distributions.pdf`, `figures/figure_01_metric_distributions.png` | Show realization-level variability and central tendency for PR-AUC and ROC-AUC across all models. | Whether ranking is stable across the 10 reconstructed negative-sample realizations rather than driven by one favorable run. |",
        "| QA Figure 2 | `figures/figure_02_pinn_pairwise_forest.pdf`, `figures/figure_02_pinn_pairwise_forest.png` | Quantify paired PR-AUC differences between GINN and each baseline. | Whether GINN consistently beats the MLP ablation and the GCN-Transformer, and by how much. |",
        "| QA Figure 3 | `figures/figure_03_mean_curves.pdf`, `figures/figure_03_mean_curves.png` | Compare mean PR and ROC curves with one-standard-deviation ribbons. | How ranking changes across operating thresholds, not only at scalar summary metrics. |",
        "| QA Figure 4 | `figures/figure_04_pinn_mlp_ablation.pdf`, `figures/figure_04_pinn_mlp_ablation.png` | Compare the geological-prior loss by pairing GINN with the matched MLP backbone. | Whether the implemented prior loss improves precision-recall performance and calibration jointly rather than by chance in one split. |",
        "| QA Figure 5 | `figures/figure_05_cumulative_gains.pdf`, `figures/figure_05_cumulative_gains.png` | Show cumulative positive capture as the fraction of sampled test cells expands from the highest-scoring cells downward. | Which model captures ore-related positives fastest among the top-ranked sampled cells, a more decision-relevant diagnostic than a fixed 0.5 threshold. |",
    ]
    (cfg.output_dir / "figure-catalog.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_long, verification = collect_metrics(cfg)
    summary_df = summarize_metrics(metrics_long)
    assumption_df = assumption_checks(metrics_long)
    paired_df, omnibus = paired_tests(metrics_long)

    metrics_long.to_csv(cfg.metrics_long_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(cfg.summary_csv, index=False, encoding="utf-8-sig")
    paired_df.to_csv(cfg.paired_tests_csv, index=False, encoding="utf-8-sig")
    assumption_df.to_csv(cfg.assumption_checks_csv, index=False, encoding="utf-8-sig")
    with cfg.verification_json.open("w", encoding="utf-8") as stream:
        json.dump(verification, stream, ensure_ascii=False, indent=2)

    with plt.rc_context(PLOT_STYLE):
        plot_metric_distributions(metrics_long, cfg.figures_dir / "figure_01_metric_distributions")
        plot_pinn_forest(metrics_long, paired_df, cfg.figures_dir / "figure_02_pinn_pairwise_forest")
        plot_mean_curves(metrics_long, cfg.figures_dir / "figure_03_mean_curves")
        plot_pinn_mlp_ablation(metrics_long, cfg.figures_dir / "figure_04_pinn_mlp_ablation")
        plot_cumulative_gains(metrics_long, cfg.figures_dir / "figure_05_cumulative_gains")

    write_analysis_report(cfg, summary_df, paired_df, assumption_df, omnibus, verification)
    write_stats_appendix(cfg, summary_df, paired_df, assumption_df, omnibus)
    write_figure_catalog(cfg)


if __name__ == "__main__":
    main()
