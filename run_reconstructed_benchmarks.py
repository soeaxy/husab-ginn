from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from manuscript_protocol import (
    BENCHMARK_ALGORITHMS,
    MODEL_SEED,
    SPATIAL_GRID_SIZE,
    manuscript_model_label,
    normalize_algorithm_key,
)
from train_multi_physics_model import grouped_holdout_indices, spatial_block_groups

DEFAULT_ALGORITHMS = BENCHMARK_ALGORITHMS
DEEP_ALGORITHMS = {"pinn", "mlp", "gcn_transformer"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def realization_paths(data_root: Path, realization_ids: list[int]) -> list[Path]:
    paths = [
        data_root / f"realization_{realization_id:02d}" / "combined_samples_rebuilt.shp"
        for realization_id in realization_ids
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing reconstructed sample files: {missing}")
    return paths


def build_fixed_split_manifest(
    sample_paths: list[Path],
    output_path: Path,
    grid_size: int,
    split_seed: int,
    test_size: float,
    val_size: float,
) -> dict[str, Any]:
    bounds_rows: list[tuple[float, float, float, float]] = []
    populated_blocks: set[int] = set()
    frames: list[gpd.GeoDataFrame] = []
    for path in sample_paths:
        frame = gpd.read_file(path)
        if frame.empty or "Class" not in frame.columns:
            raise ValueError(f"Invalid training sample file: {path}")
        frames.append(frame)
        bounds_rows.append(tuple(float(value) for value in frame.total_bounds))

    bounds_array = np.asarray(bounds_rows, dtype=np.float64)
    bounds = (
        float(bounds_array[:, 0].min()),
        float(bounds_array[:, 1].min()),
        float(bounds_array[:, 2].max()),
        float(bounds_array[:, 3].max()),
    )
    for frame in frames:
        coords = np.column_stack([frame.geometry.x.to_numpy(), frame.geometry.y.to_numpy()])
        populated_blocks.update(int(value) for value in np.unique(spatial_block_groups(coords, grid_size, bounds)))

    canonical = frames[0]
    coords = np.column_stack([canonical.geometry.x.to_numpy(), canonical.geometry.y.to_numpy()])
    labels = canonical["Class"].to_numpy(dtype=np.int64)
    groups = spatial_block_groups(coords, grid_size, bounds)
    indices = np.arange(labels.size)
    train_indices, test_indices = grouped_holdout_indices(
        indices, labels, groups, test_size, split_seed
    )
    train_indices, val_indices = grouped_holdout_indices(
        train_indices,
        labels[train_indices],
        groups[train_indices],
        val_size,
        split_seed + 1,
    )
    partitions = {
        "train": sorted(int(value) for value in np.unique(groups[train_indices])),
        "validation": sorted(int(value) for value in np.unique(groups[val_indices])),
        "test": sorted(int(value) for value in np.unique(groups[test_indices])),
    }
    assigned = set(partitions["train"]) | set(partitions["validation"]) | set(partitions["test"])
    partitions["train"].extend(sorted(populated_blocks - assigned))
    partitions["train"] = sorted(set(partitions["train"]))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "split_mode": "spatial_block",
        "grid_size": int(grid_size),
        "bounds": {
            "min_x": bounds[0],
            "min_y": bounds[1],
            "max_x": bounds[2],
            "max_y": bounds[3],
        },
        "partitions": partitions,
        "split_seed": int(split_seed),
        "canonical_realization": str(sample_paths[0].resolve()),
        "canonical_counts": {
            "train": int(train_indices.size),
            "validation": int(val_indices.size),
            "test": int(test_indices.size),
            "train_positive": int(labels[train_indices].sum()),
            "validation_positive": int(labels[val_indices].sum()),
            "test_positive": int(labels[test_indices].sum()),
        },
        "input_shapefile_sha256": {
            str(path.resolve()): sha256_file(path)
            for path in sample_paths
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_command(
    *,
    train_script: Path,
    sample_path: Path,
    feature_dir: Path,
    prior_dir: Path,
    output_dir: Path,
    split_manifest: Path,
    algorithm: str,
    model_seed: int,
    split_seed: int,
    grid_size: int,
    deep_epochs: int,
    patience: int,
    physics_weight: float,
    selection_metric: str,
    device: str,
) -> list[str]:
    epochs = deep_epochs if algorithm in DEEP_ALGORITHMS else 1
    command = [
        sys.executable,
        str(train_script),
        "--shapefile",
        str(sample_path),
        "--feature-dir",
        str(feature_dir),
        "--output-root",
        str(output_dir),
        "--algorithm",
        algorithm,
        "--optimizer",
        "none",
        "--seed",
        str(model_seed),
        "--split-seed",
        str(split_seed),
        "--split-mode",
        "spatial_block",
        "--spatial-block-grid",
        str(grid_size),
        "--split-manifest",
        str(split_manifest),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--physics-weight",
        str(physics_weight),
        "--selection-metric",
        selection_metric,
        "--batch-size-cpu",
        "2048",
        "--batch-size-cuda",
        "4096",
        "--device",
        device,
        "--disable-shap",
    ]
    if algorithm == "pinn":
        command.extend(
            [
                "--dome-tif",
                str(prior_dir / "dome.tif"),
                "--fault-tif",
                str(prior_dir / "fault.tif"),
                "--strata-tif",
                str(prior_dir / "strata.tif"),
            ]
        )
    return command


def metrics_complete(output_dir: Path) -> bool:
    required = (
        output_dir / "metrics" / "metrics.json",
        output_dir / "metrics" / "run_config.json",
        output_dir / "metrics" / "split_assignments.csv",
        output_dir / "metrics" / "test_predictions.csv",
    )
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "realization",
        "algorithm",
        "model_seed",
        "status",
        "duration_seconds",
        "exit_code",
        "output_dir",
        "log_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_ints(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values:
        raise argparse.ArgumentTypeError("At least one realization is required.")
    return values


def parse_csv_strings(value: str) -> list[str]:
    values = [normalize_algorithm_key(item) for item in value.split(",") if item.strip()]
    unsupported = sorted(set(values) - set(DEFAULT_ALGORITHMS))
    if unsupported:
        raise argparse.ArgumentTypeError(f"Unsupported benchmark algorithms: {unsupported}")
    return values


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the fixed spatial benchmark on reconstructed negatives.")
    parser.add_argument("--data-root", type=Path, default=root / "data" / "reconstructed_samples_v2")
    parser.add_argument("--feature-dir", type=Path, default=root / "data" / "factors")
    parser.add_argument("--prior-dir", type=Path, default=root / "data" / "priors")
    parser.add_argument("--output-root", type=Path, default=root / "experiments" / "reconstructed_v2_spatial")
    parser.add_argument("--realizations", type=parse_csv_ints, default=list(range(10)))
    parser.add_argument("--algorithms", type=parse_csv_strings, default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--split-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--grid-size", type=int, default=SPATIAL_GRID_SIZE)
    parser.add_argument("--deep-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--geology-weight",
        "--physics-weight",
        dest="physics_weight",
        type=float,
        default=0.1,
        help="GINN geological-prior loss weight; --physics-weight is a legacy alias.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["roc_auc", "pr_auc"],
        default="pr_auc",
        help="Validation metric used to select deep-model checkpoints.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--timeout-minutes", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent
    train_script = root / "train_multi_physics_model.py"
    sample_paths = realization_paths(args.data_root, args.realizations)
    args.output_root.mkdir(parents=True, exist_ok=True)
    split_manifest = args.output_root / "fixed_spatial_split.json"
    split_payload = build_fixed_split_manifest(
        sample_paths,
        split_manifest,
        grid_size=args.grid_size,
        split_seed=args.split_seed,
        test_size=0.2,
        val_size=0.2,
    )

    benchmark_manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "train_script": str(train_script.resolve()),
        "train_script_sha256": sha256_file(train_script),
        "algorithms": args.algorithms,
        "manuscript_model_labels": {
            algorithm: manuscript_model_label(algorithm) for algorithm in args.algorithms
        },
        "realizations": args.realizations,
        "model_seed": args.model_seed,
        "split_seed": args.split_seed,
        "deep_epochs": args.deep_epochs,
        "patience": args.patience,
        "physics_weight": args.physics_weight,
        "selection_metric": args.selection_metric,
        "primary_metric": "test_pr_auc",
        "secondary_metrics": ["test_roc_auc", "test_mcc", "test_recall", "test_brier_score"],
        "comparison_unit": "negative_sample_realization",
        "split_manifest": split_payload,
    }
    (args.output_root / "benchmark_manifest.json").write_text(
        json.dumps(benchmark_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    ledger_path = args.output_root / "run_ledger.csv"
    for realization_id, sample_path in zip(args.realizations, sample_paths, strict=True):
        for algorithm in args.algorithms:
            output_dir = args.output_root / f"realization_{realization_id:02d}" / algorithm / f"seed_{args.model_seed}"
            log_path = output_dir / "training.log"
            row: dict[str, Any] = {
                "realization": realization_id,
                "algorithm": algorithm,
                "model_seed": args.model_seed,
                "status": "pending",
                "duration_seconds": 0.0,
                "exit_code": "",
                "output_dir": str(output_dir.resolve()),
                "log_path": str(log_path.resolve()),
            }
            if metrics_complete(output_dir):
                row["status"] = "skipped_complete"
                rows.append(row)
                write_ledger(ledger_path, rows)
                print(f"[skip] realization={realization_id:02d} algorithm={algorithm}", flush=True)
                continue

            command = build_command(
                train_script=train_script,
                sample_path=sample_path,
                feature_dir=args.feature_dir,
                prior_dir=args.prior_dir,
                output_dir=output_dir,
                split_manifest=split_manifest,
                algorithm=algorithm,
                model_seed=args.model_seed,
                split_seed=args.split_seed,
                grid_size=args.grid_size,
                deep_epochs=args.deep_epochs,
                patience=args.patience,
                physics_weight=args.physics_weight,
                selection_metric=args.selection_metric,
                device=args.device,
            )
            if args.dry_run:
                row["status"] = "dry_run"
                rows.append(row)
                write_ledger(ledger_path, rows)
                print(subprocess.list2cmdline(command), flush=True)
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            print(f"[start] realization={realization_id:02d} algorithm={algorithm}", flush=True)
            try:
                with log_path.open("w", encoding="utf-8") as log_handle:
                    completed = subprocess.run(
                        command,
                        cwd=root,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        timeout=max(60.0, args.timeout_minutes * 60.0),
                        check=False,
                    )
                row["exit_code"] = completed.returncode
                row["status"] = "completed" if completed.returncode == 0 and metrics_complete(output_dir) else "failed"
            except subprocess.TimeoutExpired:
                row["status"] = "timeout"
                row["exit_code"] = "timeout"
            row["duration_seconds"] = round(time.perf_counter() - started, 3)
            rows.append(row)
            write_ledger(ledger_path, rows)
            print(
                f"[{row['status']}] realization={realization_id:02d} algorithm={algorithm} "
                f"duration={row['duration_seconds']}s",
                flush=True,
            )
            if args.fail_fast and row["status"] != "completed":
                raise RuntimeError(f"Benchmark run failed: realization={realization_id:02d}, algorithm={algorithm}")

    failures = [row for row in rows if row["status"] in {"failed", "timeout"}]
    print(f"Benchmark matrix finished: {len(rows)} rows, {len(failures)} failures.", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
