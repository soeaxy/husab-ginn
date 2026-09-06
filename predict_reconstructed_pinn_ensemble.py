from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rasterio
import torch

from mineral_deep_models import resolve_torch_device
from physics_informed_model import GeologyInformedClassifier
from predict_multi_physics_model import (
    build_feature_stack,
    discover_tif_files,
    load_research_shapes,
    predict_probabilities,
    preprocess_features,
)


def parse_csv_ints(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values:
        raise argparse.ArgumentTypeError("At least one realization is required.")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create mean and uncertainty rasters from the 10 reconstructed-sample GINN models."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=root / "experiments" / "reconstructed_v2_spatial_prselected",
    )
    parser.add_argument("--realizations", type=parse_csv_ints, default=list(range(10)))
    parser.add_argument("--model-seed", type=int, default=2026)
    parser.add_argument("--feature-dir", type=Path, default=root / "data" / "factors")
    parser.add_argument(
        "--research-shp",
        type=Path,
        default=root / "data" / "research" / "Husab_Square.shp",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "analysis_output" / "reconstructed_v2_pinn_ensemble_prselected",
    )
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--write-members", action="store_true")
    return parser


def model_paths(experiment_root: Path, realization: int, model_seed: int) -> tuple[Path, Path, Path]:
    run_root = (
        experiment_root
        / f"realization_{realization:02d}"
        / "pinn"
        / f"seed_{model_seed}"
    )
    return (
        run_root / "models" / "mineral_model_multi_phy_final.pth",
        run_root / "models" / "preproc.joblib",
        run_root / "metrics" / "metrics.json",
    )


def write_probability_raster(
    path: Path,
    values: np.ndarray,
    flat_valid: np.ndarray,
    profile: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.full(flat_valid.size, np.nan, dtype=np.float32)
    output[flat_valid] = values.astype(np.float32, copy=False)
    output = output.reshape(profile["height"], profile["width"])
    output_profile = profile.copy()
    output_profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan, compress="deflate")
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(output, 1)


def update_online_moments(
    mean: np.ndarray | None,
    m2: np.ndarray | None,
    values: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    current = values.astype(np.float64, copy=False)
    if mean is None or m2 is None:
        return current.copy(), np.zeros_like(current)
    delta = current - mean
    mean = mean + delta / count
    m2 = m2 + delta * (current - mean)
    return mean, m2


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected GINN checkpoint payload: {path}")
    return state


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_torch_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_files: list[tuple[int, Path, Path, Path]] = []
    for realization in args.realizations:
        model_path, preproc_path, metrics_path = model_paths(
            args.experiment_root, realization, args.model_seed
        )
        missing = [path for path in (model_path, preproc_path, metrics_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete GINN run for realization {realization:02d}: {missing}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("selection_metric") != "pr_auc":
            raise ValueError(f"Realization {realization:02d} was not selected by validation PR-AUC.")
        run_files.append((realization, model_path, preproc_path, metrics_path))

    first_preproc = joblib.load(run_files[0][2])
    canonical_features = list(first_preproc.get("all_feature_names", []))
    if not canonical_features:
        raise ValueError("The first preprocessing artifact has no all_feature_names metadata.")

    feature_map = discover_tif_files(args.feature_dir)
    shapes = load_research_shapes(args.research_shp)
    x_raw, flat_valid, profile, _, _ = build_feature_stack(
        required_feature_names=canonical_features,
        feature_map=feature_map,
        shapes=shapes,
    )

    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    member_records: list[dict[str, Any]] = []
    for count, (realization, model_path, preproc_path, metrics_path) in enumerate(run_files, start=1):
        preproc = joblib.load(preproc_path)
        if str(preproc.get("algorithm", "pinn")).lower() != "pinn":
            raise ValueError(f"Unexpected algorithm metadata in {preproc_path}")
        if list(preproc.get("all_feature_names", [])) != canonical_features:
            raise ValueError(f"Feature order drifted in {preproc_path}")
        kept_features = list(preproc.get("kept_feature_names", []))
        x_processed = preprocess_features(x_raw, preproc)
        if x_processed.shape[1] != len(kept_features):
            raise ValueError(f"Processed feature count does not match metadata in {preproc_path}")

        model = GeologyInformedClassifier(input_dim=x_processed.shape[1]).to(device)
        model.load_state_dict(load_state_dict(model_path, device))
        probabilities = predict_probabilities(model, x_processed, args.batch_size, device)
        mean, m2 = update_online_moments(mean, m2, probabilities, count)

        member_path: Path | None = None
        if args.write_members:
            member_path = args.output_dir / f"pinn_probability_realization_{realization:02d}.tif"
            write_probability_raster(member_path, probabilities, flat_valid, profile)
        member_records.append(
            {
                "realization": realization,
                "model_path": str(model_path.resolve()),
                "model_sha256": sha256_file(model_path),
                "preproc_path": str(preproc_path.resolve()),
                "preproc_sha256": sha256_file(preproc_path),
                "metrics_path": str(metrics_path.resolve()),
                "prediction_min": float(probabilities.min()),
                "prediction_max": float(probabilities.max()),
                "prediction_mean": float(probabilities.mean()),
                "member_raster": str(member_path.resolve()) if member_path else None,
            }
        )

    assert mean is not None and m2 is not None
    std = np.sqrt(m2 / (len(run_files) - 1)) if len(run_files) > 1 else np.zeros_like(mean)
    mean_path = args.output_dir / "pinn_ensemble_mean.tif"
    std_path = args.output_dir / "pinn_ensemble_std.tif"
    write_probability_raster(mean_path, mean, flat_valid, profile)
    write_probability_raster(std_path, std, flat_valid, profile)

    manifest = {
        "schema_version": 1,
        "manuscript_model_name": "GINN",
        "internal_algorithm_key": "pinn",
        "legacy_output_prefix_retained": True,
        "experiment_root": str(args.experiment_root.resolve()),
        "selection_metric": "pr_auc",
        "model_seed": args.model_seed,
        "realizations": args.realizations,
        "n_members": len(run_files),
        "uncertainty_scope": "pseudo-absence realization uncertainty conditional on one model seed and one fixed spatial split",
        "feature_names": canonical_features,
        "valid_cells": int(flat_valid.sum()),
        "ensemble_mean_path": str(mean_path.resolve()),
        "ensemble_mean_sha256": sha256_file(mean_path),
        "ensemble_std_path": str(std_path.resolve()),
        "ensemble_std_sha256": sha256_file(std_path),
        "ensemble_mean_range": [float(mean.min()), float(mean.max())],
        "ensemble_std_range": [float(std.min()), float(std.max())],
        "members": member_records,
    }
    (args.output_dir / "ensemble_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"mean": str(mean_path), "std": str(std_path), "members": len(run_files)}))


if __name__ == "__main__":
    main()
