"""Train-only sampling-design sensitivity with byte-stable held-out identities.

This separate runner never edits the locked 72-run OGR experiment archive.
It reuses existing candidate coordinates but evaluates eligibility only inside
the fixed training blocks. No candidate-grid Class values are read or reused.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely import contains_xy, distance, points
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from rebuild_negative_samples import (
    SamplingConfig, assign_strata, compute_quotas, load_polygon_union,
    prior_from_distance, raster_valid_mask, sample_raster_values,
    select_spatially_balanced, shapefile_hashes, spatial_cell_ids, write_shapefile,
)
from run_reconstructed_benchmarks import build_command, metrics_complete, sha256_file
from train_multi_physics_model import load_spatial_split_manifest, spatial_block_groups


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiments" / "ogr_revision_sampling_design_20260905"
ANALYSIS = ROOT / "analysis_output" / "ogr_revision_sampling_design_20260905"
ARCHIVE = ROOT / "experiments" / "reconstructed_v2_spatial_prselected"
SPLIT = ARCHIVE / "fixed_spatial_split.json"
BASE = ROOT / "data" / "reconstructed_samples_v2" / "realization_00" / "combined_samples_rebuilt.shp"
VARIANTS = ("uniform_eligible", "mixture_65_5_30")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_pool() -> tuple[pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    protocol = json.loads((BASE.parent.parent / "protocol.json").read_text(encoding="utf-8"))
    realization = json.loads((BASE.parent / "manifest.json").read_text(encoding="utf-8"))
    inputs = protocol["inputs"]
    base = gpd.read_file(BASE)
    assignments = pd.read_csv(ARCHIVE / "realization_00" / "pinn" / "seed_2026" / "metrics" / "split_assignments.csv")
    if base.sample_id.tolist() != assignments.sample_id.tolist():
        raise ValueError("Baseline sample rows do not match the authoritative split assignments.")
    cache = OUTPUT / "training_candidate_pool.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            return pd.DataFrame({name: stored[name] for name in stored.files}), base, assignments

    started = time.perf_counter()
    raw = pyogrio.read_dataframe(inputs["candidate_grid"], columns=[])
    x_all, y_all = raw.geometry.x.to_numpy(), raw.geometry.y.to_numpy()
    origin = (float(x_all.min()), float(y_all.min()))
    grid, bounds, partitions, _ = load_spatial_split_manifest(SPLIT)
    inside_bounds = (x_all >= bounds[0]) & (y_all >= bounds[1]) & (x_all <= bounds[2]) & (y_all <= bounds[3])
    x_all, y_all = x_all[inside_bounds], y_all[inside_bounds]
    groups = spatial_block_groups(np.column_stack([x_all, y_all]), grid, bounds)
    # Restrict before any geological classification or feature/eligibility sampling.
    keep = np.isin(groups, partitions["train"])
    x, y, split_block = x_all[keep], y_all[keep], groups[keep]
    del raw, x_all, y_all, groups
    print(f"[pool] Existing grid restricted to {len(x)} training-block candidate points.", flush=True)
    crs = base.crs
    mine, _ = load_polygon_union(Path(inputs["known_zones"]), crs, "EPSG:32733", deduplicate=True)
    structure, _ = load_polygon_union(Path(inputs["structure_zone"]), crs, "EPSG:32733")
    lithology = None
    for path in inputs["lithology_zones"]:
        geom, _ = load_polygon_union(Path(path), crs, "EPSG:32733")
        lithology = geom if lithology is None else lithology.union(geom)
    priors = {name: prior_from_distance(sample_raster_values(x, y, Path(path)), name) for name, path in inputs["prior_rasters"].items()}
    geo = np.nanmax(np.column_stack(list(priors.values())), axis=1)
    feature_valid = raster_valid_mask(x, y, [Path(path) for path in inputs["feature_rasters"]])
    mine_dist = distance(points(x, y), mine)
    in_structure, in_lithology = contains_xy(structure, x, y), contains_xy(lithology, x, y)
    eligible = np.isfinite(geo) & feature_valid & (mine_dist >= 1000.0)
    strata = assign_strata(geo, in_structure, in_lithology, eligible, SamplingConfig())
    thin = spatial_cell_ids(x, y, 80.0, origin, tuple(realization["thin_grid_offset_m"]))
    held_out_negatives = base[(assignments.partition != "train") & (base.Class == 0)]
    # Preserve the same thinning grid and reserve all existing held-out cells.
    eligible &= ~np.isin(thin, held_out_negatives.thin_id.to_numpy(dtype=np.int64))
    pool = pd.DataFrame({
        "x": x, "y": y, "split_block": split_block, "thin_id": thin,
        "block_id": spatial_cell_ids(x, y, 1000.0, origin),
        "stratum": strata.astype("U10"), "geo_score": geo,
        "p_dome": priors["dome"], "p_fault": priors["fault"], "p_strata": priors["strata"],
        "in_struct": in_structure.astype(np.int16), "in_lith": in_lithology.astype(np.int16),
        "mine_dist": mine_dist,
    }).loc[eligible].reset_index(drop=True)
    counts = {name: {"points": int((pool.stratum == name).sum()), "thin_cells": int(pool.loc[pool.stratum == name, "thin_id"].nunique())} for name in ("hard", "transition", "background")}
    needed = int(((assignments.partition == "train") & (base.Class == 0)).sum())
    quotas = compute_quotas(needed, SamplingConfig(hard_fraction=0.65, transition_fraction=0.05, background_fraction=0.30))
    info = {
        "repeat_axis": "sampling_design", "sampling_seed": 2026,
        "candidate_source": inputs["candidate_grid"], "candidate_Class_field_read": False,
        "training_block_candidates": int(len(x)), "eligible_training_points": len(pool),
        "eligible_training_thin_cells": int(pool.thin_id.nunique()), "stratum_counts": counts,
        "requested_training_negatives": needed, "mixture_quotas": quotas,
        "thinning_cell_m": 80.0, "thinning_grid_origin": origin,
        "thinning_grid_offset": realization["thin_grid_offset_m"], "mine_exclusion_m": 1000.0,
        "heldout_thinning_cells_reserved": int(held_out_negatives.thin_id.nunique()),
        "pool_preparation_seconds": time.perf_counter() - started,
        "baseline_hashes": shapefile_hashes(BASE), "split_sha256": sha256_file(SPLIT),
        "inputs": inputs,
        "original_training_strata_counts": base.loc[(assignments.partition == "train") & (base.Class == 0), "stratum"].value_counts().to_dict(),
        "excluded_initial_design": {"mixture": "50/5/45", "required_background": 12205, "available_background_cells": 9361, "diagnostic": str(OUTPUT / "pool_protocol.json")},
        "replacement_design_selection": "65/5/30 selected solely from training-pool capacity before training or looking at test results",
    }
    write_json(OUTPUT / "pool_protocol_feasible_65_5_30.json", info)
    if pool.thin_id.nunique() < needed or any(counts[name]["thin_cells"] < quota for name, quota in quotas.items()):
        raise ValueError(f"Requested quotas cannot be met under the original held-out/thinning constraints: {counts}; quotas={quotas}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{name: pool[name].to_numpy(dtype="U10" if name == "stratum" else None) for name in pool.columns})
    print(f"[pool] Eligible counts {counts}; quotas {quotas}; preparation {info['pool_preparation_seconds']:.1f}s.", flush=True)
    return pool, base, assignments


def select_uniform_cells(pool: pd.DataFrame, quota: int, seed: int = 2026) -> np.ndarray:
    """Uniform eligible 80-m cells with a random representative per cell; no strata quotas."""
    rng = np.random.default_rng(seed)
    candidate = pool.assign(priority=rng.random(len(pool))).sort_values("priority").drop_duplicates("thin_id")
    return rng.choice(candidate.index.to_numpy(), size=quota, replace=False)


def build_samples(pool: pd.DataFrame, base: gpd.GeoDataFrame, assignments: pd.DataFrame) -> dict[str, Path]:
    rows = np.flatnonzero((assignments.partition == "train") & (base.Class == 0))
    heldout = np.flatnonzero(assignments.partition != "train")
    positives = np.flatnonzero(base.Class == 1)
    quota = len(rows)
    paths = {}
    for variant in VARIANTS:
        path = OUTPUT / "samples" / variant / "combined_samples_train_variant.shp"
        paths[variant] = path
        if path.exists():
            continue
        if variant == "uniform_eligible":
            indices = select_uniform_cells(pool, quota)
        else:
            quotas = compute_quotas(quota, SamplingConfig(hard_fraction=0.65, transition_fraction=0.05, background_fraction=0.30))
            rng, used, chosen = np.random.default_rng(2026), set(), []
            for name in ("transition", "background", "hard"):
                selected = select_spatially_balanced(np.flatnonzero(pool.stratum == name), pool.thin_id.to_numpy(), pool.block_id.to_numpy(), quotas[name], rng, used)
                used.update(int(cell) for cell in pool.thin_id.iloc[selected])
                chosen.append(selected)
            indices = np.concatenate(chosen)
            indices = indices[rng.permutation(len(indices))]
        selected = pool.iloc[indices].reset_index(drop=True)
        negatives = gpd.GeoDataFrame({
            "sample_id": [f"{'NU' if variant == 'uniform_eligible' else 'NM'}{index:08d}" for index in range(quota)],
            "Class": np.zeros(quota, dtype=np.int64), "source": "pseudo_abs",
            "stratum": selected.stratum, "zone_id": np.full(quota, -1), "realiz": np.zeros(quota, dtype=np.int64),
            **{name: selected[name].to_numpy() for name in ("geo_score", "p_dome", "p_fault", "p_strata", "in_struct", "in_lith", "mine_dist", "block_id", "thin_id")},
        }, geometry=gpd.points_from_xy(selected.x, selected.y), crs=base.crs)
        variant_samples = base.copy()
        for column in base.columns:
            variant_samples.loc[rows, column] = negatives[column].astype(variant_samples[column].dtype).to_numpy()
        pd.testing.assert_frame_equal(variant_samples.iloc[heldout], base.iloc[heldout], check_exact=True)
        pd.testing.assert_frame_equal(variant_samples.iloc[positives], base.iloc[positives], check_exact=True)
        if variant_samples.sample_id.duplicated().any() or variant_samples.geometry.duplicated().any():
            raise ValueError("Duplicate sample identities or coordinates.")
        all_negative = variant_samples[variant_samples.Class == 0]
        if all_negative.thin_id.duplicated().any() or all_negative.mine_dist.min() < 1000.0:
            raise ValueError("Thinning or mine-exclusion rule violated.")
        write_shapefile(variant_samples, path)
        reopened = gpd.read_file(path)
        pd.testing.assert_frame_equal(reopened.iloc[heldout], base.iloc[heldout], check_exact=True, check_dtype=False)
        pd.testing.assert_frame_equal(reopened.iloc[positives], base.iloc[positives], check_exact=True, check_dtype=False)
        grid, bounds, partitions, _ = load_spatial_split_manifest(SPLIT)
        new_groups = spatial_block_groups(np.column_stack([reopened.geometry.x, reopened.geometry.y]), grid, bounds)
        if not np.isin(new_groups[rows], partitions["train"]).all():
            raise ValueError("A replacement training negative escaped the fixed training blocks.")
        write_json(path.parent / "sampling_audit.json", {
            "passed": True, "variant": variant, "training_negatives": quota,
            "training_positives": int(((assignments.partition == "train") & (base.Class == 1)).sum()),
            "heldout_rows": len(heldout), "heldout_rows_attributes_labels_geometry_exact": True,
            "all_positive_rows_exact": True, "all_negative_thinning_cells_unique": True,
            "training_strata_counts": selected.stratum.value_counts().to_dict(),
            "minimum_negative_mine_distance_m": float(all_negative.mine_dist.min()),
            "sample_hashes": shapefile_hashes(path),
        })
    return paths


def train(variant: str, sample: Path, algorithm: str) -> dict[str, Any]:
    output = OUTPUT / "sampling_design" / variant / algorithm / "seed_2026"
    completion = output / "completed.json"
    if completion.exists() and metrics_complete(output):
        return json.loads(completion.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    command = build_command(train_script=ROOT / "train_multi_physics_model.py", sample_path=sample,
        feature_dir=ROOT / "data" / "factors", prior_dir=ROOT / "data" / "priors", output_dir=output,
        split_manifest=SPLIT, algorithm=algorithm, model_seed=2026, split_seed=2026, grid_size=5,
        deep_epochs=80, patience=20, physics_weight=0.1, selection_metric="pr_auc", device="cpu")
    command.extend(["--torch-threads", "2"])
    write_json(output / "command.json", {"argv": command, "variant": variant, "axis": "sampling_design"})
    environment = os.environ.copy()
    environment.update({"PYTHONUTF8": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "LOKY_MAX_CPU_COUNT": "2"})
    started = time.perf_counter()
    print(f"[start] sampling_design/{variant}/{algorithm}", flush=True)
    with (output / "training.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            timeout=600, check=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode or not metrics_complete(output):
        raise RuntimeError(f"Training failed for {variant}/{algorithm}; inspect its own log.")
    metrics = json.loads((output / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    for partition in ("validation", "test"):
        pred = pd.read_csv(output / "metrics" / f"{partition}_predictions.csv")
        reference = pd.read_csv(ARCHIVE / "realization_00" / algorithm / "seed_2026" / "metrics" / f"{partition}_predictions.csv")
        columns = ["row_index", "sample_id", "partition", "block_id", "label", "x", "y"]
        pd.testing.assert_frame_equal(pred[columns], reference[columns], check_exact=True)
        labels, probabilities = pred.label.to_numpy(), pred.probability.to_numpy(dtype=np.float32)
        for metric, function in (("pr_auc", average_precision_score), ("roc_auc", roc_auc_score), ("brier_score", brier_score_loss)):
            if abs(function(labels, probabilities) - metrics[f"{partition}_metrics"][metric]) > 1e-10:
                raise ValueError(f"Prediction metric mismatch in {variant}/{algorithm}/{partition}/{metric}.")
    row = {"repeat_axis": "sampling_design", "variant": variant, "algorithm": algorithm,
        "realization": 0, "seed": 2026, "status": "completed", "source_output": str(output),
        "duration_seconds": time.perf_counter() - started, "heldout_prediction_identities_exact": True,
        **{f"test_{metric}": value for metric, value in metrics["test_metrics"].items() if isinstance(value, (float, int))}}
    write_json(completion, row)
    print(f"[completed] {variant}/{algorithm}: AP={row['test_pr_auc']:.6f}", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    locked_root = ROOT / "experiments" / "ogr_revision_20260905"
    locked_paths = sorted(path for path in locked_root.rglob("*") if path.is_file())
    locked_paths.append(ROOT / "analysis_output" / "ogr_revision_20260905" / "all_run_metrics.csv")
    locked_before = {str(path.relative_to(ROOT)): sha256_file(path) for path in locked_paths}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pool, base, assignments = prepare_pool()
    paths = build_samples(pool, base, assignments)
    if args.prepare_only:
        return
    rows = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(train, variant, paths[variant], algorithm) for variant in VARIANTS for algorithm in ("pinn", "rf")]
        for job in as_completed(jobs):
            rows.append(job.result())
            pd.DataFrame(rows).to_csv(OUTPUT / "run_ledger.csv", index=False, encoding="utf-8-sig")
    for algorithm in ("pinn", "rf"):
        source = ARCHIVE / "realization_00" / algorithm / "seed_2026"
        metrics = json.loads((source / "metrics" / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"repeat_axis": "sampling_design", "variant": "original_70_5_25_archive", "algorithm": algorithm,
            "realization": 0, "seed": 2026, "status": "reference", "source_output": str(source),
            **{f"test_{metric}": value for metric, value in metrics["test_metrics"].items() if isinstance(value, (float, int))}})
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ANALYSIS / "sampling_design_metrics.csv", index=False, encoding="utf-8-sig")
    locked_after = {str(path.relative_to(ROOT)): sha256_file(path) for path in locked_paths}
    if locked_before != locked_after:
        raise ValueError("An artifact in the locked 72-run archive changed during sampling-design execution.")
    write_json(ANALYSIS / "sampling_design_audit.json", {"passed": True, "new_runs": 4,
        "all_validation_test_rows_labels_coordinates_unchanged": True,
        "original_72_run_archive_edited": False, "comparison_unit": "single r00 train-only sampling design; descriptive, not n=10",
        "uniform_definition": "uniform eligible 80-m training cells and a random within-cell representative; no geological strata quotas or 1-km balance",
        "mixture_definition": "hard/transition/background 65/5/30 with original 1-km spatial balance and global 80-m thinning",
        "excluded_design": "50/5/45 infeasible: 12,205 background samples required but only 9,361 eligible training thinning cells; replacement 65/5/30 chosen by training capacity before examining new test results",
        "locked_artifact_files_hash_verified_unchanged": len(locked_before),
        "runner_sha256": sha256_file(Path(__file__)), "training_script_sha256": sha256_file(ROOT / "train_multi_physics_model.py")})


if __name__ == "__main__":
    main()
