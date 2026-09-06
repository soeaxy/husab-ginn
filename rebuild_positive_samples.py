"""Reconstruct the manuscript's positive-sample chain from vector inputs.

The implementation intentionally preserves the recovered operation order:

1. anchor a 20 m point grid at the study-area minimum bounds (x outer loop),
2. retain points strictly within the study area,
3. label points strictly within the mapped deposit polygons as ``Class=1``,
4. draw ``floor(n / 10)`` rows with pandas, without replacement and seed 1,
5. remove sampled rows intersecting the tailings polygons.

All paths are supplied through the command line.  GeoPackage is used by default
to avoid the multi-file and field-name limitations of ESRI Shapefile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import Point

from manuscript_protocol import (
    POSITIVE_GRID_SPACING_M,
    POSITIVE_SUBSAMPLE_FRACTION,
    POSITIVE_SUBSAMPLE_RANDOM_STATE,
)

DEFAULT_SPACING_M = POSITIVE_GRID_SPACING_M
DEFAULT_SAMPLE_DIVISOR = round(1 / POSITIVE_SUBSAMPLE_FRACTION)
DEFAULT_RANDOM_STATE = POSITIVE_SUBSAMPLE_RANDOM_STATE


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_hashes(path: Path) -> dict[str, str]:
    """Hash a vector dataset, including all Shapefile sidecars when relevant."""

    if path.suffix.lower() == ".shp":
        recognized_endings = {
            ".shp",
            ".shx",
            ".dbf",
            ".prj",
            ".cpg",
            ".qix",
            ".sbn",
            ".sbx",
            ".shp.xml",
        }
        files = sorted(
            candidate
            for candidate in path.parent.glob(f"{path.stem}.*")
            if candidate.is_file()
            and any(
                candidate.name.lower().endswith(ending) for ending in recognized_endings
            )
        )
    else:
        files = [path]
    return {candidate.name: file_sha256(candidate) for candidate in files}


def _validate_polygon_frame(frame: gpd.GeoDataFrame, label: str) -> None:
    if frame.empty:
        raise ValueError(f"{label} contains no features.")
    if frame.crs is None:
        raise ValueError(f"{label} has no CRS.")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise ValueError(f"{label} contains null or empty geometries.")
    if not frame.geometry.is_valid.all():
        raise ValueError(f"{label} contains invalid geometries.")
    if not frame.geom_type.isin({"Polygon", "MultiPolygon"}).all():
        raise ValueError(f"{label} must contain only polygon geometries.")


def _validate_metric_crs(crs: Any) -> CRS:
    parsed = CRS.from_user_input(crs)
    if not parsed.is_projected:
        raise ValueError("The study-area CRS must be projected for metre spacing.")
    units = {axis.unit_name.lower() for axis in parsed.axis_info if axis.unit_name}
    if units and not any("metre" in unit or "meter" in unit for unit in units):
        raise ValueError("The study-area CRS axes must use metres.")
    return parsed


def _union(frame: gpd.GeoDataFrame):
    if hasattr(frame.geometry, "union_all"):
        return frame.geometry.union_all()
    return frame.geometry.unary_union


def align_to_study_crs(
    frame: gpd.GeoDataFrame,
    study_crs: Any,
    label: str,
) -> gpd.GeoDataFrame:
    """Validate a polygon layer and reproject it to the study-area CRS."""

    _validate_polygon_frame(frame, label)
    if CRS.from_user_input(frame.crs) != CRS.from_user_input(study_crs):
        return frame.to_crs(study_crs)
    return frame


def build_positive_candidates(
    study_area: gpd.GeoDataFrame,
    known_polygons: gpd.GeoDataFrame,
    spacing_m: float = DEFAULT_SPACING_M,
) -> tuple[gpd.GeoDataFrame, dict[str, int | float | list[float]]]:
    """Build ordered ``Class=1`` candidates using strict containment.

    Coordinates follow the recovered script exactly: both axes use
    ``numpy.arange(minimum, maximum, spacing)`` and x is the outer loop.
    Boundary points are excluded first by the study area and then by the known
    polygons because Shapely ``within`` is a strict interior predicate.
    """

    _validate_polygon_frame(study_area, "study area")
    _validate_metric_crs(study_area.crs)
    if not math.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("spacing_m must be finite and positive.")
    known = align_to_study_crs(known_polygons, study_area.crs, "known polygons")

    minx, miny, maxx, maxy = map(float, study_area.total_bounds)
    x_coords = np.arange(minx, maxx, spacing_m, dtype=np.float64)
    y_coords = np.arange(miny, maxy, spacing_m, dtype=np.float64)
    if x_coords.size == 0 or y_coords.size == 0:
        raise ValueError("The study-area bounds are smaller than the grid spacing.")

    points: list[Point] = []
    grid_order: list[int] = []
    x_indices: list[int] = []
    y_indices: list[int] = []
    y_count = int(y_coords.size)
    study_union = _union(study_area)
    known_union = _union(known)
    within_study_count = 0
    for x_index, x_coord in enumerate(x_coords):
        column = gpd.GeoSeries(
            gpd.points_from_xy(
                np.full(y_coords.shape, x_coord, dtype=np.float64),
                y_coords,
            ),
            crs=study_area.crs,
        )
        within_study = column.within(study_union).to_numpy(dtype=bool)
        within_study_count += int(within_study.sum())
        positive_y_indices = np.flatnonzero(
            within_study & column.within(known_union).to_numpy(dtype=bool)
        )
        for y_index in positive_y_indices:
            points.append(column.iloc[int(y_index)])
            grid_order.append(x_index * y_count + int(y_index))
            x_indices.append(x_index)
            y_indices.append(int(y_index))

    candidates = gpd.GeoDataFrame(
        {
            "grid_ord": np.asarray(grid_order, dtype=np.int64),
            "x_idx": np.asarray(x_indices, dtype=np.int64),
            "y_idx": np.asarray(y_indices, dtype=np.int64),
        },
        geometry=points,
        crs=study_area.crs,
    )
    full_grid_count = int(x_coords.size * y_coords.size)
    candidates.insert(
        0,
        "cand_id",
        [f"C{candidate_index:08d}" for candidate_index in range(len(candidates))],
    )
    candidates["Class"] = np.ones(len(candidates), dtype=np.int16)
    candidates.reset_index(drop=True, inplace=True)

    audit: dict[str, int | float | list[float]] = {
        "spacing_m": float(spacing_m),
        "study_bounds": [minx, miny, maxx, maxy],
        "x_coordinate_count": int(x_coords.size),
        "y_coordinate_count": int(y_coords.size),
        "full_bounding_grid_count": int(full_grid_count),
        "strictly_within_study_area_count": int(within_study_count),
        "positive_candidate_count": int(len(candidates)),
    }
    return candidates, audit


def sample_positive_candidates(
    candidates: gpd.GeoDataFrame,
    tailings: gpd.GeoDataFrame | None = None,
    sample_divisor: int = DEFAULT_SAMPLE_DIVISOR,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Sample candidates without replacement, then exclude tailings intersections."""

    if sample_divisor <= 0:
        raise ValueError("sample_divisor must be positive.")
    if candidates.crs is None:
        raise ValueError("Positive candidates have no CRS.")
    sample_count = len(candidates) // sample_divisor
    if sample_count < 1:
        raise ValueError(
            "Too few positive candidates: floor(candidate_count / sample_divisor) is zero."
        )

    sampled = candidates.sample(
        n=sample_count,
        random_state=random_state,
        replace=False,
    ).copy()
    before_tailings = len(sampled)
    if tailings is not None:
        aligned_tailings = align_to_study_crs(tailings, candidates.crs, "tailings")
        sampled = sampled.loc[
            ~sampled.geometry.intersects(_union(aligned_tailings))
        ].copy()
    excluded_tailings = before_tailings - len(sampled)
    sampled.insert(
        0,
        "sample_id",
        [f"P{sample_index:07d}" for sample_index in range(len(sampled))],
    )
    sampled.reset_index(drop=True, inplace=True)
    audit = {
        "sample_size_before_tailings": int(before_tailings),
        "tailings_intersection_excluded_count": int(excluded_tailings),
        "final_positive_sample_count": int(len(sampled)),
    }
    return sampled, audit


def _remove_dataset(path: Path) -> None:
    """Remove only the requested owned dataset, never unrelated directory entries."""

    if path.suffix.lower() == ".shp":
        sidecars = list(path.parent.glob(f"{path.stem}.*"))
        for sidecar in sidecars:
            if sidecar.is_file():
                sidecar.unlink()
    elif path.exists():
        if not path.is_file():
            raise ValueError(f"Refusing to replace non-file output: {path}")
        path.unlink()


def prepare_outputs(output_dir: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    """Prepare fixed output paths without deleting unknown files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "positive_candidates.gpkg"
    sample_path = output_dir / "positive_samples.gpkg"
    manifest_path = output_dir / "positive_sampling_manifest.json"
    existing_owned = [
        path for path in (candidate_path, sample_path, manifest_path) if path.exists()
    ]
    if existing_owned and not overwrite:
        raise FileExistsError(
            "Generated outputs already exist; pass --overwrite to replace only those files: "
            + ", ".join(str(path) for path in existing_owned)
        )
    if overwrite:
        for path in (candidate_path, sample_path, manifest_path):
            _remove_dataset(path)
    return candidate_path, sample_path, manifest_path


def write_vector(frame: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".gpkg":
        frame.to_file(path, driver="GPKG", layer=path.stem, index=False)
    elif suffix == ".shp":
        frame.to_file(path, driver="ESRI Shapefile", encoding="UTF-8", index=False)
    elif suffix in {".geojson", ".json"}:
        frame.to_file(path, driver="GeoJSON", index=False)
    else:
        raise ValueError(f"Unsupported vector output format: {path.suffix}")


def build_manifest(
    *,
    study_area_path: Path,
    known_polygons_path: Path,
    tailings_path: Path | None,
    candidate_path: Path,
    sample_path: Path,
    candidates: gpd.GeoDataFrame,
    samples: gpd.GeoDataFrame,
    grid_audit: dict[str, int | float | list[float]],
    sample_audit: dict[str, int],
    sample_divisor: int,
    random_state: int,
    assumed_missing_crs: str | None,
) -> dict[str, Any]:
    inputs = {
        "study_area": {
            "path": str(study_area_path),
            "sha256": dataset_hashes(study_area_path),
        },
        "known_polygons": {
            "path": str(known_polygons_path),
            "sha256": dataset_hashes(known_polygons_path),
        },
        "tailings": None,
    }
    if tailings_path is not None:
        inputs["tailings"] = {
            "path": str(tailings_path),
            "sha256": dataset_hashes(tailings_path),
        }
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "grid_anchor": "study_area_min_bounds",
            "coordinate_generation": "numpy.arange(minimum, maximum, spacing)",
            "loop_order": "x_outer_y_inner",
            "study_area_predicate": "strict_within",
            "positive_polygon_predicate": "strict_within",
            "sample_size_rule": "floor(positive_candidate_count / sample_divisor)",
            "sampling_engine": "pandas.DataFrame.sample",
            "replace": False,
            "tailings_operation_order": "after_random_sampling",
            "tailings_predicate": "exclude_intersects",
        },
        "parameters": {
            "spacing_m": grid_audit["spacing_m"],
            "sample_divisor": int(sample_divisor),
            "random_state": int(random_state),
            "assumed_missing_crs": assumed_missing_crs,
        },
        "crs": candidates.crs.to_string() if candidates.crs is not None else None,
        "counts": {**grid_audit, **sample_audit},
        "validation": {
            "candidate_class_values": sorted(map(int, candidates["Class"].unique())),
            "sample_class_values": sorted(map(int, samples["Class"].unique())),
            "candidate_coordinate_count": int(
                candidates.geometry.apply(lambda point: (point.x, point.y)).nunique()
            ),
            "sample_coordinate_count": int(
                samples.geometry.apply(lambda point: (point.x, point.y)).nunique()
            ),
            "sample_without_replacement": bool(samples["cand_id"].is_unique),
        },
        "inputs": inputs,
        "outputs": {
            "positive_candidates": {
                "path": str(candidate_path),
                "sha256": dataset_hashes(candidate_path),
            },
            "positive_samples": {
                "path": str(sample_path),
                "sha256": dataset_hashes(sample_path),
            },
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct ordered 20 m positive candidates and sampled positives."
    )
    parser.add_argument("--study-area", type=Path, required=True)
    parser.add_argument("--known-polygons", type=Path, required=True)
    parser.add_argument("--tailings", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spacing-m", type=float, default=DEFAULT_SPACING_M)
    parser.add_argument("--sample-divisor", type=int, default=DEFAULT_SAMPLE_DIVISOR)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--assume-missing-crs",
        default=None,
        help=(
            "Explicit CRS applied only to inputs whose metadata are missing; "
            "for example EPSG:32733. Omit to reject such inputs."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this tool's three named outputs; leave all unknown files untouched.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [args.study_area, args.known_polygons]
    if args.tailings is not None:
        input_paths.append(args.tailings)
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required vector inputs are missing: {missing}")

    expected_outputs = (
        args.output_dir / "positive_candidates.gpkg",
        args.output_dir / "positive_samples.gpkg",
        args.output_dir / "positive_sampling_manifest.json",
    )
    output_paths = {path.resolve() for path in expected_outputs}
    if any(path.resolve() in output_paths for path in input_paths):
        raise ValueError("An input path must not be reused as a generated output path.")
    candidate_path, sample_path, manifest_path = prepare_outputs(
        args.output_dir, args.overwrite
    )

    study_area = gpd.read_file(args.study_area)
    known_polygons = gpd.read_file(args.known_polygons)
    tailings = gpd.read_file(args.tailings) if args.tailings is not None else None
    frames = [study_area, known_polygons]
    if tailings is not None:
        frames.append(tailings)
    if args.assume_missing_crs is not None:
        assumed_crs = CRS.from_user_input(args.assume_missing_crs)
        for frame in frames:
            if frame.crs is None:
                frame.set_crs(assumed_crs, inplace=True)
    candidates, grid_audit = build_positive_candidates(
        study_area,
        known_polygons,
        spacing_m=args.spacing_m,
    )
    samples, sample_audit = sample_positive_candidates(
        candidates,
        tailings,
        sample_divisor=args.sample_divisor,
        random_state=args.random_state,
    )
    write_vector(candidates, candidate_path)
    write_vector(samples, sample_path)
    manifest = build_manifest(
        study_area_path=args.study_area,
        known_polygons_path=args.known_polygons,
        tailings_path=args.tailings,
        candidate_path=candidate_path,
        sample_path=sample_path,
        candidates=candidates,
        samples=samples,
        grid_audit=grid_audit,
        sample_audit=sample_audit,
        sample_divisor=args.sample_divisor,
        random_state=args.random_state,
        assumed_missing_crs=args.assume_missing_crs,
    )
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def main() -> None:
    manifest = run(parse_args())
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
