from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from shapely import contains_xy, distance, points
from shapely.geometry.base import BaseGeometry

from manuscript_protocol import (
    PSEUDO_ABSENCE_BASE_SEED,
    PSEUDO_ABSENCE_SEED_STEP,
    PSEUDO_ABSENCE_SEEDS,
)

LOGGER = logging.getLogger("negative-sample-reconstruction")


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for one family of reconstructed negative samples."""

    negative_ratio: float = 10.0
    realizations: int = len(PSEUDO_ABSENCE_SEEDS)
    base_seed: int = PSEUDO_ABSENCE_BASE_SEED
    positive_buffer_m: float = 1000.0
    thinning_cell_m: float = 80.0
    balance_block_m: float = 1000.0
    hard_fraction: float = 0.70
    transition_fraction: float = 0.05
    background_fraction: float = 0.25
    hard_prior_threshold: float = 0.50
    transition_prior_threshold: float = 0.20

    def validate(self) -> None:
        if self.negative_ratio <= 0:
            raise ValueError("negative_ratio must be positive.")
        if self.realizations < 1:
            raise ValueError("realizations must be at least 1.")
        if self.positive_buffer_m <= 0:
            raise ValueError("positive_buffer_m must be positive.")
        if self.thinning_cell_m <= 0 or self.balance_block_m <= 0:
            raise ValueError("Spatial cell sizes must be positive.")
        fractions = (
            self.hard_fraction,
            self.transition_fraction,
            self.background_fraction,
        )
        if any(value < 0 for value in fractions):
            raise ValueError("Sampling fractions cannot be negative.")
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("Sampling fractions must sum to 1.0.")
        if not 0 <= self.transition_prior_threshold < self.hard_prior_threshold <= 1:
            raise ValueError(
                "Prior thresholds must satisfy 0 <= transition < hard <= 1."
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shapefile_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for sidecar in sorted(path.parent.glob(f"{path.stem}.*")):
        if sidecar.is_file():
            hashes[sidecar.name] = file_sha256(sidecar)
    return hashes


def compute_quotas(total: int, config: SamplingConfig) -> dict[str, int]:
    if total < 1:
        raise ValueError("The requested negative-sample count must be positive.")
    hard = int(round(total * config.hard_fraction))
    transition = int(round(total * config.transition_fraction))
    background = total - hard - transition
    if min(hard, transition, background) < 0:
        raise ValueError("Invalid stratum quota after rounding.")
    return {
        "hard": hard,
        "transition": transition,
        "background": background,
    }


def prior_from_distance(values: np.ndarray, prior_type: str) -> np.ndarray:
    distances = np.asarray(values, dtype=np.float64)
    if prior_type == "dome":
        prior = np.exp(-((distances**2) / (2.0 * 1000.0**2)))
    elif prior_type == "fault":
        prior = np.exp(-0.002 * distances)
    elif prior_type == "strata":
        prior = np.exp(-((distances**2) / (2.0 * 800.0**2)))
    else:
        raise ValueError(f"Unsupported prior type: {prior_type}")
    prior[~np.isfinite(distances)] = np.nan
    return np.clip(prior, 0.0, 1.0).astype(np.float32)


def assign_strata(
    geo_score: np.ndarray,
    in_structure: np.ndarray,
    in_lithology: np.ndarray,
    eligible: np.ndarray,
    config: SamplingConfig,
) -> np.ndarray:
    """Assign eligible candidates to mutually exclusive difficulty strata."""

    score = np.asarray(geo_score, dtype=np.float64)
    structure = np.asarray(in_structure, dtype=bool)
    lithology = np.asarray(in_lithology, dtype=bool)
    eligible_mask = np.asarray(eligible, dtype=bool)
    if not (score.shape == structure.shape == lithology.shape == eligible_mask.shape):
        raise ValueError("Candidate arrays must have matching shapes.")

    strata = np.full(score.shape, "ineligible", dtype=object)
    hard = eligible_mask & (
        structure | lithology | (score >= config.hard_prior_threshold)
    )
    transition = (
        eligible_mask
        & ~hard
        & (score >= config.transition_prior_threshold)
    )
    background = eligible_mask & ~hard & ~transition
    strata[hard] = "hard"
    strata[transition] = "transition"
    strata[background] = "background"
    return strata


def spatial_cell_ids(
    x: np.ndarray,
    y: np.ndarray,
    cell_size: float,
    origin: tuple[float, float],
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    ix = np.floor((x - origin[0] + offset[0]) / cell_size).astype(np.int64)
    iy = np.floor((y - origin[1] + offset[1]) / cell_size).astype(np.int64)
    return (ix << np.int64(32)) ^ (iy & np.int64(0xFFFFFFFF))


def select_spatially_balanced(
    candidate_indices: np.ndarray,
    thin_ids: np.ndarray,
    block_ids: np.ndarray,
    quota: int,
    rng: np.random.Generator,
    used_thin_ids: set[int] | None = None,
) -> np.ndarray:
    """Select one point per thinning cell and rotate selections across blocks."""

    if quota == 0:
        return np.empty(0, dtype=np.int64)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("No candidates are available for a requested stratum.")
    used = used_thin_ids if used_thin_ids is not None else set()
    local_thin = thin_ids[indices]
    if used:
        keep = ~np.isin(local_thin, np.fromiter(used, dtype=np.int64))
        indices = indices[keep]
        local_thin = local_thin[keep]
    if indices.size < quota:
        raise ValueError(
            f"Only {indices.size} candidates remain for a quota of {quota}."
        )

    frame = pd.DataFrame(
        {
            "idx": indices,
            "thin_id": local_thin,
            "block_id": block_ids[indices],
            "priority": rng.random(indices.size),
        }
    )
    frame = frame.sort_values("priority").drop_duplicates("thin_id", keep="first")
    if len(frame) < quota:
        raise ValueError(
            f"Only {len(frame)} unique thinning cells are available for a quota of {quota}."
        )

    frame = frame.iloc[rng.permutation(len(frame))].copy()
    frame["round_rank"] = frame.groupby("block_id", sort=False).cumcount()
    unique_blocks = frame["block_id"].unique()
    block_order = pd.Series(
        rng.permutation(len(unique_blocks)), index=unique_blocks, dtype=np.int64
    )
    frame["block_order"] = frame["block_id"].map(block_order)
    frame["tie"] = rng.random(len(frame))
    selected = frame.sort_values(
        ["round_rank", "block_order", "tie"], kind="mergesort"
    ).head(quota)
    return selected["idx"].to_numpy(dtype=np.int64)


def sample_raster_values(
    x: np.ndarray, y: np.ndarray, raster_path: Path
) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        rr, cc = rasterio.transform.rowcol(src.transform, x, y, op=np.floor)
        rr = np.asarray(rr, dtype=np.int64)
        cc = np.asarray(cc, dtype=np.int64)
        in_bounds = (rr >= 0) & (cc >= 0) & (rr < src.height) & (cc < src.width)
        values = np.full(x.shape, np.nan, dtype=np.float32)
        band = src.read(1)
        values[in_bounds] = band[rr[in_bounds], cc[in_bounds]].astype(np.float32)
        if src.nodata is not None:
            values[np.isclose(values, src.nodata)] = np.nan
        return values


def raster_valid_mask(
    x: np.ndarray, y: np.ndarray, raster_paths: Iterable[Path]
) -> np.ndarray:
    valid = np.ones(x.shape, dtype=bool)
    for raster_path in raster_paths:
        valid &= np.isfinite(sample_raster_values(x, y, raster_path))
    return valid


def load_polygon_union(
    path: Path,
    target_crs: object,
    assume_crs: str,
    deduplicate: bool = False,
) -> tuple[BaseGeometry, gpd.GeoDataFrame]:
    layer = pyogrio.read_dataframe(path)
    if layer.empty:
        raise ValueError(f"Polygon layer is empty: {path}")
    if layer.crs is None:
        layer = layer.set_crs(assume_crs)
    if str(layer.crs) != str(target_crs):
        layer = layer.to_crs(target_crs)
    if deduplicate:
        layer = layer.assign(_wkb=layer.geometry.apply(lambda geom: geom.wkb))
        layer = layer.drop_duplicates("_wkb").drop(columns="_wkb")
    union = layer.geometry.union_all()
    if union.is_empty:
        raise ValueError(f"Polygon union is empty: {path}")
    return union, layer


def assign_positive_zones(
    x: np.ndarray, y: np.ndarray, zones: gpd.GeoDataFrame
) -> np.ndarray:
    zone_ids = np.full(x.shape, -1, dtype=np.int16)
    for zone_id, geom in enumerate(zones.geometry):
        inside = contains_xy(geom.buffer(0.01), x, y)
        zone_ids[(zone_ids < 0) & inside] = zone_id
    if np.any(zone_ids < 0):
        missing = int((zone_ids < 0).sum())
        raise ValueError(f"{missing} positive samples could not be assigned to a zone.")
    return zone_ids


def validate_training_samples(
    samples: gpd.GeoDataFrame,
    expected_positive: int,
    expected_negative: int,
    positive_buffer: BaseGeometry | None = None,
    minimum_negative_distance_m: float | None = None,
) -> dict[str, int | float]:
    required = {"Class", "sample_id", "source", "stratum", "geometry"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Training output is missing fields: {sorted(missing)}")
    if samples.crs is None or not samples.crs.is_projected:
        raise ValueError("Training output must use a projected CRS.")
    if samples.empty or samples.geometry.is_empty.any() or samples.geometry.isna().any():
        raise ValueError("Training output contains missing or empty geometry.")
    if not (samples.geom_type == "Point").all():
        raise ValueError("Training output must contain only Point geometry.")
    labels = samples["Class"].astype(int)
    if set(labels.unique()) != {0, 1}:
        raise ValueError("Class must contain both binary labels 0 and 1.")
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives != expected_positive or negatives != expected_negative:
        raise ValueError(
            f"Unexpected class counts: positive={positives}, negative={negatives}."
        )
    coordinates = np.column_stack(
        [samples.geometry.x.to_numpy(), samples.geometry.y.to_numpy()]
    )
    if len(np.unique(coordinates, axis=0)) != len(samples):
        raise ValueError("Training output contains duplicate coordinates.")
    if samples["sample_id"].duplicated().any():
        raise ValueError("Training output contains duplicate sample_id values.")
    if positive_buffer is not None:
        negatives_gdf = samples.loc[labels == 0]
        inside = contains_xy(
            positive_buffer,
            negatives_gdf.geometry.x.to_numpy(),
            negatives_gdf.geometry.y.to_numpy(),
        )
        if inside.any():
            raise ValueError("Negative samples intersect the positive exclusion buffer.")
    if minimum_negative_distance_m is not None:
        if "mine_dist" not in samples.columns:
            raise ValueError(
                "mine_dist is required when a minimum negative distance is checked."
            )
        minimum_distance = float(samples.loc[labels == 0, "mine_dist"].min())
        if minimum_distance + 1e-6 < minimum_negative_distance_m:
            raise ValueError(
                "Negative samples violate the minimum mine-distance constraint: "
                f"{minimum_distance:.6f} < {minimum_negative_distance_m:.6f} m."
            )
    return {
        "rows": int(len(samples)),
        "positive": positives,
        "negative": negatives,
        "unique_coordinates": int(len(samples)),
    }


def build_positive_metadata(
    positives: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    dome_tif: Path,
    fault_tif: Path,
    strata_tif: Path,
    structure_union: BaseGeometry | None,
    lithology_union: BaseGeometry | None,
) -> gpd.GeoDataFrame:
    x = positives.geometry.x.to_numpy()
    y = positives.geometry.y.to_numpy()
    p_dome = prior_from_distance(sample_raster_values(x, y, dome_tif), "dome")
    p_fault = prior_from_distance(sample_raster_values(x, y, fault_tif), "fault")
    p_strata = prior_from_distance(sample_raster_values(x, y, strata_tif), "strata")
    geo_score = np.nanmax(np.column_stack([p_dome, p_fault, p_strata]), axis=1)
    in_structure = (
        contains_xy(structure_union, x, y)
        if structure_union is not None
        else np.zeros(len(positives), dtype=bool)
    )
    in_lithology = (
        contains_xy(lithology_union, x, y)
        if lithology_union is not None
        else np.zeros(len(positives), dtype=bool)
    )
    zone_ids = assign_positive_zones(x, y, zones)
    frame = gpd.GeoDataFrame(
        {
            "sample_id": [f"P{idx:07d}" for idx in range(len(positives))],
            "Class": np.ones(len(positives), dtype=np.int16),
            "source": "known_zone",
            "stratum": "positive",
            "zone_id": zone_ids,
            "realiz": np.full(len(positives), -1, dtype=np.int16),
            "geo_score": geo_score.astype(np.float32),
            "p_dome": p_dome,
            "p_fault": p_fault,
            "p_strata": p_strata,
            "in_struct": in_structure.astype(np.int16),
            "in_lith": in_lithology.astype(np.int16),
            "mine_dist": np.zeros(len(positives), dtype=np.float32),
            "block_id": np.full(len(positives), -1, dtype=np.int64),
            "thin_id": np.full(len(positives), -1, dtype=np.int64),
        },
        geometry=positives.geometry.to_numpy(),
        crs=positives.crs,
    )
    return frame


def make_negative_frame(
    selected: dict[str, np.ndarray],
    realization: int,
    x: np.ndarray,
    y: np.ndarray,
    thin_ids: np.ndarray,
    block_ids: np.ndarray,
    geo_score: np.ndarray,
    p_dome: np.ndarray,
    p_fault: np.ndarray,
    p_strata: np.ndarray,
    in_structure: np.ndarray,
    in_lithology: np.ndarray,
    mine_union: BaseGeometry,
    crs: object,
) -> gpd.GeoDataFrame:
    ordered_indices: list[np.ndarray] = []
    ordered_strata: list[str] = []
    for stratum in ("hard", "transition", "background"):
        values = selected[stratum]
        ordered_indices.append(values)
        ordered_strata.extend([stratum] * len(values))
    indices = np.concatenate(ordered_indices)
    geometries = points(x[indices], y[indices])
    mine_distance = distance(geometries, mine_union).astype(np.float32)
    prefix = {"hard": "H", "transition": "T", "background": "B"}
    counters = {key: 0 for key in prefix}
    sample_ids: list[str] = []
    for stratum in ordered_strata:
        sample_ids.append(
            f"N{realization:02d}{prefix[stratum]}{counters[stratum]:06d}"
        )
        counters[stratum] += 1
    return gpd.GeoDataFrame(
        {
            "sample_id": sample_ids,
            "Class": np.zeros(len(indices), dtype=np.int16),
            "source": "pseudo_abs",
            "stratum": ordered_strata,
            "zone_id": np.full(len(indices), -1, dtype=np.int16),
            "realiz": np.full(len(indices), realization, dtype=np.int16),
            "geo_score": geo_score[indices].astype(np.float32),
            "p_dome": p_dome[indices].astype(np.float32),
            "p_fault": p_fault[indices].astype(np.float32),
            "p_strata": p_strata[indices].astype(np.float32),
            "in_struct": in_structure[indices].astype(np.int16),
            "in_lith": in_lithology[indices].astype(np.int16),
            "mine_dist": mine_distance,
            "block_id": block_ids[indices].astype(np.int64),
            "thin_id": thin_ids[indices].astype(np.int64),
        },
        geometry=geometries,
        crs=crs,
    )


def write_shapefile(samples: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_file(path, driver="ESRI Shapefile", encoding="UTF-8", index=False)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild spatially balanced pseudo-absence samples without excluding "
            "all favorable geological settings."
        )
    )
    parser.add_argument("--candidate-grid", type=Path, required=True)
    parser.add_argument(
        "--positive-samples",
        type=Path,
        default=project_root / "data" / "combined_samples.shp",
    )
    parser.add_argument("--known-zones", type=Path, required=True)
    parser.add_argument("--structure-zone", type=Path, default=None)
    parser.add_argument(
        "--lithology-zone", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--feature-dir", type=Path, default=project_root / "data" / "factors"
    )
    parser.add_argument(
        "--dome-tif", type=Path, default=project_root / "data" / "priors" / "dome.tif"
    )
    parser.add_argument(
        "--fault-tif", type=Path, default=project_root / "data" / "priors" / "fault.tif"
    )
    parser.add_argument(
        "--strata-tif",
        type=Path,
        default=project_root / "data" / "priors" / "strata.tif",
    )
    parser.add_argument("--label-col", default="Class")
    parser.add_argument("--assume-zone-crs", default="EPSG:32733")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "reconstructed_samples_v2",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing reconstruction in the selected output directory.",
    )
    parser.add_argument("--negative-ratio", type=float, default=10.0)
    parser.add_argument("--realizations", type=int, default=len(PSEUDO_ABSENCE_SEEDS))
    parser.add_argument("--base-seed", type=int, default=PSEUDO_ABSENCE_BASE_SEED)
    parser.add_argument("--positive-buffer-m", type=float, default=1000.0)
    parser.add_argument("--thinning-cell-m", type=float, default=80.0)
    parser.add_argument("--balance-block-m", type=float, default=1000.0)
    parser.add_argument("--hard-fraction", type=float, default=0.70)
    parser.add_argument("--transition-fraction", type=float, default=0.05)
    parser.add_argument("--background-fraction", type=float, default=0.25)
    parser.add_argument("--hard-prior-threshold", type=float, default=0.50)
    parser.add_argument("--transition-prior-threshold", type=float, default=0.20)
    return parser.parse_args()


def ensure_inputs(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required inputs are missing: {missing}")


def clear_generated_output(output_dir: Path) -> None:
    """Remove only artifacts that match this tool's exact output contract."""

    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")

    root_files = {"protocol.json", "realization_summary.csv"}
    dataset_stems = {"negative_samples_rebuilt", "combined_samples_rebuilt"}
    shapefile_suffixes = {".shp", ".shx", ".dbf", ".prj", ".cpg"}
    recognized_files: list[Path] = []
    realization_dirs: list[Path] = []
    unexpected: list[Path] = []
    for child in output_dir.iterdir():
        if child.is_symlink():
            unexpected.append(child)
        elif child.is_file() and child.name in root_files:
            recognized_files.append(child)
        elif child.is_dir() and re.fullmatch(r"realization_\d+", child.name):
            realization_dirs.append(child)
            for artifact in child.iterdir():
                if artifact.is_symlink() or not artifact.is_file():
                    unexpected.append(artifact)
                elif artifact.name == "manifest.json" or (
                    artifact.stem in dataset_stems
                    and artifact.suffix.lower() in shapefile_suffixes
                ):
                    recognized_files.append(artifact)
                else:
                    unexpected.append(artifact)
        else:
            unexpected.append(child)
    if unexpected:
        preview = ", ".join(str(path) for path in unexpected[:5])
        raise ValueError(
            "Refusing --overwrite because the output directory contains files "
            f"not owned by this generator: {preview}"
        )

    for artifact in recognized_files:
        artifact.unlink()
    for realization_dir in realization_dirs:
        realization_dir.rmdir()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = SamplingConfig(
        negative_ratio=args.negative_ratio,
        realizations=args.realizations,
        base_seed=args.base_seed,
        positive_buffer_m=args.positive_buffer_m,
        thinning_cell_m=args.thinning_cell_m,
        balance_block_m=args.balance_block_m,
        hard_fraction=args.hard_fraction,
        transition_fraction=args.transition_fraction,
        background_fraction=args.background_fraction,
        hard_prior_threshold=args.hard_prior_threshold,
        transition_prior_threshold=args.transition_prior_threshold,
    )
    config.validate()

    feature_files = sorted(args.feature_dir.glob("*.tif"))
    required_paths = [
        args.candidate_grid,
        args.positive_samples,
        args.known_zones,
        args.dome_tif,
        args.fault_tif,
        args.strata_tif,
        *feature_files,
        *args.lithology_zone,
    ]
    if args.structure_zone is not None:
        required_paths.append(args.structure_zone)
    ensure_inputs(required_paths)
    if not feature_files:
        raise FileNotFoundError(f"No feature rasters found in {args.feature_dir}")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {args.output_dir}")
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            "Output directory is not empty; choose a new path or pass --overwrite: "
            f"{args.output_dir}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.overwrite:
        LOGGER.info(
            "Clearing recognized generated artifacts because --overwrite was set: %s",
            args.output_dir,
        )
        clear_generated_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading candidate grid: %s", args.candidate_grid)
    candidate_gdf = pyogrio.read_dataframe(args.candidate_grid, columns=[])
    if candidate_gdf.empty or candidate_gdf.crs is None:
        raise ValueError("Candidate grid must be non-empty and have a CRS.")
    if not candidate_gdf.crs.is_projected:
        raise ValueError("Candidate grid must use a projected CRS.")
    if not (candidate_gdf.geom_type == "Point").all():
        raise ValueError("Candidate grid must contain Point geometry.")
    crs = candidate_gdf.crs
    x = candidate_gdf.geometry.x.to_numpy(dtype=np.float64)
    y = candidate_gdf.geometry.y.to_numpy(dtype=np.float64)
    del candidate_gdf

    positive_source = pyogrio.read_dataframe(args.positive_samples)
    if args.label_col not in positive_source.columns:
        raise ValueError(f"Missing label field: {args.label_col}")
    if positive_source.crs is None:
        raise ValueError("Positive samples must have a CRS.")
    if str(positive_source.crs) != str(crs):
        positive_source = positive_source.to_crs(crs)
    positives = positive_source.loc[
        positive_source[args.label_col].astype(int) == 1, ["geometry"]
    ].copy()
    positives = gpd.GeoDataFrame(positives, geometry="geometry", crs=crs)
    if positives.empty:
        raise ValueError("No positive samples were found.")

    mine_union, zones = load_polygon_union(
        args.known_zones,
        crs,
        args.assume_zone_crs,
        deduplicate=True,
    )
    positive_buffer = mine_union.buffer(config.positive_buffer_m)
    structure_union: BaseGeometry | None = None
    if args.structure_zone is not None:
        structure_union, _ = load_polygon_union(
            args.structure_zone, crs, args.assume_zone_crs
        )
    lithology_union: BaseGeometry | None = None
    for lithology_path in args.lithology_zone:
        layer_union, _ = load_polygon_union(
            lithology_path, crs, args.assume_zone_crs
        )
        lithology_union = (
            layer_union
            if lithology_union is None
            else lithology_union.union(layer_union)
        )

    LOGGER.info("Sampling geological priors for %d candidates.", len(x))
    p_dome = prior_from_distance(sample_raster_values(x, y, args.dome_tif), "dome")
    p_fault = prior_from_distance(sample_raster_values(x, y, args.fault_tif), "fault")
    p_strata = prior_from_distance(sample_raster_values(x, y, args.strata_tif), "strata")
    geo_score = np.nanmax(np.column_stack([p_dome, p_fault, p_strata]), axis=1)
    prior_valid = np.isfinite(geo_score)
    feature_valid = raster_valid_mask(x, y, feature_files)
    # Use exact point-to-polygon distance for the scientific threshold. A buffered
    # polygon alone can be short by about one metre because curved buffer segments
    # are represented by finite chords.
    candidate_mine_distance = distance(points(x, y), mine_union)
    excluded_buffer = candidate_mine_distance < config.positive_buffer_m
    in_structure = (
        contains_xy(structure_union, x, y)
        if structure_union is not None
        else np.zeros(len(x), dtype=bool)
    )
    in_lithology = (
        contains_xy(lithology_union, x, y)
        if lithology_union is not None
        else np.zeros(len(x), dtype=bool)
    )
    eligible = prior_valid & feature_valid & ~excluded_buffer
    strata = assign_strata(
        geo_score, in_structure, in_lithology, eligible, config
    )

    requested_negative = int(round(len(positives) * config.negative_ratio))
    quotas = compute_quotas(requested_negative, config)
    candidate_counts = {
        key: int((strata == key).sum())
        for key in ("hard", "transition", "background")
    }
    LOGGER.info(
        "Positive=%d; requested negative=%d; candidate strata=%s; quotas=%s",
        len(positives),
        requested_negative,
        candidate_counts,
        quotas,
    )

    positive_metadata = build_positive_metadata(
        positives,
        zones,
        args.dome_tif,
        args.fault_tif,
        args.strata_tif,
        structure_union,
        lithology_union,
    )
    origin = (float(x.min()), float(y.min()))
    fixed_block_ids = spatial_cell_ids(
        x, y, config.balance_block_m, origin
    )
    summaries: list[dict[str, int | float | str]] = []

    for realization in range(config.realizations):
        seed = config.base_seed + realization * PSEUDO_ABSENCE_SEED_STEP
        rng = np.random.default_rng(seed)
        offset = tuple(rng.uniform(0.0, config.thinning_cell_m, size=2))
        thin_ids = spatial_cell_ids(
            x, y, config.thinning_cell_m, origin, offset
        )
        selected: dict[str, np.ndarray] = {}
        used_thin_ids: set[int] = set()
        # Scarce strata are selected first so hard negatives cannot consume their cells.
        for stratum in ("transition", "background", "hard"):
            candidate_indices = np.flatnonzero(strata == stratum)
            chosen = select_spatially_balanced(
                candidate_indices,
                thin_ids,
                fixed_block_ids,
                quotas[stratum],
                rng,
                used_thin_ids,
            )
            selected[stratum] = chosen
            used_thin_ids.update(int(value) for value in thin_ids[chosen])

        negatives = make_negative_frame(
            selected,
            realization,
            x,
            y,
            thin_ids,
            fixed_block_ids,
            geo_score,
            p_dome,
            p_fault,
            p_strata,
            in_structure,
            in_lithology,
            mine_union,
            crs,
        )
        positives_this_run = positive_metadata.copy()
        positives_this_run["realiz"] = realization
        combined = pd.concat([positives_this_run, negatives], ignore_index=True)
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=crs)
        combined = combined.iloc[rng.permutation(len(combined))].reset_index(drop=True)
        validation = validate_training_samples(
            combined,
            expected_positive=len(positives),
            expected_negative=requested_negative,
            positive_buffer=positive_buffer,
            minimum_negative_distance_m=config.positive_buffer_m,
        )

        realization_dir = args.output_dir / f"realization_{realization:02d}"
        negative_path = realization_dir / "negative_samples_rebuilt.shp"
        combined_path = realization_dir / "combined_samples_rebuilt.shp"
        write_shapefile(negatives, negative_path)
        write_shapefile(combined, combined_path)
        manifest = {
            "realization": realization,
            "seed": seed,
            "thin_grid_offset_m": [float(offset[0]), float(offset[1])],
            "class_counts": {
                "positive": len(positives),
                "negative": requested_negative,
            },
            "negative_strata": {
                key: int((negatives["stratum"] == key).sum()) for key in quotas
            },
            "validation": validation,
            "negative_min_mine_distance_m": float(negatives["mine_dist"].min()),
            "negative_unique_balance_blocks": int(negatives["block_id"].nunique()),
            "negative_unique_thinning_cells": int(negatives["thin_id"].nunique()),
            "files": {
                "negative_samples": str(negative_path),
                "combined_samples": str(combined_path),
            },
            "hashes": {
                "negative_samples": shapefile_hashes(negative_path),
                "combined_samples": shapefile_hashes(combined_path),
            },
        }
        with (realization_dir / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        summaries.append(
            {
                "realization": realization,
                "seed": seed,
                "positive": len(positives),
                "negative": requested_negative,
                "hard": quotas["hard"],
                "transition": quotas["transition"],
                "background": quotas["background"],
                "unique_blocks": int(negatives["block_id"].nunique()),
                "unique_thin_cells": int(negatives["thin_id"].nunique()),
                "min_mine_distance_m": float(negatives["mine_dist"].min()),
                "mean_geo_score": float(negatives["geo_score"].mean()),
            }
        )
        LOGGER.info("Wrote realization %02d to %s", realization, realization_dir)

    protocol = {
        "protocol_name": "geology-overlap spatially balanced pseudo-absence reconstruction",
        "config": asdict(config),
        "inputs": {
            "candidate_grid": str(args.candidate_grid),
            "positive_samples": str(args.positive_samples),
            "known_zones": str(args.known_zones),
            "structure_zone": str(args.structure_zone) if args.structure_zone else None,
            "lithology_zones": [str(path) for path in args.lithology_zone],
            "feature_rasters": [str(path) for path in feature_files],
            "prior_rasters": {
                "dome": str(args.dome_tif),
                "fault": str(args.fault_tif),
                "strata": str(args.strata_tif),
            },
        },
        "input_hashes": {
            "candidate_grid": shapefile_hashes(args.candidate_grid),
            "positive_samples": shapefile_hashes(args.positive_samples),
            "known_zones": shapefile_hashes(args.known_zones),
            "structure_zone": (
                shapefile_hashes(args.structure_zone)
                if args.structure_zone is not None
                else None
            ),
            "lithology_zones": {
                str(path): shapefile_hashes(path) for path in args.lithology_zone
            },
            "feature_rasters": {
                path.name: file_sha256(path) for path in feature_files
            },
            "prior_rasters": {
                "dome": file_sha256(args.dome_tif),
                "fault": file_sha256(args.fault_tif),
                "strata": file_sha256(args.strata_tif),
            },
        },
        "candidate_counts": {
            "total": int(len(x)),
            "excluded_positive_buffer": int(excluded_buffer.sum()),
            "invalid_feature_or_prior": int((~(prior_valid & feature_valid)).sum()),
            "eligible": int(eligible.sum()),
            **candidate_counts,
        },
        "positive_units": {
            "positive_grid_cells": int(len(positives)),
            "unique_known_zones": int(len(zones)),
        },
        "quotas_per_realization": quotas,
        "method_note": (
            "High-favorability structural and lithological settings are included "
            "as hard pseudo-absences rather than excluded from the negative class."
        ),
    }
    with (args.output_dir / "protocol.json").open("w", encoding="utf-8") as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "realization_summary.csv", index=False, encoding="utf-8-sig"
    )
    LOGGER.info("Completed %d negative-sample realizations.", config.realizations)


if __name__ == "__main__":
    main()
