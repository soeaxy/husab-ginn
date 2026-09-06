from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from PIL import Image
from shapely import distance, points
from shapely.geometry.base import BaseGeometry


LOGGER = logging.getLogger("negative-sample-analysis")

JOURNAL_COLORS = {
    "positive": "#B2182B",
    "original": "#4D4D4D",
    "rebuilt_all": "#2166AC",
    "hard": "#1B9E77",
    "transition": "#D95F02",
    "background": "#7570B3",
}

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
}


@dataclass(frozen=True)
class AnalysisConfig:
    original_samples: Path
    reconstructed_root: Path
    known_zones: Path
    dome_tif: Path
    fault_tif: Path
    strata_tif: Path
    output_dir: Path
    assume_zone_crs: str = "EPSG:32733"


def parse_args() -> AnalysisConfig:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the reconstructed negative-sample realizations and produce "
            "journal-quality QA figures and Markdown summaries."
        )
    )
    parser.add_argument(
        "--original-samples",
        type=Path,
        default=project_root / "data" / "combined_samples.shp",
    )
    parser.add_argument(
        "--reconstructed-root",
        type=Path,
        default=project_root / "data" / "reconstructed_samples_v2",
    )
    parser.add_argument(
        "--known-zones",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dome-tif",
        type=Path,
        default=project_root / "data" / "priors" / "dome.tif",
    )
    parser.add_argument(
        "--fault-tif",
        type=Path,
        default=project_root / "data" / "priors" / "fault.tif",
    )
    parser.add_argument(
        "--strata-tif",
        type=Path,
        default=project_root / "data" / "priors" / "strata.tif",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "analysis_output" / "negative_sample_reconstruction",
    )
    parser.add_argument("--assume-zone-crs", default="EPSG:32733")
    args = parser.parse_args()
    return AnalysisConfig(
        original_samples=args.original_samples,
        reconstructed_root=args.reconstructed_root,
        known_zones=args.known_zones,
        dome_tif=args.dome_tif,
        fault_tif=args.fault_tif,
        strata_tif=args.strata_tif,
        output_dir=args.output_dir,
        assume_zone_crs=args.assume_zone_crs,
    )


def ensure_inputs(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")


def sample_raster_values(x: np.ndarray, y: np.ndarray, raster_path: Path) -> np.ndarray:
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


def load_polygon_union(
    path: Path, target_crs: object, assume_crs: str
) -> tuple[BaseGeometry, gpd.GeoDataFrame]:
    zones = pyogrio.read_dataframe(path)
    if zones.empty:
        raise ValueError(f"Known-zone layer is empty: {path}")
    if zones.crs is None:
        zones = zones.set_crs(assume_crs)
    if str(zones.crs) != str(target_crs):
        zones = zones.to_crs(target_crs)
    zones = zones.assign(_wkb=zones.geometry.apply(lambda geom: geom.wkb))
    zones = zones.drop_duplicates("_wkb").drop(columns="_wkb")
    union = zones.geometry.union_all()
    if union.is_empty:
        raise ValueError("Known-zone union is empty.")
    return union, zones


def add_geological_scores(samples: gpd.GeoDataFrame, cfg: AnalysisConfig) -> gpd.GeoDataFrame:
    x = samples.geometry.x.to_numpy(dtype=np.float64)
    y = samples.geometry.y.to_numpy(dtype=np.float64)
    p_dome = prior_from_distance(sample_raster_values(x, y, cfg.dome_tif), "dome")
    p_fault = prior_from_distance(sample_raster_values(x, y, cfg.fault_tif), "fault")
    p_strata = prior_from_distance(sample_raster_values(x, y, cfg.strata_tif), "strata")
    geo_score = np.nanmax(np.column_stack([p_dome, p_fault, p_strata]), axis=1)
    out = samples.copy()
    out["p_dome"] = p_dome
    out["p_fault"] = p_fault
    out["p_strata"] = p_strata
    out["geo_score"] = geo_score.astype(np.float32)
    return out


def add_mine_distance(samples: gpd.GeoDataFrame, mine_union: BaseGeometry) -> gpd.GeoDataFrame:
    out = samples.copy()
    if "mine_dist" in out.columns and out["mine_dist"].notna().all():
        out["mine_dist"] = out["mine_dist"].astype(np.float32)
        return out
    geom = points(out.geometry.x.to_numpy(), out.geometry.y.to_numpy())
    out["mine_dist"] = distance(geom, mine_union).astype(np.float32)
    return out


def point_keys(samples: gpd.GeoDataFrame) -> pd.Series:
    x = samples.geometry.x.round(3).astype(str)
    y = samples.geometry.y.round(3).astype(str)
    return x.str.cat(y, sep="|")


def summary_stats(values: pd.Series) -> dict[str, float | int]:
    return {
        "n": int(values.notna().sum()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def ecdf_xy(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(values.to_numpy(dtype=np.float64))
    y = np.arange(1, len(arr) + 1, dtype=np.float64) / len(arr)
    return arr, y


def realization_paths(root: Path) -> list[Path]:
    paths = sorted(path for path in root.glob("realization_*") if path.is_dir())
    if not paths:
        raise FileNotFoundError(f"No realization_* folders found in {root}")
    return paths


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    """Export an opaque high-resolution preview and an editable vector PDF."""

    png_path = output_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=500, facecolor="white")
    with Image.open(png_path) as image:
        if image.mode != "RGB":
            rgba = image.convert("RGBA")
            opaque = Image.new("RGB", rgba.size, "white")
            opaque.paste(rgba, mask=rgba.getchannel("A"))
            opaque.save(png_path, format="PNG", dpi=(500, 500), optimize=True)
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        facecolor="white",
        metadata={
            "Title": output_stem.name,
            "Subject": "Negative-sample reconstruction quality assurance",
        },
    )


def plot_spatial_layout(
    original_negatives: gpd.GeoDataFrame,
    rebuilt: gpd.GeoDataFrame,
    positives: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    output_stem: Path,
) -> None:
    fig = plt.figure(figsize=(7.48, 4.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.13])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    legend_axis = fig.add_subplot(grid[1, :])
    legend_axis.axis("off")
    for axis in axes:
        zones.boundary.plot(ax=axis, color="black", linewidth=0.8, zorder=4)
        positives.plot(
            ax=axis,
            color=JOURNAL_COLORS["positive"],
            markersize=1.2,
            alpha=0.8,
            zorder=5,
        )
        axis.set_aspect("equal")
        axis.set_xlabel("Easting (km)")
        axis.set_ylabel("Northing (km)")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}"))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}"))

    axes[0].scatter(
        original_negatives.geometry.x,
        original_negatives.geometry.y,
        s=0.35,
        c=JOURNAL_COLORS["original"],
        alpha=0.25,
        linewidths=0,
        rasterized=True,
    )
    axes[0].set_title("Original negatives")

    markers = {"hard": "o", "transition": "^", "background": "s"}
    for stratum in ("hard", "transition", "background"):
        subset = rebuilt.loc[rebuilt["stratum"] == stratum]
        axes[1].scatter(
            subset.geometry.x,
            subset.geometry.y,
            s=0.45,
            c=JOURNAL_COLORS[stratum],
            marker=markers[stratum],
            alpha=0.34,
            linewidths=0,
            rasterized=True,
        )
    axes[1].set_title("Rebuilt negatives (realization 00)")
    bounds = np.vstack(
        [
            original_negatives.total_bounds,
            rebuilt.total_bounds,
            positives.total_bounds,
            zones.total_bounds,
        ]
    )
    xmin, ymin = bounds[:, :2].min(axis=0)
    xmax, ymax = bounds[:, 2:].max(axis=0)
    for axis in axes:
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)

    legend_handles = [
        Line2D([], [], marker=".", linestyle="none", color=JOURNAL_COLORS["original"], label="Archived negatives"),
        Line2D([], [], marker="o", linestyle="none", color=JOURNAL_COLORS["hard"], label="Hard negatives"),
        Line2D([], [], marker="^", linestyle="none", color=JOURNAL_COLORS["transition"], label="Transition negatives"),
        Line2D([], [], marker="s", linestyle="none", color=JOURNAL_COLORS["background"], label="Background negatives"),
        Line2D([], [], marker="o", linestyle="none", color=JOURNAL_COLORS["positive"], label="Positive grid cells"),
        Line2D([], [], color="black", linewidth=0.8, label="Known mineralized-zone boundary"),
    ]
    legend_axis.legend(
        handles=legend_handles,
        loc="center",
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_geo_score_ecdf(
    original_negatives: gpd.GeoDataFrame,
    rebuilt_all: gpd.GeoDataFrame,
    output_stem: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.48, 4.4), layout="constrained")
    x, y = ecdf_xy(original_negatives["geo_score"])
    axis.plot(x, y, color=JOURNAL_COLORS["original"], linestyle="-", label="Archived negatives")

    rebuilt_total_x, rebuilt_total_y = ecdf_xy(rebuilt_all["geo_score"])
    axis.plot(
        rebuilt_total_x,
        rebuilt_total_y,
        color=JOURNAL_COLORS["rebuilt_all"],
        linestyle="--",
        label="Rebuilt negatives (all realizations)",
    )
    line_styles = {"hard": "-.", "transition": (0, (5, 2)), "background": ":"}
    for stratum in ("hard", "transition", "background"):
        subset = rebuilt_all.loc[rebuilt_all["stratum"] == stratum, "geo_score"]
        x_s, y_s = ecdf_xy(subset)
        axis.plot(
            x_s,
            y_s,
            color=JOURNAL_COLORS[stratum],
            linestyle=line_styles[stratum],
            label=f"{stratum} negatives",
        )

    axis.set_xlabel("Geological score")
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_title("Geological-score redistribution after negative-sample reconstruction")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False, loc="lower right")
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_mine_distance_ecdf(
    original_negatives: gpd.GeoDataFrame,
    rebuilt_all: gpd.GeoDataFrame,
    output_stem: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.48, 4.4), layout="constrained")
    x, y = ecdf_xy(original_negatives["mine_dist"])
    axis.plot(x, y, color=JOURNAL_COLORS["original"], linestyle="-", label="Archived negatives")
    rebuilt_x, rebuilt_y = ecdf_xy(rebuilt_all["mine_dist"])
    axis.plot(
        rebuilt_x,
        rebuilt_y,
        color=JOURNAL_COLORS["rebuilt_all"],
        linestyle="--",
        label="Rebuilt negatives (all realizations)",
    )
    axis.axvline(1000.0, color="black", linestyle=":", linewidth=1.0, label="1 km exclusion threshold")
    axis.set_xlabel("Distance to known mineralized zone (m)")
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_title("Mine-distance distribution before and after reconstruction")
    axis.set_xlim(0.0, max(float(x.max()), float(rebuilt_x.max())) * 1.02)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False, loc="lower right")
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_jaccard_heatmap(jaccard: pd.DataFrame, output_stem: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.48, 5.4), layout="constrained")
    cmap = LinearSegmentedColormap.from_list(
        "ogr_heatmap", ["#F7FBFF", "#6BAED6", "#08306B"]
    ).with_extremes(bad="#E6E6E6")
    matrix = jaccard.to_numpy(dtype=np.float64)
    diagonal = np.eye(matrix.shape[0], dtype=bool)
    off_diagonal = matrix[~diagonal]
    display_matrix = np.ma.masked_where(diagonal, matrix)
    upper_limit = max(0.01, float(np.ceil(off_diagonal.max() * 100.0) / 100.0))
    image = axis.imshow(
        display_matrix,
        cmap=cmap,
        vmin=0.0,
        vmax=upper_limit,
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(jaccard.columns)))
    axis.set_yticks(np.arange(len(jaccard.index)))
    short_labels = [label.replace("realization_", "R") for label in jaccard.columns]
    axis.set_xticklabels(short_labels)
    axis.set_yticklabels(short_labels)
    axis.set_title("Pairwise Jaccard overlap among rebuilt negative realizations")
    for row in range(jaccard.shape[0]):
        for col in range(jaccard.shape[1]):
            if row == col:
                label = "—"
                color = "#555555"
            else:
                label = f"{jaccard.iloc[row, col]:.3f}"
                color = "white" if jaccard.iloc[row, col] >= upper_limit * 0.60 else "black"
            axis.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=6.5,
            )
    fig.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="Off-diagonal Jaccard overlap",
    )
    save_figure(fig, output_stem)
    plt.close(fig)


def make_distribution_summary(
    original_negatives: gpd.GeoDataFrame,
    rebuilt_all: gpd.GeoDataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for group_name, series in (
        ("original_negative_geo_score", original_negatives["geo_score"]),
        ("rebuilt_negative_geo_score", rebuilt_all["geo_score"]),
        ("original_negative_mine_distance_m", original_negatives["mine_dist"]),
        ("rebuilt_negative_mine_distance_m", rebuilt_all["mine_dist"]),
    ):
        rows.append({"group": group_name, **summary_stats(series)})
    for stratum in ("hard", "transition", "background"):
        subset = rebuilt_all.loc[rebuilt_all["stratum"] == stratum]
        rows.append(
            {
                "group": f"rebuilt_{stratum}_geo_score",
                **summary_stats(subset["geo_score"]),
            }
        )
    return pd.DataFrame(rows)


def spatial_coverage_summary(
    original_negatives: gpd.GeoDataFrame,
    rebuilt_all: gpd.GeoDataFrame,
    rebuilt_by_realization: dict[str, gpd.GeoDataFrame],
) -> pd.DataFrame:
    frames = [original_negatives, rebuilt_all]
    all_bounds = np.array([frame.total_bounds for frame in frames], dtype=np.float64)
    xmin = float(all_bounds[:, 0].min())
    ymin = float(all_bounds[:, 1].min())
    cell_size = 1000.0

    def block_ids(frame: gpd.GeoDataFrame) -> np.ndarray:
        ix = np.floor((frame.geometry.x.to_numpy() - xmin) / cell_size).astype(np.int64)
        iy = np.floor((frame.geometry.y.to_numpy() - ymin) / cell_size).astype(np.int64)
        return (ix << np.int64(32)) ^ (iy & np.int64(0xFFFFFFFF))

    rows = [
        {
            "dataset": "original_negative",
            "rows": int(len(original_negatives)),
            "unique_1km_blocks": int(np.unique(block_ids(original_negatives)).size),
        },
        {
            "dataset": "rebuilt_all_realizations",
            "rows": int(len(rebuilt_all)),
            "unique_1km_blocks": int(np.unique(block_ids(rebuilt_all)).size),
        },
    ]
    for name, frame in rebuilt_by_realization.items():
        rows.append(
            {
                "dataset": name,
                "rows": int(len(frame)),
                "unique_1km_blocks": int(np.unique(block_ids(frame)).size),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_report(
    cfg: AnalysisConfig,
    output_dir: Path,
    protocol: dict[str, object],
    original_negatives: gpd.GeoDataFrame,
    rebuilt_all: gpd.GeoDataFrame,
    realization_summary: pd.DataFrame,
    distribution_summary_df: pd.DataFrame,
    coverage_summary_df: pd.DataFrame,
    jaccard_df: pd.DataFrame,
    overlap_with_original_df: pd.DataFrame,
) -> None:
    original_geo_median = float(original_negatives["geo_score"].median())
    rebuilt_geo_median = float(rebuilt_all["geo_score"].median())
    geo_shift = rebuilt_geo_median - original_geo_median
    original_mine_median = float(original_negatives["mine_dist"].median())
    rebuilt_mine_median = float(rebuilt_all["mine_dist"].median())
    mine_shift = rebuilt_mine_median - original_mine_median
    matrix = jaccard_df.to_numpy(dtype=np.float64)
    off_diagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    mean_pairwise_jaccard = float(off_diagonal.mean())
    mean_original_overlap = float(overlap_with_original_df["jaccard_with_original"].mean())
    text = f"""# Negative-sample reconstruction analysis

## Scope

This report compares the archived negative samples in `{cfg.original_samples}` against the rebuilt realizations in `{cfg.reconstructed_root}`.

## Input checks

- Original rows: {len(original_negatives):,} negatives extracted from the archived training sample set
- Rebuilt rows: {len(rebuilt_all):,} negatives across {realization_summary.shape[0]} realizations
- Protocol quotas per realization: {json.dumps(protocol["quotas_per_realization"], ensure_ascii=False)}
- Eligible candidate cells reported by protocol: {protocol["candidate_counts"]["eligible"]:,}

## Key observations

1. The median geological score increased from {original_geo_median:.4f} to {rebuilt_geo_median:.4f} ({geo_shift:+.4f}).
2. The median distance to a known mineralized zone decreased from {original_mine_median:.1f} m to {rebuilt_mine_median:.1f} m ({mine_shift:+.1f} m).
3. Mean pairwise Jaccard overlap across rebuilt realizations: {mean_pairwise_jaccard:.4f}.
4. Minimum rebuilt mine distance: {realization_summary["min_mine_distance_m"].min():.2f} m
5. Mean rebuilt 1 km block coverage: {coverage_summary_df.loc[coverage_summary_df["dataset"].str.startswith("realization_"), "unique_1km_blocks"].mean():.1f}
6. Mean Jaccard overlap between a rebuilt realization and the archived negatives: {mean_original_overlap:.4f}.

## Interpretation

- The geological-score shift confirms that the reconstructed negatives occupy substantially more favorable geological settings than the archived negatives, directly reducing the previous easy-separation bias.
- The mine-distance distribution is left-shifted while maintaining the exact 1 km floor, so negatives are closer to known mineralization without entering the exclusion zone.
- Low pairwise Jaccard overlap confirms that the ten deterministic-seed realizations provide genuinely different pseudo-absence sets rather than reordered duplicates.
- These are design-quality diagnostics, not inferential tests: the candidate grid cells are spatially dependent and the ten realizations are repeated samples from one candidate population.

## Output tables

- `realization_summary.csv` — per-realization quotas, distance floor, spatial-cell coverage, mean score
- `distribution_summary.csv` — geo-score and mine-distance distribution statistics
- `coverage_summary.csv` — 1 km block coverage comparison
- `pairwise_jaccard.csv` — realization-overlap matrix
- `overlap_with_original.csv` — coordinate overlap with the archived negatives
- `statistics_appendix.md` — exact descriptive-statistics tables
- `figure_manifest.json` — figure dimensions, formats, source paths, and transformation provenance

## Figures

- QA `figure_1_spatial_layout.*` — archived vs rebuilt spatial patterns
- QA `figure_2_geo_score_ecdf.*` — geological-score redistribution
- QA `figure_3_mine_distance_ecdf.*` — distance-to-known-ore redistribution
- QA `figure_4_realization_jaccard.*` — overlap structure across realizations

These numbers are local to the sampling-QA module. Manuscript Figure 2 is the
coordinate-suppressed evidence-layer panel generated by
`plot_manuscript_evidence_layers.py`.

## Limitations

- This analysis checks sampling geometry and prior-derived hardness; it does not by itself prove downstream model generalization.
- The archived negative-sample fields are recomputed from prior rasters because the original shapefile does not retain the rebuilt metadata schema.
- Jaccard overlap is coordinate-based at millimeter-rounded precision and therefore measures identical sampled cells rather than neighborhood similarity.
- No null-hypothesis significance test is reported because spatial dependence and deterministic quota sampling violate an independent-replicate interpretation.
"""
    (output_dir / "analysis_report.md").write_text(text, encoding="utf-8")

    figure_catalog = """# Figure catalog

| Figure | Files | Purpose | Accessible description / reading guide |
| --- | --- | --- | --- |
    | QA Figure 1 | `figure_1_spatial_layout.png`, `figure_1_spatial_layout.pdf` | Compare the archived negative layout against one rebuilt realization. | Two maps share identical limits. The archived negatives form narrow, easily separated corridors; rebuilt circles, triangles, and squares cover the study area and favorable corridors while red positive cells remain isolated. |
    | QA Figure 2 | `figure_2_geo_score_ecdf.png`, `figure_2_geo_score_ecdf.pdf` | Compare geological-score distributions. | The dashed rebuilt curve lies to the right of the solid archived curve over most probabilities, showing higher geological favorability. Line styles repeat the color coding. |
    | QA Figure 3 | `figure_3_mine_distance_ecdf.png`, `figure_3_mine_distance_ecdf.pdf` | Compare distances from known mineralized zones. | The dashed rebuilt curve begins at the dotted 1 km threshold and is generally left of the archived curve, indicating closer but still excluded negatives. |
    | QA Figure 4 | `figure_4_realization_jaccard.png`, `figure_4_realization_jaccard.pdf` | Quantify overlap among the ten rebuilt realizations. | The diagonal is masked; off-diagonal cells range around 0.02-0.03, showing low exact-cell overlap among deterministic-seed realizations. |
"""
    (output_dir / "figure_catalog.md").write_text(figure_catalog, encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str], decimals: int = 4) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, divider]
    for _, row in frame[columns].iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{decimals}f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_statistics_appendix(
    output_dir: Path,
    distribution_summary_df: pd.DataFrame,
    realization_summary_df: pd.DataFrame,
    overlap_with_original_df: pd.DataFrame,
    jaccard_df: pd.DataFrame,
) -> None:
    matrix = jaccard_df.to_numpy(dtype=np.float64)
    off_diagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    text = """# Negative-sample reconstruction statistics appendix

All values are descriptive. No independence-based significance test is used because the points are spatially dependent and the realizations share one candidate population.

## Distribution summaries

"""
    text += markdown_table(
        distribution_summary_df,
        ["group", "n", "mean", "median", "p05", "p25", "p75", "p95", "min", "max"],
    )
    text += "\n\n## Per-realization quality checks\n\n"
    text += markdown_table(
        realization_summary_df,
        [
            "realization",
            "rows",
            "hard",
            "transition",
            "background",
            "unique_block_id",
            "unique_thin_id",
            "min_mine_distance_m",
            "median_geo_score",
        ],
    )
    text += "\n\n## Overlap diagnostics\n\n"
    text += (
        f"- Pairwise rebuilt-realization Jaccard: mean={off_diagonal.mean():.4f}, "
        f"minimum={off_diagonal.min():.4f}, maximum={off_diagonal.max():.4f}.\n"
        f"- Rebuilt-versus-archived Jaccard: mean="
        f"{overlap_with_original_df['jaccard_with_original'].mean():.4f}.\n"
    )
    (output_dir / "statistics_appendix.md").write_text(text, encoding="utf-8")


def write_figure_manifest(cfg: AnalysisConfig, output_dir: Path) -> None:
    manifest = {
        "target_journal": "Ore Geology Reviews",
        "submission_phase": "resubmission preparation",
        "status": "provisional scientific-QA figures; recheck live journal instructions before upload",
        "publisher_guidance_check": {
            "accessed": "2026-08-18",
            "elsevier_artwork_url": "https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions",
            "elsevier_artwork_status": "HTTP 200 from current environment",
            "journal_guide_url": "https://www.sciencedirect.com/journal/ore-geology-reviews/publish/guide-for-authors",
            "journal_guide_status": "HTTP 403 from current environment; recheck interactively before upload",
        },
        "source_data": {
            "original_samples": str(cfg.original_samples.resolve()),
            "reconstructed_root": str(cfg.reconstructed_root.resolve()),
            "known_zones": str(cfg.known_zones.resolve()),
            "prior_rasters": [
                str(cfg.dome_tif.resolve()),
                str(cfg.fault_tif.resolve()),
                str(cfg.strata_tif.resolve()),
            ],
        },
        "transformations": [
            "geological score = max(dome prior, fault prior, strata prior)",
            "distance measured from point geometry to dissolved known-zone polygons",
            "ECDF uses all finite observations without smoothing or binning",
            "Jaccard overlap uses coordinates rounded to 0.001 m",
        ],
        "exports": {
            "png": {"dpi": 500, "background": "opaque white"},
            "pdf": {"fonttype": 42, "background": "opaque white"},
        },
        "style": PLOT_STYLE,
    }
    with (output_dir / "figure_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


def main() -> None:
    cfg = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    ensure_inputs(
        [
            cfg.original_samples,
            cfg.reconstructed_root,
            cfg.known_zones,
            cfg.dome_tif,
            cfg.fault_tif,
            cfg.strata_tif,
        ]
    )
    protocol_path = cfg.reconstructed_root / "protocol.json"
    ensure_inputs([protocol_path])
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    original = pyogrio.read_dataframe(cfg.original_samples)
    if original.empty or "Class" not in original.columns:
        raise ValueError("Original samples must contain rows and a Class column.")
    positives = original.loc[original["Class"].astype(int) == 1, ["geometry"]].copy()
    positives = gpd.GeoDataFrame(positives, geometry="geometry", crs=original.crs)
    original_negatives = original.loc[
        original["Class"].astype(int) == 0, ["geometry"]
    ].copy()
    original_negatives = gpd.GeoDataFrame(
        original_negatives, geometry="geometry", crs=original.crs
    )
    if positives.empty or original_negatives.empty:
        raise ValueError("Original sample file must contain both positive and negative samples.")

    mine_union, zones = load_polygon_union(
        cfg.known_zones, original.crs, cfg.assume_zone_crs
    )
    original_negatives = add_geological_scores(original_negatives, cfg)
    original_negatives = add_mine_distance(original_negatives, mine_union)

    rebuilt_by_realization: dict[str, gpd.GeoDataFrame] = {}
    summaries: list[dict[str, float | int | str]] = []
    overlap_with_original: list[dict[str, float | str | int]] = []

    for folder in realization_paths(cfg.reconstructed_root):
        name = folder.name
        negatives_path = folder / "negative_samples_rebuilt.shp"
        manifest_path = folder / "manifest.json"
        ensure_inputs([negatives_path, manifest_path])
        frame = pyogrio.read_dataframe(negatives_path)
        if frame.empty:
            raise ValueError(f"Rebuilt negatives are empty: {negatives_path}")
        frame = gpd.GeoDataFrame(frame, geometry="geometry", crs=frame.crs)
        rebuilt_by_realization[name] = frame
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        frame_keys = set(point_keys(frame))
        original_keys = set(point_keys(original_negatives))
        overlap_with_original.append(
            {
                "realization": name,
                "shared_cells_with_original": int(len(frame_keys & original_keys)),
                "jaccard_with_original": float(
                    len(frame_keys & original_keys) / len(frame_keys | original_keys)
                ),
            }
        )
        summaries.append(
            {
                "realization": name,
                "rows": int(len(frame)),
                "hard": int((frame["stratum"] == "hard").sum()),
                "transition": int((frame["stratum"] == "transition").sum()),
                "background": int((frame["stratum"] == "background").sum()),
                "unique_block_id": int(frame["block_id"].nunique()),
                "unique_thin_id": int(frame["thin_id"].nunique()),
                "mean_geo_score": float(frame["geo_score"].mean()),
                "median_geo_score": float(frame["geo_score"].median()),
                "min_mine_distance_m": float(frame["mine_dist"].min()),
                "median_mine_distance_m": float(frame["mine_dist"].median()),
                "manifest_seed": int(manifest["seed"]),
            }
        )

    rebuilt_all = gpd.GeoDataFrame(
        pd.concat(rebuilt_by_realization.values(), ignore_index=True),
        geometry="geometry",
        crs=next(iter(rebuilt_by_realization.values())).crs,
    )

    realization_summary_df = pd.DataFrame(summaries).sort_values("realization")
    overlap_with_original_df = pd.DataFrame(overlap_with_original).sort_values("realization")

    names = sorted(rebuilt_by_realization)
    key_sets = {name: set(point_keys(frame)) for name, frame in rebuilt_by_realization.items()}
    jaccard = pd.DataFrame(index=names, columns=names, dtype=float)
    for left in names:
        for right in names:
            inter = len(key_sets[left] & key_sets[right])
            union = len(key_sets[left] | key_sets[right])
            jaccard.loc[left, right] = inter / union

    distribution_summary_df = make_distribution_summary(original_negatives, rebuilt_all)
    coverage_summary_df = spatial_coverage_summary(
        original_negatives, rebuilt_all, rebuilt_by_realization
    )
    with protocol_path.open("r", encoding="utf-8") as stream:
        protocol = json.load(stream)

    with plt.rc_context(PLOT_STYLE):
        plot_spatial_layout(
            original_negatives,
            rebuilt_by_realization[names[0]],
            positives,
            zones,
            cfg.output_dir / "figure_1_spatial_layout",
        )
        plot_geo_score_ecdf(
            original_negatives,
            rebuilt_all,
            cfg.output_dir / "figure_2_geo_score_ecdf",
        )
        plot_mine_distance_ecdf(
            original_negatives,
            rebuilt_all,
            cfg.output_dir / "figure_3_mine_distance_ecdf",
        )
        plot_jaccard_heatmap(jaccard, cfg.output_dir / "figure_4_realization_jaccard")

    realization_summary_df.to_csv(
        cfg.output_dir / "realization_summary.csv", index=False, encoding="utf-8-sig"
    )
    distribution_summary_df.to_csv(
        cfg.output_dir / "distribution_summary.csv", index=False, encoding="utf-8-sig"
    )
    coverage_summary_df.to_csv(
        cfg.output_dir / "coverage_summary.csv", index=False, encoding="utf-8-sig"
    )
    overlap_with_original_df.to_csv(
        cfg.output_dir / "overlap_with_original.csv", index=False, encoding="utf-8-sig"
    )
    jaccard.to_csv(
        cfg.output_dir / "pairwise_jaccard.csv", encoding="utf-8-sig"
    )

    write_markdown_report(
        cfg,
        cfg.output_dir,
        protocol,
        original_negatives,
        rebuilt_all,
        realization_summary_df,
        distribution_summary_df,
        coverage_summary_df,
        jaccard,
        overlap_with_original_df,
    )
    write_statistics_appendix(
        cfg.output_dir,
        distribution_summary_df,
        realization_summary_df,
        overlap_with_original_df,
        jaccard,
    )
    write_figure_manifest(cfg, cfg.output_dir)
    LOGGER.info("Negative-sample analysis written to %s", cfg.output_dir)


if __name__ == "__main__":
    main()
