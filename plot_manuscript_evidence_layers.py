"""Render the coordinate-suppressed nine-panel evidence figure used in the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

EVIDENCE_LAYERS = (
    ("Norm_clipped_AGG_ERG_Rad_Pot.tif", "K radiometric anomaly"),
    ("Norm_clipped_AGG_ERG_Mag_TMI_RTP_HD_TDR.tif", "Magnetic RTP HD-TDR"),
    (
        "Norm_clipped_AGG_ERG_Mag_TMI_RTP_V1D.tif",
        "Magnetic RTP first vertical derivative",
    ),
    ("Norm_clipped_AGG_ERG_Mag_TMI.tif", "Magnetic TMI"),
    ("Norm_clipped_AGG_ERG_Rad_Tho.tif", "Th radiometric anomaly"),
    ("Norm_clipped_AGG_ERG_Mag_TMI_RTP_TDR.tif", "Magnetic RTP-TDR"),
    ("Norm_clipped_AGG_ERG_Mag_TMI_Ana.tif", "Magnetic analytic signal"),
    ("Norm_clipped_AGG_ERG_Mag_TMI_RTP.tif", "Magnetic RTP"),
    ("Norm_clipped_AGG_ERG_Rad_Ura.tif", "U radiometric anomaly"),
)
MILLIMETRES_TO_INCHES = 1 / 25.4
FULL_WIDTH_INCHES = 190 * MILLIMETRES_TO_INCHES
FIGURE_HEIGHT_INCHES = 206 * MILLIMETRES_TO_INCHES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def suppress_coordinate_information(axis: mpl.axes.Axes) -> None:
    """Hide coordinate labels, ticks, tick marks and grid without changing extent."""

    axis.grid(False)
    axis.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")


def add_north_arrow(axis: mpl.axes.Axes, x: float = 0.90, y: float = 0.91) -> None:
    axis.add_patch(
        mpl.patches.Rectangle(
            (x - 0.035, y - 0.155),
            0.07,
            0.19,
            transform=axis.transAxes,
            facecolor="white",
            edgecolor="none",
            alpha=0.76,
            zorder=9,
        )
    )
    axis.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - 0.14),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "#172238", "lw": 1.0},
        zorder=10,
    )


def add_scale_bar(axis: mpl.axes.Axes, length_km: float = 10.0) -> None:
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    length = length_km * 1000.0
    start_x = xmin + 0.07 * (xmax - xmin)
    start_y = ymin + 0.075 * (ymax - ymin)
    axis.plot(
        [start_x, start_x + length],
        [start_y, start_y],
        color="#172238",
        linewidth=2.0,
        solid_capstyle="butt",
        zorder=10,
    )
    axis.text(
        start_x + length / 2,
        start_y + 0.02 * (ymax - ymin),
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#172238",
        bbox={
            "boxstyle": "square,pad=0.15",
            "fc": "white",
            "ec": "none",
            "alpha": 0.76,
        },
        zorder=10,
    )


def _alignment_signature(dataset: rasterio.io.DatasetReader) -> dict[str, Any]:
    return {
        "crs": dataset.crs.to_string() if dataset.crs else None,
        "bounds": [float(value) for value in dataset.bounds],
        "transform": [float(value) for value in dataset.transform[:6]],
        "width": int(dataset.width),
        "height": int(dataset.height),
    }


def render_evidence_layers(
    feature_dir: Path,
    output_stem: Path,
    *,
    dpi: int = 600,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create PDF, PNG and provenance manifest without resampling source rasters."""

    if dpi < 72:
        raise ValueError("dpi must be at least 72.")
    source_paths = [feature_dir / filename for filename, _ in EVIDENCE_LAYERS]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing evidence rasters: {missing}")

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    manifest_path = output_stem.with_suffix(".export.json")
    outputs = (pdf_path, png_path, manifest_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Outputs already exist; pass --overwrite: " + ", ".join(existing)
        )
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(FULL_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    color_map = mpl.colormaps["viridis"].copy()
    color_map.set_bad("#E5E5E5")
    image = None
    source_records: list[dict[str, Any]] = []
    reference_alignment: dict[str, Any] | None = None

    for index, ((filename, title), path) in enumerate(
        zip(EVIDENCE_LAYERS, source_paths, strict=True)
    ):
        axis = axes.flat[index]
        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise ValueError(f"Expected one raster band: {path}")
            if dataset.crs is None or not dataset.crs.is_projected:
                raise ValueError(f"Evidence raster requires a projected CRS: {path}")
            data = dataset.read(1, masked=True)
            alignment = _alignment_signature(dataset)
            extent = (
                dataset.bounds.left,
                dataset.bounds.right,
                dataset.bounds.bottom,
                dataset.bounds.top,
            )
        finite = np.asarray(data.compressed(), dtype=np.float64)
        if finite.size == 0:
            raise ValueError(f"Evidence raster contains no finite cells: {path}")
        if finite.min() < -1e-6 or finite.max() > 1.0 + 1e-6:
            raise ValueError(f"Evidence raster is not normalized to [0, 1]: {path}")
        if reference_alignment is None:
            reference_alignment = alignment
        elif alignment != reference_alignment:
            raise ValueError(
                f"Evidence raster alignment differs from the first layer: {path}"
            )

        image = axis.imshow(
            data,
            extent=extent,
            origin="upper",
            cmap=color_map,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            rasterized=True,
        )
        axis.set_title(
            f"({chr(97 + index)}) {title}",
            loc="left",
            fontsize=6.1,
            fontweight="bold",
            pad=2.5,
        )
        axis.set_aspect("equal")
        suppress_coordinate_information(axis)
        source_records.append(
            {
                "filename": filename,
                "sha256": sha256_file(path),
                "alignment": alignment,
                "finite_min": float(finite.min()),
                "finite_max": float(finite.max()),
            }
        )

    if image is None:
        raise RuntimeError("No evidence layers were rendered.")
    add_north_arrow(axes[0, 0])
    add_scale_bar(axes[2, 2], 10.0)
    colorbar = figure.colorbar(
        image, ax=axes, orientation="horizontal", fraction=0.025, pad=0.035, aspect=38
    )
    colorbar.set_label("Normalized response", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.5, length=2)
    metadata = {
        "Title": "figure_2_evidence_layers",
        "Author": "ORGEO-D-26-00764 authors",
    }
    figure.savefig(pdf_path, facecolor="white", metadata=metadata)
    figure.savefig(png_path, dpi=dpi, facecolor="white", metadata=metadata)
    plt.close(figure)

    manifest = {
        "schema_version": "1.0",
        "manuscript_id": "ORGEO-D-26-00764",
        "figure": "Figure 2: normalized airborne radiometric and aeromagnetic evidence layers",
        "sources": source_records,
        "transformations": [
            "masked read of native normalized rasters",
            "common fixed 0-1 color normalization",
            "coordinate labels, coordinate ticks, tick marks and coordinate grid suppressed",
            "native raster extent and alignment retained; no resampling, smoothing or cropping",
        ],
        "orientation_and_scale": {"north_arrow": True, "scale_bar_km": 10.0},
        "outputs": {
            "pdf": {"filename": pdf_path.name, "sha256": sha256_file(pdf_path)},
            "png": {
                "filename": png_path.name,
                "sha256": sha256_file(png_path),
                "dpi": int(dpi),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=root / "data" / "factors")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=root / "publication_figures" / "figure_2_evidence_layers",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = render_evidence_layers(
        args.feature_dir,
        args.output_stem,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
