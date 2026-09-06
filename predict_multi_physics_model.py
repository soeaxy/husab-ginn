# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import rasterio
import rasterio.mask
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from manuscript_protocol import (
    GINN_INTERNAL_KEY,
    GINN_PUBLIC_KEY,
    manuscript_model_label,
    normalize_algorithm_key,
)
from mineral_deep_models import (
    DEEP_COMPARE_ALGOS,
    augment_with_knn_graph_features,
    build_spatial_ego_graphs,
    load_deep_compare_model,
    predict_probabilities_deep_compare,
    predict_probabilities_estimator,
    resolve_torch_device,
)
from physics_informed_model import GeologyInformedClassifier

CLASSICAL_ALGOS = {"rf", "lightgbm", "catboost", "xgboost"}
SUPPORTED_ALGOS = {GINN_INTERNAL_KEY} | DEEP_COMPARE_ALGOS | CLASSICAL_ALGOS


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Predict a mineral prospectivity map with a trained GINN or comparison model."
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=project_root / "data" / "factors",
    )
    parser.add_argument(
        "--research-shp",
        type=Path,
        default=project_root / "data" / "research" / "Husab_Square.shp",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=project_root
        / "results"
        / "models"
        / "mineral_model_multi_phy_final.pth",
    )
    parser.add_argument(
        "--preproc-path",
        type=Path,
        default=project_root / "results" / "models" / "preproc.joblib",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="auto",
        choices=[
            "auto",
            GINN_PUBLIC_KEY,
            GINN_INTERNAL_KEY,
            "rf",
            "lightgbm",
            "catboost",
            "xgboost",
            "mlp",
            "gnn",
            "vae",
            "transunet",
            "gcn_transformer",
        ],
        help="Inference algorithm. 'ginn' is the manuscript name; 'pinn' is a legacy alias; 'auto' reads preproc metadata.",
    )
    parser.add_argument("--output-tif", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"]
    )
    return parser


def discover_tif_files(feature_dir: Path) -> dict[str, Path]:
    tif_paths = sorted(feature_dir.glob("*.tif"))
    if not tif_paths:
        raise FileNotFoundError(f"No .tif found in {feature_dir}")
    return {path.stem: path for path in tif_paths}


def load_research_shapes(research_shp: Path) -> list[dict[str, Any]]:
    research_area = gpd.read_file(research_shp)
    if research_area.empty:
        raise ValueError("Research area shapefile is empty.")
    return [geom.__geo_interface__ for geom in research_area.geometry]


def build_feature_stack(
    required_feature_names: list[str],
    feature_map: dict[str, Path],
    shapes: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int, int]:
    missing = [name for name in required_feature_names if name not in feature_map]
    if missing:
        raise ValueError(f"Missing required feature rasters: {missing}")

    first_tif = feature_map[required_feature_names[0]]
    with rasterio.open(first_tif) as src0:
        template_arr, template_transform = rasterio.mask.mask(
            src0, shapes, crop=True, filled=True
        )
        profile = src0.profile.copy()
        profile.update(
            {
                "height": template_arr.shape[1],
                "width": template_arr.shape[2],
                "transform": template_transform,
                "dtype": "float32",
                "count": 1,
                "nodata": np.nan,
            }
        )
        out_crs = src0.crs

    h, w = profile["height"], profile["width"]
    bands: list[np.ndarray] = []
    valid_mask: np.ndarray | None = None

    for name in required_feature_names:
        tif_path = feature_map[name]
        with rasterio.open(tif_path) as src:
            if src.crs != out_crs:
                raise ValueError(
                    f"CRS mismatch for {tif_path}. Reproject to match template CRS first."
                )
            arr, transform = rasterio.mask.mask(src, shapes, crop=True, filled=True)
            if (
                transform != profile["transform"]
                or arr.shape[1] != h
                or arr.shape[2] != w
            ):
                raise ValueError(f"Shape/transform mismatch for {tif_path}.")
            data = arr[0].astype(np.float32)

            current_valid = np.isfinite(data)
            if src.nodata is not None:
                current_valid &= ~np.isclose(data, src.nodata)
            valid_mask = (
                current_valid if valid_mask is None else (valid_mask & current_valid)
            )
            bands.append(data)

    stack = np.stack(bands, axis=-1)
    assert valid_mask is not None
    flat_valid = valid_mask.reshape(-1)
    x_raw = stack.reshape(-1, stack.shape[-1])[flat_valid]
    x_raw = np.where(np.isfinite(x_raw), x_raw, np.nan).astype(np.float32)
    return x_raw, flat_valid, profile, h, w


def preprocess_features(
    x_raw: np.ndarray,
    preproc_obj: dict[str, Any],
) -> np.ndarray:
    imputer = preproc_obj.get("imputer")
    var_selector = preproc_obj.get("variance_selector")
    scaler = preproc_obj.get("scaler")

    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        x_imp = imputer.fit_transform(x_raw)
        logging.warning(
            "Imputer not found in preproc file; using fallback fit on inference data."
        )
    else:
        x_imp = imputer.transform(x_raw)

    if var_selector is not None:
        x_var = var_selector.transform(x_imp)
    else:
        x_var = x_imp
        logging.warning(
            "Variance selector not found in preproc file; skipping variance filtering."
        )

    if scaler is None:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_var)
        logging.warning(
            "Scaler not found in preproc file; using fallback fit on inference data."
        )
    else:
        x_scaled = scaler.transform(x_var)

    return x_scaled.astype(np.float32)


def coordinates_for_valid_pixels(
    profile: dict[str, Any], flat_valid: np.ndarray, width: int
) -> np.ndarray:
    """Return centre coordinates for valid raster cells in the raster CRS."""
    indices = np.flatnonzero(flat_valid)
    rows = indices // width
    cols = indices % width
    transform = profile["transform"]
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return np.column_stack([xs, ys]).astype(np.float64)


def predict_probabilities(
    model: GeologyInformedClassifier,
    x_data: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    probs = np.empty(x_data.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, x_data.shape[0], batch_size):
            end = start + batch_size
            xb = torch.from_numpy(x_data[start:end]).to(device)
            logits = model(xb)
            probs[start:end] = (
                torch.softmax(logits, dim=1)[:, 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return probs


def resolve_inference_algorithm(
    preproc_obj: dict[str, Any], requested_algorithm: str
) -> str:
    algorithm = str(preproc_obj.get("algorithm", GINN_INTERNAL_KEY)).lower()
    if requested_algorithm != "auto":
        algorithm = requested_algorithm.lower()
    algorithm = normalize_algorithm_key(algorithm)
    if algorithm not in SUPPORTED_ALGOS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return algorithm


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    device = resolve_torch_device(args.device)
    logging.info("Using device: %s", device)

    preproc_obj = joblib.load(args.preproc_path)
    all_feature_names = preproc_obj.get("all_feature_names")
    kept_feature_names = preproc_obj.get("kept_feature_names")
    if all_feature_names is None or kept_feature_names is None:
        raise ValueError(
            "preproc.joblib must contain both 'all_feature_names' and 'kept_feature_names'."
        )
    all_feature_names = list(all_feature_names)
    kept_feature_names = list(kept_feature_names)

    feature_map = discover_tif_files(args.feature_dir)
    shapes = load_research_shapes(args.research_shp)

    logging.info("Loading and clipping %d required features...", len(all_feature_names))
    x_raw, flat_valid, profile, h, w = build_feature_stack(
        required_feature_names=all_feature_names,
        feature_map=feature_map,
        shapes=shapes,
    )
    x_processed = preprocess_features(x_raw, preproc_obj)
    logging.info("Prepared matrix shape: %s", tuple(x_processed.shape))

    input_dim = x_processed.shape[1]
    if input_dim != len(kept_feature_names):
        raise ValueError(
            f"Input dim mismatch: model input_dim would be {input_dim}, but kept_feature_names has {len(kept_feature_names)}."
        )

    algorithm = resolve_inference_algorithm(preproc_obj, args.algorithm)
    logging.info(
        "Inference algorithm: %s (internal key: %s)",
        manuscript_model_label(algorithm),
        algorithm,
    )

    if algorithm == GINN_INTERNAL_KEY:
        model = GeologyInformedClassifier(input_dim=input_dim).to(device)
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)
        probs = predict_probabilities(model, x_processed, args.batch_size, device)
    elif algorithm in DEEP_COMPARE_ALGOS:
        x_infer = x_processed
        if algorithm == "gnn":
            ref = preproc_obj.get("gnn_ref_features")
            k_neighbors = int(preproc_obj.get("gnn_k", 8))
            if ref is None:
                raise ValueError(
                    "preproc.joblib missing gnn_ref_features required by gnn inference."
                )
            ref = np.asarray(ref, dtype=np.float32)
            x_infer = augment_with_knn_graph_features(
                ref, x_processed, k_neighbors=k_neighbors
            )
        checkpoint = torch.load(args.model_path, map_location=device)
        model = load_deep_compare_model(algorithm, checkpoint, device)
        if algorithm == "gcn_transformer":
            ref_features = preproc_obj.get("gcn_transformer_ref_features")
            ref_coords = preproc_obj.get("gcn_transformer_ref_coords")
            graph_k = int(preproc_obj.get("gcn_transformer_k", 8))
            if ref_features is None or ref_coords is None:
                raise ValueError(
                    "preproc.joblib is missing GCN-Transformer training-reference features or coordinates."
                )
            query_coords = coordinates_for_valid_pixels(profile, flat_valid, w)
            ref_features = np.asarray(ref_features, dtype=np.float32)
            ref_coords = np.asarray(ref_coords, dtype=np.float64)
            parts: list[np.ndarray] = []
            graph_chunk = min(65536, max(1, int(args.batch_size)))
            for start in range(0, x_processed.shape[0], graph_chunk):
                stop = min(start + graph_chunk, x_processed.shape[0])
                ego_graphs = build_spatial_ego_graphs(
                    ref_features,
                    ref_coords,
                    x_processed[start:stop],
                    query_coords[start:stop],
                    k_neighbors=graph_k,
                )
                parts.append(
                    predict_probabilities_deep_compare(
                        model, ego_graphs, args.batch_size, device, algorithm
                    )
                )
            probs = np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)
        else:
            probs = predict_probabilities_deep_compare(
                model, x_infer, args.batch_size, device, algorithm
            )
    elif algorithm in CLASSICAL_ALGOS:
        model = joblib.load(args.model_path)
        probs = predict_probabilities_estimator(model, x_processed)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    out_arr = np.full(h * w, np.nan, dtype=np.float32)
    out_arr[flat_valid] = probs
    out_arr = out_arr.reshape(h, w)

    if args.output_tif is None:
        args.output_tif = Path(f"husab_mine_proba_{args.model_path.stem}.tif")

    out_profile = profile.copy()
    out_profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(args.output_tif, "w", **out_profile) as dst:
        dst.write(out_arr, 1)

    logging.info("Prediction TIFF saved: %s", args.output_tif.resolve())


if __name__ == "__main__":
    main()
