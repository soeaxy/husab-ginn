from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import zipfile
from html import escape as html_escape
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import rasterio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from matplotlib import cm
from matplotlib import image as mpl_image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import transform as rio_transform

from mineral_prediction_api.schemas import (
    FullJobRequest,
    JobResponse,
    PredictJobRequest,
    TrainJobRequest,
)

app = FastAPI(
    title="Mineral Prediction API",
    description="Async GINN and baseline training/prediction API for mineral prospectivity mapping.",
    version="0.2.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_with_legacy(primary: str, legacy: str, default: str) -> str:
    return os.environ.get(primary) or os.environ.get(legacy) or default


TRAIN_SCRIPT = Path(
    _env_with_legacy(
        "MINERAL_TRAIN_SCRIPT",
        "PINN_TRAIN_SCRIPT",
        str(PROJECT_ROOT / "train_multi_physics_model.py"),
    )
)
PREDICT_SCRIPT = Path(
    _env_with_legacy(
        "MINERAL_PREDICT_SCRIPT",
        "PINN_PREDICT_SCRIPT",
        str(PROJECT_ROOT / "predict_multi_physics_model.py"),
    )
)
RUNS_DIR = Path(_env_with_legacy("MINERAL_RUNS_DIR", "PINN_RUNS_DIR", str(PROJECT_ROOT / "runs")))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
_MAX_LOG_LINES = 2500
_VALID_COLORMAPS = {"turbo", "viridis", "plasma", "magma", "inferno", "cividis"}

_INDEX_HTML = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/><title>Mineral Prediction API</title></head><body><h3>Mineral Prediction API</h3><p>Open <a href="/docs">/docs</a></p></body></html>"""
_UI_INDEX_PATH = Path(__file__).resolve().parent / "static" / "index.html"
if _UI_INDEX_PATH.is_file():
    _INDEX_HTML = _UI_INDEX_PATH.read_text(encoding="utf-8")
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


def _load_index_html() -> str:
    if _UI_INDEX_PATH.is_file():
        try:
            return _UI_INDEX_PATH.read_text(encoding="utf-8")
        except Exception:
            return _INDEX_HTML
    return _INDEX_HTML


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _algo_display_name(algorithm: str | None) -> str:
    mapping = {
        "ginn": "地质约束神经网络（GINN）",
        "pinn": "地质约束神经网络（GINN）",
        "rf": "随机森林",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "xgboost": "XGBoost",
        "mlp": "多层感知机",
        "gnn": "图神经网络",
        "vae": "变分自编码器",
        "transunet": "TransUNet",
        "gcn_transformer": "GCN-Transformer",
    }
    key = str(algorithm or "").lower()
    return mapping.get(key, key.upper() if key else "模型")


def _make_job_display_name(job_type: str, algorithm: str | None, model_name: str | None) -> str:
    ts = time.strftime("%m%d %H:%M", time.localtime())
    stage = "训练预测"
    algo_text = _algo_display_name(algorithm)
    if model_name:
        return f"{algo_text} {stage}任务 {ts} · {model_name}"
    return f"{algo_text} {stage}任务 {ts}"


def _append_log(job_id: str, line: str) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job["logs"].append(line.rstrip("\n"))
        if len(job["logs"]) > _MAX_LOG_LINES:
            job["logs"] = job["logs"][-_MAX_LOG_LINES:]


def _set_status(job_id: str, status: str) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job:
            job["status"] = status


def _set_error(job_id: str, error: str) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job:
            job["error"] = error


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _serialize_job(job: dict[str, Any]) -> dict[str, Any]:
    out = dict(job)
    out["job_dir"] = str(out["job_dir"])
    out["result_zip"] = str(out["result_zip"]) if out.get("result_zip") else None
    out["artifacts_root"] = str(out["artifacts_root"]) if out.get("artifacts_root") else None
    return out


def _normalize_output_root(output_root: str | None, fallback: Path) -> Path:
    return Path(output_root).expanduser().resolve() if output_root else fallback.resolve()


def _save_upload(upload: UploadFile, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def _safe_extract(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    abs_dest = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            target = (dest_dir / member).resolve()
            if abs_dest not in target.parents and target != abs_dest:
                raise ValueError("ZIP contains unsafe paths.")
        zf.extractall(dest_dir)


def _find_single_shp(root_dir: Path) -> Path:
    candidates = sorted(root_dir.rglob("*.shp"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) == 0:
        raise ValueError(f"No .shp found in {root_dir}")
    raise ValueError(f"Multiple .shp found in {root_dir}, keep exactly one.")


def _find_factors_dir(root_dir: Path) -> Path:
    tif_paths = sorted(list(root_dir.rglob("*.tif")) + list(root_dir.rglob("*.tiff")))
    if not tif_paths:
        raise ValueError(f"No .tif/.tiff found in {root_dir}")
    parents = sorted({p.parent.resolve() for p in tif_paths})
    if len(parents) == 1:
        return parents[0]
    raise ValueError(
        "Multiple folders contain tif files. Put factors into one directory inside the zip."
    )


def _find_prior_tif(priors_root: Path, pattern: str, label: str) -> Path:
    tif_paths = sorted(list(priors_root.rglob("*.tif")) + list(priors_root.rglob("*.tiff")))
    if not tif_paths:
        raise ValueError(f"No prior tif found in {priors_root}")

    pattern = pattern.strip()
    if not pattern:
        raise ValueError(f"{label} pattern cannot be empty.")

    lower_pattern = pattern.lower()
    exact_match = [p for p in tif_paths if p.name.lower() == lower_pattern]
    if len(exact_match) == 1:
        return exact_match[0].resolve()
    if len(exact_match) > 1:
        raise ValueError(f"Multiple exact matches for {label}: {pattern}")

    tokens = [tok.strip().lower() for tok in pattern.split(",") if tok.strip()]
    matched = [p for p in tif_paths if any(tok in p.stem.lower() for tok in tokens)]
    if len(matched) == 1:
        return matched[0].resolve()
    if len(matched) == 0:
        raise ValueError(f"No prior tif matched for {label} with pattern '{pattern}'.")
    raise ValueError(f"Multiple prior tif matched for {label} with pattern '{pattern}'.")


def _validate_train_priors(req: TrainJobRequest) -> None:
    if req.algorithm in {"ginn", "pinn"} and not (
        req.dome_tif or req.fault_tif or req.strata_tif or req.singularity_tif
    ):
        raise HTTPException(
            400,
            f"Algorithm '{req.algorithm}' requires at least one prior tif; the manuscript configuration uses dome/fault/strata and treats singularity as legacy optional input.",
        )


def _prepare_upload_train_inputs(
    *,
    input_root: Path,
    factors_zip_path: Path,
    samples_zip_path: Path,
    priors_zip_path: Path,
    dome_pattern: str,
    fault_pattern: str,
    strata_pattern: str,
    singularity_pattern: str,
    use_dome: bool,
    use_fault: bool,
    use_strata: bool,
    use_singularity: bool,
    require_priors: bool = True,
) -> tuple[Path, Path, Path | None, Path | None, Path | None, Path | None]:
    if require_priors and not (use_dome or use_fault or use_strata or use_singularity):
        raise ValueError("At least one prior type must be selected.")
    factors_dir = input_root / "factors"
    samples_dir = input_root / "samples"
    priors_dir = input_root / "priors"
    _safe_extract(factors_zip_path, factors_dir)
    _safe_extract(samples_zip_path, samples_dir)
    if use_dome or use_fault or use_strata or use_singularity:
        _safe_extract(priors_zip_path, priors_dir)

    resolved_factors_dir = _find_factors_dir(factors_dir)
    samples_shp = _find_single_shp(samples_dir)
    dome_tif = _find_prior_tif(priors_dir, dome_pattern, "dome") if use_dome else None
    fault_tif = _find_prior_tif(priors_dir, fault_pattern, "fault") if use_fault else None
    strata_tif = _find_prior_tif(priors_dir, strata_pattern, "strata") if use_strata else None
    singularity_tif = _find_prior_tif(priors_dir, singularity_pattern, "singularity") if use_singularity else None
    return resolved_factors_dir, samples_shp, dome_tif, fault_tif, strata_tif, singularity_tif


def _build_train_command(req: TrainJobRequest, output_root: Path) -> list[str]:
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--shapefile", req.shapefile]
    for feature_dir in req.feature_dirs:
        cmd.extend(["--feature-dir", feature_dir])

    if req.dome_tif:
        cmd.extend(["--dome-tif", req.dome_tif])
    if req.fault_tif:
        cmd.extend(["--fault-tif", req.fault_tif])
    if req.strata_tif:
        cmd.extend(["--strata-tif", req.strata_tif])
    if req.singularity_tif:
        cmd.extend(["--singularity-tif", req.singularity_tif])

    cmd.extend(
        [
            "--label-col",
            req.label_col,
            "--output-root",
            str(output_root),
            "--model-name",
            req.model_name,
            "--algorithm",
            req.algorithm,
            "--optimizer",
            req.optimizer,
            "--bayes-trials",
            str(req.bayes_trials),
            "--seed",
            str(req.seed),
            "--test-size",
            str(req.test_size),
            "--val-size",
            str(req.val_size),
            "--epochs",
            str(req.epochs),
            "--patience",
            str(req.patience),
            "--lr",
            str(req.lr),
            "--weight-decay",
            str(req.weight_decay),
            "--physics-weight",
            str(req.physics_weight),
            "--grad-clip-norm",
            str(req.grad_clip_norm),
            "--batch-size-cuda",
            str(req.batch_size_cuda),
            "--batch-size-cpu",
            str(req.batch_size_cpu),
            "--num-workers",
            str(req.num_workers),
            "--threshold",
            str(req.threshold),
            "--shap-background-size",
            str(req.shap_background_size),
            "--shap-eval-size",
            str(req.shap_eval_size),
            "--shap-nsamples",
            str(req.shap_nsamples),
        ]
    )
    if req.disable_shap:
        cmd.append("--disable-shap")
    return cmd


def _model_artifact_suffix(algorithm: str) -> str:
    return ".pth" if algorithm in {"ginn", "pinn", "mlp", "gnn", "vae", "transunet", "gcn_transformer"} else ".joblib"


def _build_predict_command(req: PredictJobRequest, output_tif: Path | None = None) -> list[str]:
    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--feature-dir",
        req.feature_dir,
        "--research-shp",
        req.research_shp,
        "--model-path",
        req.model_path,
        "--preproc-path",
        req.preproc_path,
        "--batch-size",
        str(req.batch_size),
    ]
    final_output = output_tif or (Path(req.output_tif).resolve() if req.output_tif else None)
    if final_output is not None:
        cmd.extend(["--output-tif", str(final_output)])
    return cmd


def _run_command(job_id: str, cmd: list[str], cwd: Path) -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    _append_log(job_id, f"$ {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        _append_log(job_id, line.rstrip("\n"))
    return process.wait()


def _collect_artifacts(root: Path) -> list[str]:
    if not root.exists():
        return []
    artifacts: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            artifacts.append(str(path.relative_to(root)).replace("\\", "/"))
    return artifacts


def _resolve_prediction_tif(job: dict[str, Any]) -> Path | None:
    output_tif = job.get("output_tif")
    if output_tif:
        candidate = Path(output_tif).expanduser().resolve()
        if candidate.is_file():
            return candidate

    job_dir = Path(job["job_dir"]).resolve()
    candidates: list[tuple[int, float, Path]] = []
    for rel_path in job.get("artifacts", []):
        rel_lower = rel_path.lower()
        if not rel_lower.endswith((".tif", ".tiff")):
            continue
        path = (job_dir / rel_path).resolve()
        if not path.is_file():
            continue
        score = 0
        rel_posix = rel_path.replace("\\", "/").lower()
        if "output/predict/" in rel_posix:
            score += 2
        if "prob" in path.stem.lower() or "proba" in path.stem.lower():
            score += 2
        candidates.append((score, path.stat().st_mtime, path))

    if not candidates:
        for glob_pattern in ("output/predict/*.tif", "output/predict/*.tiff"):
            for path in job_dir.glob(glob_pattern):
                if path.is_file():
                    candidates.append((3, path.stat().st_mtime, path.resolve()))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _prediction_summary(tif_path: Path) -> dict[str, Any]:
    with rasterio.open(tif_path) as src:
        total_pixels = int(src.width * src.height)
        valid_pixels = 0
        value_sum = 0.0
        value_min = np.inf
        value_max = -np.inf

        for _, window in src.block_windows(1):
            block = src.read(1, window=window, masked=True).astype(np.float32)
            values = np.asarray(block.compressed(), dtype=np.float64)
            if values.size == 0:
                continue
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            valid_pixels += int(values.size)
            value_sum += float(values.sum())
            block_min = float(values.min())
            block_max = float(values.max())
            if block_min < value_min:
                value_min = block_min
            if block_max > value_max:
                value_max = block_max

        mean_value = float(value_sum / valid_pixels) if valid_pixels > 0 else None
        min_value = float(value_min) if valid_pixels > 0 else None
        max_value = float(value_max) if valid_pixels > 0 else None
        nodata_value = float(src.nodata) if src.nodata is not None else None

        bounds = [[None, None], [None, None]]
        if src.crs is not None:
            left, bottom, right, top = src.bounds
            try:
                from pyproj import Transformer

                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                corners = [
                    (left, bottom),
                    (left, top),
                    (right, bottom),
                    (right, top),
                ]
                lons: list[float] = []
                lats: list[float] = []
                for x, y in corners:
                    lon, lat = transformer.transform(x, y)
                    if not (math.isfinite(lon) and math.isfinite(lat)):
                        continue
                    lons.append(float(lon))
                    lats.append(float(lat))

                if lons and lats:
                    south = max(-90.0, min(90.0, min(lats)))
                    north = max(-90.0, min(90.0, max(lats)))
                    west = max(-180.0, min(180.0, min(lons)))
                    east = max(-180.0, min(180.0, max(lons)))
                    if south < north and west < east:
                        bounds = [[south, west], [north, east]]
            except Exception:
                pass

        return {
            "path": str(tif_path),
            "width": int(src.width),
            "height": int(src.height),
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixels,
            "valid_ratio": float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0,
            "min_probability": min_value,
            "max_probability": max_value,
            "mean_probability": mean_value,
            "nodata": nodata_value,
            "crs": str(src.crs) if src.crs else None,
            "bounds": bounds,
        }


def _render_prediction_preview(tif_path: Path, *, max_size: int, colormap: str) -> bytes:
    with rasterio.open(tif_path) as src:
        scale = min(1.0, float(max_size) / float(max(src.height, src.width)))
        out_h = max(1, int(round(src.height * scale)))
        out_w = max(1, int(round(src.width * scale)))
        data = src.read(
            1,
            masked=True,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

        dense = np.asarray(data.filled(np.nan), dtype=np.float32)
        invalid = ~np.isfinite(dense)
        if src.nodata is not None:
            invalid |= np.isclose(dense, src.nodata)

        norm = np.clip(dense, 0.0, 1.0)
        rgba = cm.get_cmap(colormap)(norm, bytes=True)
        rgba[invalid, 3] = 0

        buf = io.BytesIO()
        mpl_image.imsave(buf, rgba, format="png")
        return buf.getvalue()


def _resolve_job_artifact_path(job_id: str, artifact_path: str) -> tuple[Path, Path]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        job_dir = Path(job["job_dir"]).resolve()

    target = (job_dir / artifact_path).resolve()
    if job_dir not in target.parents and target != job_dir:
        raise HTTPException(400, "unsafe artifact path")
    if not target.is_file():
        raise HTTPException(404, "artifact not found")
    return job_dir, target


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
        return None
    except Exception:
        return None


def _find_artifact_by_tokens(artifacts: list[str], candidates: list[str]) -> str | None:
    lowered = [(raw, str(raw).lower()) for raw in (artifacts or [])]
    for token in candidates:
        key = str(token).lower()
        for raw, low in lowered:
            if key in low:
                return raw
    return None


def _artifact_url(job_id: str, artifact_path: str) -> str:
    parts = [quote(seg, safe="") for seg in str(artifact_path).replace("\\", "/").split("/") if seg]
    return f"/jobs/{job_id}/artifacts/" + "/".join(parts)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _fmt_number(value: Any, digits: int = 4) -> str:
    n = _to_float(value)
    if n is None:
        return "-"
    return f"{n:.{digits}f}"


def _fmt_percent(value: Any, digits: int = 2) -> str:
    n = _to_float(value)
    if n is None:
        return "-"
    return f"{n * 100:.{digits}f}%"


def _count_tif_files(paths: list[str]) -> int:
    if not isinstance(paths, list):
        return 0
    files: set[Path] = set()
    for raw in paths:
        try:
            root = Path(str(raw)).expanduser()
        except Exception:
            continue
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}:
                files.add(p.resolve())
    return len(files)


def _read_shap_top_features(csv_path: Path, top_n: int = 10) -> list[dict[str, Any]]:
    if top_n < 1 or not csv_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature = str(row.get("feature") or "").strip()
                score = _to_float(row.get("mean_abs_shap"))
                if not feature or score is None:
                    continue
                rows.append({"feature": feature, "mean_abs_shap": score})
    except Exception:
        return []
    rows.sort(key=lambda item: float(item["mean_abs_shap"]), reverse=True)
    return rows[:top_n]


def _build_report_documents(job_id: str, job: dict[str, Any], job_dir: Path, artifacts: list[str]) -> tuple[str, str, dict[str, Any]]:
    metrics_rel = _find_artifact_by_tokens(artifacts, ["metrics/metrics.json"])
    run_cfg_rel = _find_artifact_by_tokens(artifacts, ["metrics/run_config.json"])
    full_req_rel = _find_artifact_by_tokens(artifacts, ["config/full_request.json", "full_request.json"])
    model_params_rel = _find_artifact_by_tokens(artifacts, ["metrics/model_parameters.json"])
    lib_versions_rel = _find_artifact_by_tokens(artifacts, ["metrics/library_versions.json"])
    train_history_rel = _find_artifact_by_tokens(artifacts, ["metrics/train_history.csv"])
    cls_report_rel = _find_artifact_by_tokens(artifacts, ["metrics/classification_report.txt"])
    shap_csv_rel = _find_artifact_by_tokens(artifacts, ["explainability/shap_feature_importance.csv", "shap_feature_importance.csv"])
    roc_csv_rel = _find_artifact_by_tokens(artifacts, ["metrics/roc_curve_test.csv"])
    pr_csv_rel = _find_artifact_by_tokens(artifacts, ["metrics/pr_curve_test.csv"])
    loss_rel = _find_artifact_by_tokens(artifacts, ["figures/loss_curve.png", "loss_curve.png"])
    roc_rel = _find_artifact_by_tokens(artifacts, ["figures/roc_curve_test.png", "roc_curve_test.png"])
    pr_rel = _find_artifact_by_tokens(artifacts, ["figures/pr_curve_test.png", "pr_curve_test.png"])
    shap_bar_rel = _find_artifact_by_tokens(artifacts, ["explainability/shap_bar.png", "shap_bar.png"])
    shap_summary_rel = _find_artifact_by_tokens(artifacts, ["explainability/shap_summary.png", "shap_summary.png"])
    generated_at = _now_iso()

    def _norm_path(path: str | None) -> str:
        return str(path).replace("\\", "/") if path else "-"

    def _artifact_exists(rel_path: str | None) -> bool:
        if not rel_path:
            return False
        try:
            return (job_dir / rel_path).is_file()
        except Exception:
            return False

    def _status_text(flag: bool) -> str:
        return "已验证 / Verified" if flag else "缺失 / Missing"

    full_req = _read_json_if_exists(job_dir / "config" / "full_request.json") or {}
    if not full_req_rel and (job_dir / "config" / "full_request.json").is_file():
        full_req_rel = "config/full_request.json"
    metrics = (_read_json_if_exists(job_dir / metrics_rel) if metrics_rel else {}) or {}
    run_cfg = (_read_json_if_exists(job_dir / run_cfg_rel) if run_cfg_rel else {}) or {}
    train_req = full_req.get("train") if isinstance(full_req.get("train"), dict) else {}
    data_cfg = run_cfg.get("data_config") if isinstance(run_cfg.get("data_config"), dict) else {}
    train_cfg = run_cfg.get("train_config") if isinstance(run_cfg.get("train_config"), dict) else {}
    algo_cfg = run_cfg.get("algo_config") if isinstance(run_cfg.get("algo_config"), dict) else {}
    enabled_priors = run_cfg.get("enabled_priors") if isinstance(run_cfg.get("enabled_priors"), dict) else {}

    feature_dirs: list[str] = []
    if isinstance(train_req.get("feature_dirs"), list):
        feature_dirs = [str(x) for x in train_req["feature_dirs"]]
    elif isinstance(data_cfg.get("feature_dirs"), list):
        feature_dirs = [str(x) for x in data_cfg["feature_dirs"]]
    factor_count = _count_tif_files(feature_dirs)

    sample_shp_raw = str(train_req.get("shapefile") or data_cfg.get("shapefile_path") or "")
    research_shp_raw = str(full_req.get("predict_research_shp") or "")
    sample_shp = Path(sample_shp_raw).name if sample_shp_raw else "-"
    research_shp = Path(research_shp_raw).name if research_shp_raw else "-"

    algorithm = str(
        job.get("algorithm")
        or metrics.get("algorithm")
        or algo_cfg.get("algorithm")
        or train_req.get("algorithm")
        or ""
    ).lower()
    optimizer = str(
        job.get("optimizer")
        or metrics.get("optimizer")
        or algo_cfg.get("optimizer")
        or train_req.get("optimizer")
        or "-"
    )
    display_algo = _algo_display_name(algorithm)
    model_name = str(job.get("model_name") or train_req.get("model_name") or "-")

    prior_dome = bool(train_req.get("dome_tif")) or bool(enabled_priors.get("dome"))
    prior_fault = bool(train_req.get("fault_tif")) or bool(enabled_priors.get("fault"))
    prior_strata = bool(train_req.get("strata_tif")) or bool(enabled_priors.get("strata"))
    prior_singularity = bool(train_req.get("singularity_tif")) or bool(enabled_priors.get("singularity"))

    val_metrics = metrics.get("validation_metrics") if isinstance(metrics.get("validation_metrics"), dict) else {}
    test_metrics = metrics.get("test_metrics") if isinstance(metrics.get("test_metrics"), dict) else {}
    n_samples = test_metrics.get("n_samples") or val_metrics.get("n_samples") or "-"
    pos_ratio = test_metrics.get("positive_ratio")
    if pos_ratio is None:
        pos_ratio = val_metrics.get("positive_ratio")

    pred_summary: dict[str, Any] = {}
    pred_tif = _resolve_prediction_tif(job)
    if pred_tif is not None:
        try:
            pred_summary = _prediction_summary(pred_tif)
        except Exception:
            pred_summary = {}

    shap_rows: list[dict[str, Any]] = []
    if shap_csv_rel:
        shap_rows = _read_shap_top_features(job_dir / shap_csv_rel, top_n=10)

    metric_items: list[tuple[str, str]] = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("mcc", "MCC"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("brier_score", "Brier Score"),
        ("log_loss", "Log Loss"),
    ]

    test_auc = _to_float(test_metrics.get("roc_auc"))
    test_pr_auc = _to_float(test_metrics.get("pr_auc"))
    test_f1 = _to_float(test_metrics.get("f1"))
    test_recall = _to_float(test_metrics.get("recall"))
    test_precision = _to_float(test_metrics.get("precision"))
    val_auc = _to_float(val_metrics.get("roc_auc"))

    if test_auc is None:
        perf_note_cn = "本次任务未找到完整测试集 AUC，建议先检查 metrics.json 产物完整性。"
        perf_note_en = "Test ROC-AUC is unavailable; check the integrity of metrics.json first."
    elif test_auc >= 0.95:
        perf_note_cn = "模型在测试集上具备较强的区分能力（ROC-AUC >= 0.95）。"
        perf_note_en = "The model shows strong discrimination on the test set (ROC-AUC >= 0.95)."
    elif test_auc >= 0.90:
        perf_note_cn = "模型在测试集上表现良好（ROC-AUC >= 0.90），但仍有优化空间。"
        perf_note_en = "The model performs well on the test set (ROC-AUC >= 0.90), with room for improvement."
    else:
        perf_note_cn = "模型区分能力处于中等水平，建议从样本均衡与先验约束两方面继续优化。"
        perf_note_en = "The model shows moderate discrimination; optimize sample balance and prior constraints."

    if test_precision is not None and test_recall is not None:
        if test_recall + 0.08 < test_precision:
            pr_tradeoff_cn = "当前模型 precision 明显高于 recall，误报较少但漏检相对偏多。"
            pr_tradeoff_en = "Precision is clearly higher than recall: fewer false alarms but more missed positives."
        elif test_precision + 0.08 < test_recall:
            pr_tradeoff_cn = "当前模型 recall 明显高于 precision，能够捕获更多阳性但误报偏高。"
            pr_tradeoff_en = "Recall is clearly higher than precision: more positives captured but with more false alarms."
        else:
            pr_tradeoff_cn = "precision 与 recall 较为均衡。"
            pr_tradeoff_en = "Precision and recall are relatively balanced."
    else:
        pr_tradeoff_cn = "未读取到完整的 precision/recall 指标。"
        pr_tradeoff_en = "Precision/recall metrics are incomplete."

    if test_auc is not None and val_auc is not None:
        gap = abs(test_auc - val_auc)
        if gap <= 0.02:
            generalization_cn = f"验证与测试 AUC 差值为 {gap:.4f}，泛化稳定性较好。"
            generalization_en = f"The validation-test AUC gap is {gap:.4f}, indicating stable generalization."
        else:
            generalization_cn = f"验证与测试 AUC 差值为 {gap:.4f}，存在一定泛化波动，建议进一步排查过拟合。"
            generalization_en = f"The validation-test AUC gap is {gap:.4f}, showing generalization drift; investigate overfitting."
    else:
        generalization_cn = "未能计算验证/测试 AUC 差值。"
        generalization_en = "The validation-test AUC gap cannot be computed."

    valid_ratio = _to_float(pred_summary.get("valid_ratio"))
    if valid_ratio is None:
        raster_note_cn = "未读取到预测栅格统计信息。"
        raster_note_en = "Prediction raster statistics are unavailable."
    elif valid_ratio >= 0.98:
        raster_note_cn = "预测栅格有效像元占比较高，空间覆盖完整性较好。"
        raster_note_en = "The prediction raster has a high valid-pixel ratio, indicating good spatial coverage."
    elif valid_ratio >= 0.90:
        raster_note_cn = "预测栅格存在少量无效像元，建议复核掩膜与投影一致性。"
        raster_note_en = "The prediction raster contains a small amount of invalid pixels; verify mask and projection consistency."
    else:
        raster_note_cn = "预测栅格无效像元占比较高，建议优先检查输入栅格有效范围。"
        raster_note_en = "The prediction raster has too many invalid pixels; check valid extents of input rasters first."

    abstract_cn = [
        f"本报告针对任务 `{job_id}` 自动汇总了 {display_algo} 模型在 Husab 数据集上的训练与预测结果。",
        f"在测试集上，模型达到 ROC-AUC={_fmt_number(test_auc, 4)}、PR-AUC={_fmt_number(test_pr_auc, 4)}、F1={_fmt_number(test_f1, 4)}。",
        f"数据层面共使用 {factor_count if factor_count > 0 else '-'} 个因子栅格，先验信息启用情况为 Dome={prior_dome}、Fault={prior_fault}、Strata={prior_strata}。",
        "报告包含关键指标表、SHAP 特征重要性表、编号图件、结论摘要和可追溯清单，便于论文写作与结果复核。",
    ]
    abstract_en = [
        f"This report summarizes training and prediction outcomes for job `{job_id}` using {display_algo} on the Husab dataset.",
        f"On the test set, the model reaches ROC-AUC={_fmt_number(test_auc, 4)}, PR-AUC={_fmt_number(test_pr_auc, 4)}, and F1={_fmt_number(test_f1, 4)}.",
        f"A total of {factor_count if factor_count > 0 else '-'} factor rasters are used, with priors enabled as Dome={prior_dome}, Fault={prior_fault}, and Strata={prior_strata}.",
        "The report includes metrics tables, SHAP importance, numbered figures, executive conclusions, and a traceability checklist.",
    ]

    figures: list[dict[str, Any]] = [
        {
            "title_cn": "训练损失曲线",
            "title_en": "Training Loss Curve",
            "path": loss_rel,
            "url": _artifact_url(job_id, loss_rel) if loss_rel else None,
        },
        {
            "title_cn": "测试集 ROC 曲线",
            "title_en": "Test ROC Curve",
            "path": roc_rel,
            "url": _artifact_url(job_id, roc_rel) if roc_rel else None,
        },
        {
            "title_cn": "测试集 PR 曲线",
            "title_en": "Test PR Curve",
            "path": pr_rel,
            "url": _artifact_url(job_id, pr_rel) if pr_rel else None,
        },
        {
            "title_cn": "SHAP 特征重要性柱状图",
            "title_en": "SHAP Feature Importance Bar Plot",
            "path": shap_bar_rel,
            "url": _artifact_url(job_id, shap_bar_rel) if shap_bar_rel else None,
        },
        {
            "title_cn": "SHAP 汇总图",
            "title_en": "SHAP Summary Plot",
            "path": shap_summary_rel,
            "url": _artifact_url(job_id, shap_summary_rel) if shap_summary_rel else None,
        },
        {
            "title_cn": "预测概率栅格预览图",
            "title_en": "Prediction Probability Raster Preview",
            "path": "output/predict/probability.tif" if pred_tif else None,
            "url": f"/jobs/{job_id}/prediction/preview.png?max_size=1280&colormap=turbo" if pred_tif else None,
        },
    ]

    trace_rows: list[dict[str, Any]] = [
        {
            "item": "任务请求快照 / Unified request snapshot",
            "path": full_req_rel,
            "url": _artifact_url(job_id, full_req_rel) if full_req_rel else None,
            "verified": bool(full_req),
            "notes": "输入参数与上传路径 / request parameters and upload paths",
        },
        {
            "item": "训练运行配置 / runtime config",
            "path": run_cfg_rel,
            "url": _artifact_url(job_id, run_cfg_rel) if run_cfg_rel else None,
            "verified": bool(run_cfg_rel and run_cfg),
            "notes": "训练阶段生效配置 / effective training config",
        },
        {
            "item": "核心指标 / core metrics",
            "path": metrics_rel,
            "url": _artifact_url(job_id, metrics_rel) if metrics_rel else None,
            "verified": bool(metrics_rel and metrics),
            "notes": "指标与结论来源 / source of metrics and conclusions",
        },
        {
            "item": "分类报告 / classification report",
            "path": cls_report_rel,
            "url": _artifact_url(job_id, cls_report_rel) if cls_report_rel else None,
            "verified": _artifact_exists(cls_report_rel),
            "notes": "类别精度细项 / class-wise precision-recall details",
        },
        {
            "item": "训练历史 / train history",
            "path": train_history_rel,
            "url": _artifact_url(job_id, train_history_rel) if train_history_rel else None,
            "verified": _artifact_exists(train_history_rel),
            "notes": "收敛曲线来源 / optimization history source",
        },
        {
            "item": "模型参数 / model parameters",
            "path": model_params_rel,
            "url": _artifact_url(job_id, model_params_rel) if model_params_rel else None,
            "verified": _artifact_exists(model_params_rel),
            "notes": "模型参数摘要 / model and training settings",
        },
        {
            "item": "环境版本 / library versions",
            "path": lib_versions_rel,
            "url": _artifact_url(job_id, lib_versions_rel) if lib_versions_rel else None,
            "verified": _artifact_exists(lib_versions_rel),
            "notes": "环境复现依据 / environment reproducibility evidence",
        },
        {
            "item": "SHAP 数据 / SHAP csv",
            "path": shap_csv_rel,
            "url": _artifact_url(job_id, shap_csv_rel) if shap_csv_rel else None,
            "verified": _artifact_exists(shap_csv_rel),
            "notes": "解释性结果来源 / explainability source",
        },
        {
            "item": "ROC/PR 曲线数据 / ROC-PR csv",
            "path": f"{_norm_path(roc_csv_rel)} ; {_norm_path(pr_csv_rel)}",
            "url": _artifact_url(job_id, roc_csv_rel) if roc_csv_rel else (_artifact_url(job_id, pr_csv_rel) if pr_csv_rel else None),
            "verified": _artifact_exists(roc_csv_rel) and _artifact_exists(pr_csv_rel),
            "notes": "重绘曲线与核验 AUC / curve redraw and AUC cross-check",
        },
        {
            "item": "任务产物接口 / artifacts endpoint",
            "path": f"/jobs/{job_id}/artifacts",
            "url": f"/jobs/{job_id}/artifacts",
            "verified": True,
            "notes": "统一产物索引 / complete artifact index",
        },
    ]

    trace_verified = sum(1 for row in trace_rows if bool(row.get("verified")))
    trace_total = len(trace_rows)
    trace_missing = trace_total - trace_verified

    conclusion_cn = [
        f"测试集 ROC-AUC={_fmt_number(test_auc, 4)}，PR-AUC={_fmt_number(test_pr_auc, 4)}，F1={_fmt_number(test_f1, 4)}。",
        pr_tradeoff_cn,
        generalization_cn,
        raster_note_cn,
        f"可追溯清单通过 {trace_verified}/{trace_total} 项，缺失 {trace_missing} 项。",
    ]
    conclusion_en = [
        f"Test ROC-AUC={_fmt_number(test_auc, 4)}, PR-AUC={_fmt_number(test_pr_auc, 4)}, F1={_fmt_number(test_f1, 4)}.",
        pr_tradeoff_en,
        generalization_en,
        raster_note_en,
        f"Traceability checklist passed {trace_verified}/{trace_total} items, with {trace_missing} missing.",
    ]

    md_lines: list[str] = []
    md_lines.append("# 矿产预测标准实验报告 / Standard Experiment Report")
    md_lines.append("")
    md_lines.append("> 报告特性 / Features: 中英双语 / Bilingual · 图件编号 / Figure Numbering · 结论摘要 / Executive Summary · 可追溯清单 / Traceability Checklist")
    md_lines.append("")
    md_lines.append("## 0. 报告信息 / Report Metadata")
    md_lines.append("")
    md_lines.append(f"- 任务 ID: `{job_id}`")
    md_lines.append(f"- 生成时间 / Generated At: {generated_at}")
    md_lines.append(f"- 模型算法: {display_algo} (`{algorithm or '-'}`)")
    md_lines.append(f"- 模型名称: `{model_name}`")
    md_lines.append(f"- 可追溯检查 / Traceability: {trace_verified}/{trace_total} verified")
    md_lines.append("")
    md_lines.append("## 1. 中文摘要")
    md_lines.append("")
    md_lines.extend([f"- {line}" for line in abstract_cn])
    md_lines.append("")
    md_lines.append("## 2. English Abstract")
    md_lines.append("")
    md_lines.extend([f"- {line}" for line in abstract_en])
    md_lines.append("")
    md_lines.append("## 3. 数据集介绍 / Dataset Overview")
    md_lines.append("")
    md_lines.append("| 项目 | 内容 |")
    md_lines.append("| --- | --- |")
    md_lines.append(f"| 样本 Shapefile | `{sample_shp}` |")
    md_lines.append(f"| 研究区 Shapefile | `{research_shp}` |")
    md_lines.append(f"| 因子目录数量 | {len(feature_dirs)} |")
    md_lines.append(f"| 因子栅格数量（.tif/.tiff） | {factor_count if factor_count > 0 else '-'} |")
    md_lines.append(f"| 样本量（n_samples） | {n_samples} |")
    md_lines.append(f"| 阳性样本占比 | {_fmt_percent(pos_ratio, 2)} |")
    md_lines.append(
        f"| 先验启用（Dome/Fault/Strata/Singularity） | {prior_dome} / {prior_fault} / {prior_strata} / {prior_singularity} |"
    )
    md_lines.append("")
    md_lines.append("## 3. 方法介绍 / Method")
    md_lines.append("")
    md_lines.append(f"本任务采用 **{display_algo}** 算法，优化方式为 `{optimizer}`。")
    md_lines.append("")
    md_lines.append("| 参数 | 数值 |")
    md_lines.append("| --- | --- |")
    md_lines.append(f"| epochs | {train_req.get('epochs', train_cfg.get('epochs', '-'))} |")
    md_lines.append(f"| patience | {train_req.get('patience', train_cfg.get('patience', '-'))} |")
    md_lines.append(f"| learning_rate | {train_req.get('lr', train_cfg.get('lr', '-'))} |")
    md_lines.append(f"| weight_decay | {train_req.get('weight_decay', train_cfg.get('weight_decay', '-'))} |")
    md_lines.append(f"| threshold | {train_req.get('threshold', train_cfg.get('threshold', '-'))} |")
    md_lines.append(f"| batch_size_cuda | {train_req.get('batch_size_cuda', train_cfg.get('batch_size_cuda', '-'))} |")
    md_lines.append(f"| batch_size_cpu | {train_req.get('batch_size_cpu', train_cfg.get('batch_size_cpu', '-'))} |")
    md_lines.append(f"| physics_weight | {train_req.get('physics_weight', train_cfg.get('physics_weight', '-'))} |")
    md_lines.append("")
    md_lines.append("## 4. 结果 / Results")
    md_lines.append("")
    md_lines.append("### 4.1 关键指标表 / Key Metrics")
    md_lines.append("")
    md_lines.append("| 指标 | Validation | Test |")
    md_lines.append("| --- | ---: | ---: |")
    for key, label_cn, label_en in metric_items:
        md_lines.append(f"| {label_cn} / {label_en} (`{key}`) | {_fmt_number(val_metrics.get(key), 4)} | {_fmt_number(test_metrics.get(key), 4)} |")
    md_lines.append("")

    cm = test_metrics.get("confusion_matrix")
    if isinstance(cm, list) and len(cm) >= 2 and all(isinstance(row, list) and len(row) >= 2 for row in cm[:2]):
        md_lines.append("### 4.2 测试集混淆矩阵 / Confusion Matrix")
        md_lines.append("")
        md_lines.append("|  | Pred=0 | Pred=1 |")
        md_lines.append("| --- | ---: | ---: |")
        md_lines.append(f"| True=0 | {cm[0][0]} | {cm[0][1]} |")
        md_lines.append(f"| True=1 | {cm[1][0]} | {cm[1][1]} |")
        md_lines.append("")

    md_lines.append("### 4.3 图件索引与图件 / Figure Index and Figures")
    md_lines.append("")
    md_lines.append("| 编号 / No. | 图题 / Caption | 证据路径 / Evidence | 状态 / Status |")
    md_lines.append("| ---: | --- | --- | --- |")
    for idx, fig in enumerate(figures, start=1):
        fig_title = f"{fig['title_cn']} / {fig['title_en']}"
        fig_path = _norm_path(fig.get("path"))
        fig_url = fig.get("url")
        if fig_path != "-" and fig_url:
            evidence = f"[`{fig_path}`]({fig_url})"
        elif fig_path != "-":
            evidence = f"`{fig_path}`"
        elif fig_url:
            evidence = f"[link]({fig_url})"
        else:
            evidence = "-"
        md_lines.append(f"| {idx} | {fig_title} | {evidence} | {_status_text(bool(fig_url))} |")
    md_lines.append("")
    for idx, fig in enumerate(figures, start=1):
        title = f"{fig['title_cn']} / {fig['title_en']}"
        url = fig.get("url")
        if url:
            md_lines.append(f"**图 {idx} / Figure {idx}. {title}**")
            md_lines.append("")
            md_lines.append(f"![图 {idx} / Figure {idx}. {title}]({url})")
            md_lines.append("")
        else:
            md_lines.append(f"- 图 {idx} / Figure {idx}. {title}: 缺失 / Missing.")
    md_lines.append("")

    md_lines.append("### 4.4 SHAP 特征重要性 / SHAP Feature Importance（Top 10）")
    md_lines.append("")
    if shap_rows:
        md_lines.append("| 排名 | 特征 | mean_abs_shap |")
        md_lines.append("| ---: | --- | ---: |")
        for idx, row in enumerate(shap_rows, start=1):
            md_lines.append(f"| {idx} | `{row['feature']}` | {_fmt_number(row['mean_abs_shap'], 6)} |")
    else:
        md_lines.append("未检测到 `shap_feature_importance.csv`，无法生成 SHAP 排名表。")
    md_lines.append("")

    md_lines.append("### 4.5 预测栅格统计 / Prediction Raster Summary")
    md_lines.append("")
    if pred_summary:
        md_lines.append("| 项目 | 数值 |")
        md_lines.append("| --- | --- |")
        md_lines.append(f"| 尺寸（宽 x 高） | {pred_summary.get('width', '-')} x {pred_summary.get('height', '-')} |")
        md_lines.append(f"| 有效像元 / 总像元 | {pred_summary.get('valid_pixels', '-')} / {pred_summary.get('total_pixels', '-')} |")
        md_lines.append(f"| 有效像元占比 | {_fmt_percent(pred_summary.get('valid_ratio'), 2)} |")
        md_lines.append(f"| 最小概率 | {_fmt_number(pred_summary.get('min_probability'), 6)} |")
        md_lines.append(f"| 最大概率 | {_fmt_number(pred_summary.get('max_probability'), 6)} |")
        md_lines.append(f"| 平均概率 | {_fmt_number(pred_summary.get('mean_probability'), 6)} |")
        md_lines.append(f"| CRS | `{pred_summary.get('crs') or '-'}` |")
    else:
        md_lines.append("未检测到预测栅格，无法生成栅格统计表。")
    md_lines.append("")

    md_lines.append("## 5. 结论摘要 / Executive Conclusion")
    md_lines.append("")
    md_lines.append("### 5.1 中文结论摘要")
    md_lines.append("")
    md_lines.extend([f"- {line}" for line in conclusion_cn])
    md_lines.append("")
    md_lines.append("### 5.2 English Executive Summary")
    md_lines.append("")
    md_lines.extend([f"- {line}" for line in conclusion_en])
    md_lines.append("")
    md_lines.append("### 5.3 讨论要点 / Discussion Highlights")
    md_lines.append("")
    md_lines.append(f"1. {perf_note_cn} / {perf_note_en}")
    md_lines.append(f"2. {pr_tradeoff_cn} / {pr_tradeoff_en}")
    md_lines.append(f"3. {generalization_cn} / {generalization_en}")
    md_lines.append(f"4. {raster_note_cn} / {raster_note_en}")
    md_lines.append("")
    md_lines.append("## 6. 可追溯清单 / Traceability Checklist")
    md_lines.append("")
    md_lines.append("| 编号 / No. | 清单项 / Checklist Item | 证据路径 / Evidence Path | 状态 / Status | 备注 / Notes |")
    md_lines.append("| ---: | --- | --- | --- | --- |")
    for idx, row in enumerate(trace_rows, start=1):
        path_text = _norm_path(row.get("path"))
        row_url = row.get("url")
        if path_text != "-" and row_url:
            evidence = f"[`{path_text}`]({row_url})"
        elif path_text != "-":
            evidence = f"`{path_text}`"
        elif row_url:
            evidence = f"[link]({row_url})"
        else:
            evidence = "-"
        md_lines.append(f"| {idx} | {row.get('item')} | {evidence} | {_status_text(bool(row.get('verified')))} | {row.get('notes')} |")
    md_lines.append("")
    md_lines.append("## 7. 复现与产物索引 / Reproducibility")
    md_lines.append("")
    md_lines.append(f"- 任务详情 / Job detail: [/jobs/{job_id}](/jobs/{job_id})")
    md_lines.append(f"- 任务产物列表 / Artifacts: [/jobs/{job_id}/artifacts](/jobs/{job_id}/artifacts)")
    md_lines.append(f"- 结果压缩包 / Result ZIP: [/jobs/{job_id}/result](/jobs/{job_id}/result)")
    if metrics_rel:
        md_lines.append(f"- 指标文件 / Metrics JSON: [{metrics_rel}]({_artifact_url(job_id, metrics_rel)})")
    if run_cfg_rel:
        md_lines.append(f"- 训练配置 / Run config: [{run_cfg_rel}]({_artifact_url(job_id, run_cfg_rel)})")
    if shap_csv_rel:
        md_lines.append(f"- SHAP 数据 / SHAP CSV: [{shap_csv_rel}]({_artifact_url(job_id, shap_csv_rel)})")
    md_lines.append("")

    markdown = "\n".join(md_lines).strip() + "\n"

    metric_html_rows = "\n".join(
        [
            "<tr>"
            f"<td>{html_escape(label_cn)} / {html_escape(label_en)} (<code>{html_escape(key)}</code>)</td>"
            f"<td style='text-align:right'>{_fmt_number(val_metrics.get(key), 4)}</td>"
            f"<td style='text-align:right'>{_fmt_number(test_metrics.get(key), 4)}</td>"
            "</tr>"
            for key, label_cn, label_en in metric_items
        ]
    )
    shap_html_rows = (
        "\n".join(
            [
                "<tr>"
                f"<td style='text-align:right'>{idx}</td>"
                f"<td><code>{html_escape(str(row['feature']))}</code></td>"
                f"<td style='text-align:right'>{_fmt_number(row['mean_abs_shap'], 6)}</td>"
                "</tr>"
                for idx, row in enumerate(shap_rows, start=1)
            ]
        )
        if shap_rows
        else "<tr><td colspan='3'>未检测到 SHAP 排名数据。</td></tr>"
    )
    fig_html_blocks = "\n".join(
        [
            (
                f"<figure><figcaption>图 {idx} / Figure {idx}. {html_escape(str(fig['title_cn']))} / {html_escape(str(fig['title_en']))}</figcaption>"
                f"<img src=\"{html_escape(str(fig['url']))}\" alt=\"{html_escape(str(fig['title_en']))}\" /></figure>"
            )
            if fig.get("url")
            else f"<p>图 {idx} / Figure {idx}. {html_escape(str(fig['title_cn']))} / {html_escape(str(fig['title_en']))}：未找到对应图件产物 / Missing figure artifact.</p>"
            for idx, fig in enumerate(figures, start=1)
        ]
    )
    figure_index_html_rows = "\n".join(
        [
            (
                "<tr>"
                f"<td style='text-align:right'>{idx}</td>"
                f"<td>{html_escape(str(fig['title_cn']))} / {html_escape(str(fig['title_en']))}</td>"
                f"<td>{html_escape(_norm_path(fig.get('path')))}</td>"
                f"<td>{html_escape(_status_text(bool(fig.get('url'))))}</td>"
                "</tr>"
            )
            for idx, fig in enumerate(figures, start=1)
        ]
    )
    traceability_html_rows = "\n".join(
        [
            (
                "<tr>"
                f"<td style='text-align:right'>{idx}</td>"
                f"<td>{html_escape(str(row.get('item') or '-'))}</td>"
                f"<td>{html_escape(_norm_path(row.get('path')))}</td>"
                f"<td>{html_escape(_status_text(bool(row.get('verified'))))}</td>"
                f"<td>{html_escape(str(row.get('notes') or '-'))}</td>"
                "</tr>"
            )
            for idx, row in enumerate(trace_rows, start=1)
        ]
    )
    abstract_cn_html = "\n".join([f"<li>{html_escape(line)}</li>" for line in abstract_cn])
    abstract_en_html = "\n".join([f"<li>{html_escape(line)}</li>" for line in abstract_en])
    conclusion_cn_html = "\n".join([f"<li>{html_escape(line)}</li>" for line in conclusion_cn])
    conclusion_en_html = "\n".join([f"<li>{html_escape(line)}</li>" for line in conclusion_en])

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>矿产预测标准实验报告 / Standard Report - {html_escape(job_id)}</title>
  <style>
    body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 24px; color: #1f2430; line-height: 1.6; }}
    h1, h2, h3 {{ color: #0f3554; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 16px; }}
    th, td {{ border: 1px solid #d5dde7; padding: 6px 8px; font-size: 14px; }}
    th {{ background: #eef4fb; text-align: left; }}
    code {{ background: #f5f7fa; padding: 1px 4px; border-radius: 4px; }}
    figure {{ margin: 14px 0 22px; }}
    figure img {{ width: min(1100px, 100%); border: 1px solid #d5dde7; border-radius: 8px; background: #fff; }}
    figure figcaption {{ font-weight: 600; margin-bottom: 6px; }}
    .meta {{ background: #f5f8fc; border: 1px solid #d5dde7; border-radius: 8px; padding: 10px 12px; }}
  </style>
</head>
<body>
  <h1>矿产预测标准实验报告 / Standard Experiment Report</h1>
  <div class="meta">
    <div>任务 ID / Job ID: <code>{html_escape(job_id)}</code></div>
    <div>生成时间 / Generated At: {html_escape(generated_at)}</div>
    <div>模型算法 / Algorithm: {html_escape(display_algo)} (<code>{html_escape(algorithm or "-")}</code>)</div>
    <div>模型名称 / Model Name: <code>{html_escape(model_name)}</code></div>
    <div>可追溯检查 / Traceability: {trace_verified}/{trace_total} verified</div>
  </div>

  <h2>摘要 / Abstract</h2>
  <h3>中文摘要</h3>
  <ul>{abstract_cn_html}</ul>
  <h3>English Abstract</h3>
  <ul>{abstract_en_html}</ul>

  <h2>1. 数据集介绍 / Dataset Overview</h2>
  <table>
    <tr><th>项目</th><th>内容</th></tr>
    <tr><td>样本 Shapefile</td><td><code>{html_escape(sample_shp)}</code></td></tr>
    <tr><td>研究区 Shapefile</td><td><code>{html_escape(research_shp)}</code></td></tr>
    <tr><td>因子目录数量</td><td>{len(feature_dirs)}</td></tr>
    <tr><td>因子栅格数量（.tif/.tiff）</td><td>{factor_count if factor_count > 0 else '-'}</td></tr>
    <tr><td>样本量（n_samples）</td><td>{html_escape(str(n_samples))}</td></tr>
    <tr><td>阳性样本占比</td><td>{_fmt_percent(pos_ratio, 2)}</td></tr>
    <tr><td>先验启用（Dome/Fault/Strata/Singularity）</td><td>{prior_dome} / {prior_fault} / {prior_strata} / {prior_singularity}</td></tr>
  </table>

  <h2>2. 方法介绍 / Method</h2>
  <p>本任务采用 <strong>{html_escape(display_algo)}</strong> 算法，优化方式为 <code>{html_escape(optimizer)}</code>。<br/>This run uses <strong>{html_escape(display_algo)}</strong> with optimizer mode <code>{html_escape(optimizer)}</code>.</p>
  <table>
    <tr><th>参数</th><th>数值</th></tr>
    <tr><td>epochs</td><td>{train_req.get('epochs', train_cfg.get('epochs', '-'))}</td></tr>
    <tr><td>patience</td><td>{train_req.get('patience', train_cfg.get('patience', '-'))}</td></tr>
    <tr><td>learning_rate</td><td>{train_req.get('lr', train_cfg.get('lr', '-'))}</td></tr>
    <tr><td>weight_decay</td><td>{train_req.get('weight_decay', train_cfg.get('weight_decay', '-'))}</td></tr>
    <tr><td>threshold</td><td>{train_req.get('threshold', train_cfg.get('threshold', '-'))}</td></tr>
    <tr><td>batch_size_cuda</td><td>{train_req.get('batch_size_cuda', train_cfg.get('batch_size_cuda', '-'))}</td></tr>
    <tr><td>batch_size_cpu</td><td>{train_req.get('batch_size_cpu', train_cfg.get('batch_size_cpu', '-'))}</td></tr>
    <tr><td>physics_weight</td><td>{train_req.get('physics_weight', train_cfg.get('physics_weight', '-'))}</td></tr>
  </table>

  <h2>3. 结果 / Results</h2>
  <h3>3.1 关键指标表 / Key Metrics</h3>
  <table>
    <tr><th>指标</th><th style="text-align:right">Validation</th><th style="text-align:right">Test</th></tr>
    {metric_html_rows}
  </table>

  <h3>3.2 图件索引 / Figure Index</h3>
  <table>
    <tr><th style="text-align:right">No.</th><th>图题 / Caption</th><th>证据路径 / Evidence</th><th>状态 / Status</th></tr>
    {figure_index_html_rows}
  </table>
  <h3>3.3 图件结果 / Figures</h3>
  {fig_html_blocks}

  <h3>3.4 SHAP 特征重要性（Top 10）/ SHAP Feature Importance</h3>
  <table>
    <tr><th style="text-align:right">排名</th><th>特征</th><th style="text-align:right">mean_abs_shap</th></tr>
    {shap_html_rows}
  </table>

  <h2>4. 结论摘要 / Executive Conclusion</h2>
  <h3>4.1 中文结论摘要</h3>
  <ol>{conclusion_cn_html}</ol>
  <h3>4.2 English Executive Summary</h3>
  <ol>{conclusion_en_html}</ol>
  <h3>4.3 讨论要点 / Discussion Highlights</h3>
  <ol>
    <li>{html_escape(perf_note_cn)} / {html_escape(perf_note_en)}</li>
    <li>{html_escape(pr_tradeoff_cn)} / {html_escape(pr_tradeoff_en)}</li>
    <li>{html_escape(generalization_cn)} / {html_escape(generalization_en)}</li>
    <li>{html_escape(raster_note_cn)} / {html_escape(raster_note_en)}</li>
  </ol>

  <h2>5. 可追溯清单 / Traceability Checklist</h2>
  <table>
    <tr><th style="text-align:right">No.</th><th>清单项 / Checklist Item</th><th>证据路径 / Evidence</th><th>状态 / Status</th><th>备注 / Notes</th></tr>
    {traceability_html_rows}
  </table>

  <h2>6. 复现入口 / Reproducibility</h2>
  <ul>
    <li>Job Detail: <code>/jobs/{html_escape(job_id)}</code></li>
    <li>Artifacts List: <code>/jobs/{html_escape(job_id)}/artifacts</code></li>
    <li>Result ZIP: <code>/jobs/{html_escape(job_id)}/result</code></li>
  </ul>
</body>
</html>
"""

    return markdown, html, {
        "metrics_path": metrics_rel,
        "run_config_path": run_cfg_rel,
        "full_request_path": full_req_rel,
        "shap_csv_path": shap_csv_rel,
        "prediction_tif_path": "output/predict/probability.tif" if pred_tif else None,
        "traceability_total": trace_total,
        "traceability_verified": trace_verified,
        "traceability_missing": trace_missing,
        "report_spec_version": "v2-bilingual-traceable",
    }


def _update_bounds_xy(bounds: list[float] | None, x: float, y: float) -> list[float]:
    if bounds is None:
        return [x, y, x, y]
    bounds[0] = min(bounds[0], x)
    bounds[1] = min(bounds[1], y)
    bounds[2] = max(bounds[2], x)
    bounds[3] = max(bounds[3], y)
    return bounds


def _update_bounds_from_coords(bounds: list[float] | None, coords: Any) -> list[float] | None:
    if coords is None:
        return bounds
    if isinstance(coords, (list, tuple)):
        if len(coords) >= 2 and all(isinstance(v, (int, float)) for v in coords[:2]):
            return _update_bounds_xy(bounds, float(coords[0]), float(coords[1]))
        out = bounds
        for item in coords:
            out = _update_bounds_from_coords(out, item)
        return out
    return bounds


def _parse_shp_record_geometry(shape_type: int, payload: bytes) -> dict[str, Any] | None:
    if shape_type == 0:
        return None

    if shape_type in {1, 11, 21}:  # Point/PointZ/PointM
        if len(payload) < 16:
            return None
        x, y = struct.unpack("<2d", payload[:16])
        return {"type": "Point", "coordinates": [x, y]}

    if shape_type in {8, 18, 28}:  # MultiPoint
        if len(payload) < 36:
            return None
        num_points = struct.unpack("<i", payload[32:36])[0]
        if num_points < 1:
            return None
        points_off = 36
        need = points_off + (num_points * 16)
        if len(payload) < need:
            return None
        pts: list[list[float]] = []
        for i in range(num_points):
            off = points_off + (i * 16)
            x, y = struct.unpack("<2d", payload[off : off + 16])
            pts.append([x, y])
        return {"type": "MultiPoint", "coordinates": pts}

    if shape_type in {3, 5, 13, 15, 23, 25}:  # PolyLine/Polygon (+Z/M)
        if len(payload) < 40:
            return None
        num_parts = struct.unpack("<i", payload[32:36])[0]
        num_points = struct.unpack("<i", payload[36:40])[0]
        if num_parts < 1 or num_points < 1:
            return None
        parts_off = 40
        parts_need = parts_off + (num_parts * 4)
        if len(payload) < parts_need:
            return None
        parts = list(struct.unpack("<" + ("i" * num_parts), payload[parts_off:parts_need]))
        points_off = parts_need
        points_need = points_off + (num_points * 16)
        if len(payload) < points_need:
            return None
        points: list[list[float]] = []
        for i in range(num_points):
            off = points_off + (i * 16)
            x, y = struct.unpack("<2d", payload[off : off + 16])
            points.append([x, y])

        parts.append(num_points)
        lines: list[list[list[float]]] = []
        for i in range(num_parts):
            start = parts[i]
            end = parts[i + 1]
            if 0 <= start < end <= num_points:
                seg = points[start:end]
                if len(seg) >= 2:
                    lines.append(seg)
        if not lines:
            return None

        if shape_type in {3, 13, 23}:  # PolyLine family
            if len(lines) == 1:
                return {"type": "LineString", "coordinates": lines[0]}
            return {"type": "MultiLineString", "coordinates": lines}

        # Polygon family: treat each part as one polygon ring (robust for display)
        polys: list[list[list[list[float]]]] = []
        for ring in lines:
            if ring[0] != ring[-1]:
                ring = [*ring, ring[0]]
            if len(ring) >= 4:
                polys.append([ring])
        if not polys:
            return None
        if len(polys) == 1:
            return {"type": "Polygon", "coordinates": polys[0]}
        return {"type": "MultiPolygon", "coordinates": polys}

    return None


def _read_shp_as_geojson(shp_path: Path, *, max_features: int) -> dict[str, Any]:
    if max_features < 1:
        raise ValueError("max_features must be >= 1")

    features: list[dict[str, Any]] = []
    bounds: list[float] | None = None
    total_records = 0
    clipped = False

    with shp_path.open("rb") as f:
        header = f.read(100)
        if len(header) < 100:
            raise ValueError("invalid shp header")

        while True:
            rec_header = f.read(8)
            if not rec_header:
                break
            if len(rec_header) < 8:
                break
            _, content_len_words = struct.unpack(">2i", rec_header)
            content_len = int(content_len_words) * 2
            rec = f.read(content_len)
            if len(rec) < 4:
                continue

            total_records += 1
            shape_type = struct.unpack("<i", rec[:4])[0]
            geom = _parse_shp_record_geometry(shape_type, rec[4:])
            if geom is None:
                continue

            bounds = _update_bounds_from_coords(bounds, geom.get("coordinates"))
            if len(features) >= max_features:
                clipped = True
                continue
            features.append(
                {
                    "type": "Feature",
                    "properties": {"fid": total_records},
                    "geometry": geom,
                }
            )

    out_bounds = None
    if bounds is not None:
        out_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]  # [[south, west], [north, east]]

    return {
        "type": "FeatureCollection",
        "features": features,
        "summary": {
            "records_total": total_records,
            "features_returned": len(features),
            "clipped": clipped,
            "bounds": out_bounds,
        },
    }


def _transform_geometry_to_wgs84(geometry: dict[str, Any], source_crs: CRS) -> dict[str, Any]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")

    def tx_xy(x: float, y: float) -> list[float]:
        lon, lat = rio_transform(source_crs, "EPSG:4326", [x], [y])
        return [float(lon[0]), float(lat[0])]

    if gtype == "Point":
        return {"type": "Point", "coordinates": tx_xy(coords[0], coords[1])}
    if gtype == "MultiPoint":
        return {"type": "MultiPoint", "coordinates": [tx_xy(p[0], p[1]) for p in coords]}
    if gtype == "LineString":
        return {"type": "LineString", "coordinates": [tx_xy(p[0], p[1]) for p in coords]}
    if gtype == "MultiLineString":
        return {
            "type": "MultiLineString",
            "coordinates": [[tx_xy(p[0], p[1]) for p in line] for line in coords],
        }
    if gtype == "Polygon":
        return {
            "type": "Polygon",
            "coordinates": [[tx_xy(p[0], p[1]) for p in ring] for ring in coords],
        }
    if gtype == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [[tx_xy(p[0], p[1]) for p in ring] for ring in poly]
                for poly in coords
            ],
        }
    return geometry


def _map_geometry_xy(geometry: dict[str, Any], mapper: Callable[[float, float], list[float]]) -> dict[str, Any]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")

    if gtype == "Point":
        return {"type": "Point", "coordinates": mapper(coords[0], coords[1])}
    if gtype == "MultiPoint":
        return {"type": "MultiPoint", "coordinates": [mapper(p[0], p[1]) for p in coords]}
    if gtype == "LineString":
        return {"type": "LineString", "coordinates": [mapper(p[0], p[1]) for p in coords]}
    if gtype == "MultiLineString":
        return {
            "type": "MultiLineString",
            "coordinates": [[mapper(p[0], p[1]) for p in line] for line in coords],
        }
    if gtype == "Polygon":
        return {
            "type": "Polygon",
            "coordinates": [[mapper(p[0], p[1]) for p in ring] for ring in coords],
        }
    if gtype == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [[mapper(p[0], p[1]) for p in ring] for ring in poly]
                for poly in coords
            ],
        }
    return geometry


def _flatten_geometry_coords(coords: Any) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []

    def walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and all(isinstance(v, (int, float)) for v in node[:2]):
                out.append((float(node[0]), float(node[1])))
                return
            for item in node:
                walk(item)

    walk(coords)
    return out


def _geometry_is_lonlat(geometry: dict[str, Any]) -> bool:
    coords = _flatten_geometry_coords(geometry.get("coordinates"))
    if not coords:
        return False
    for lon, lat in coords:
        if not (math.isfinite(lon) and math.isfinite(lat)):
            return False
        if lon < -180.000001 or lon > 180.000001:
            return False
        if lat < -90.000001 or lat > 90.000001:
            return False
    return True


def _extract_utm_zone_from_wkt(wkt: str) -> tuple[int, bool] | None:
    text = wkt or ""
    m = re.search(r"UTM[_\s]*Zone[_\s]*(\d{1,2})\s*([NS])", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"ZONE[_\s]*(\d{1,2})\s*([NS])", text, flags=re.IGNORECASE)
    if not m:
        return None
    zone = int(m.group(1))
    hemi = m.group(2).upper()
    if zone < 1 or zone > 60:
        return None
    return zone, (hemi == "S")


def _utm_xy_to_lonlat(x: float, y: float, zone: int, is_south: bool) -> list[float]:
    a = 6378137.0
    e = 0.08181919084262149
    e_sq = e * e
    e1_sq = e_sq / (1.0 - e_sq)
    k0 = 0.9996

    x = x - 500000.0
    if is_south:
        y = y - 10000000.0

    m = y / k0
    mu = m / (a * (1.0 - e_sq / 4.0 - 3.0 * (e_sq**2) / 64.0 - 5.0 * (e_sq**3) / 256.0))
    e1 = (1.0 - math.sqrt(1.0 - e_sq)) / (1.0 + math.sqrt(1.0 - e_sq))
    j1 = 3.0 * e1 / 2.0 - 27.0 * (e1**3) / 32.0
    j2 = 21.0 * (e1**2) / 16.0 - 55.0 * (e1**4) / 32.0
    j3 = 151.0 * (e1**3) / 96.0
    j4 = 1097.0 * (e1**4) / 512.0
    fp = mu + j1 * math.sin(2.0 * mu) + j2 * math.sin(4.0 * mu) + j3 * math.sin(6.0 * mu) + j4 * math.sin(8.0 * mu)

    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)
    c1 = e1_sq * (cos_fp**2)
    t1 = tan_fp**2
    n1 = a / math.sqrt(1.0 - e_sq * (sin_fp**2))
    r1 = a * (1.0 - e_sq) / ((1.0 - e_sq * (sin_fp**2)) ** 1.5)
    d = x / (n1 * k0)

    q1 = n1 * tan_fp / r1
    q2 = (d**2) / 2.0
    q3 = (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * (c1**2) - 9.0 * e1_sq) * (d**4) / 24.0
    q4 = (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * (t1**2) - 252.0 * e1_sq - 3.0 * (c1**2)) * (d**6) / 720.0
    lat = fp - q1 * (q2 - q3 + q4)

    q5 = d
    q6 = (1.0 + 2.0 * t1 + c1) * (d**3) / 6.0
    q7 = (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * (c1**2) + 8.0 * e1_sq + 24.0 * (t1**2)) * (d**5) / 120.0
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    lon = lon0 + (q5 - q6 + q7) / cos_fp

    return [float(math.degrees(lon)), float(math.degrees(lat))]


def _find_projection_sidecar(shp_path: Path) -> Path | None:
    candidates = [shp_path.with_suffix(".prj"), shp_path.with_suffix(".PRJ"), shp_path.with_suffix(".qpj"), shp_path.with_suffix(".QPJ")]
    for c in candidates:
        if c.is_file():
            return c
    stem = shp_path.stem
    for c in shp_path.parent.iterdir():
        if c.is_file() and c.stem == stem and c.suffix.lower() in {".prj", ".qpj"}:
            return c
    return None


def _bounds_from_feature_list(features: list[dict[str, Any]]) -> list[list[float]] | None:
    bounds: list[float] | None = None
    for feat in features:
        geom = feat.get("geometry") or {}
        bounds = _update_bounds_from_coords(bounds, geom.get("coordinates"))
    if bounds is None:
        return None
    return [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]


def _sample_raster_lonlat(tif_path: Path, lon: float, lat: float) -> dict[str, Any]:
    with rasterio.open(tif_path) as src:
        if src.crs:
            xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
            x = float(xs[0])
            y = float(ys[0])
        else:
            x = float(lon)
            y = float(lat)

        row, col = src.index(x, y)
        inside = 0 <= row < src.height and 0 <= col < src.width
        if not inside:
            return {
                "inside": False,
                "value": None,
                "row": int(row),
                "col": int(col),
                "x": x,
                "y": y,
                "crs": str(src.crs) if src.crs else None,
            }

        values = list(src.sample([(x, y)], indexes=1))
        value = float(values[0][0]) if values else None
        if src.nodata is not None and value is not None and math.isfinite(value):
            if math.isclose(value, float(src.nodata), rel_tol=0.0, abs_tol=1e-10):
                value = None
        if value is not None and not math.isfinite(value):
            value = None

        return {
            "inside": True,
            "value": value,
            "row": int(row),
            "col": int(col),
            "x": x,
            "y": y,
            "crs": str(src.crs) if src.crs else None,
        }


def _package_job(job_id: str) -> Path:
    with _LOCK:
        job = _JOBS[job_id]
        job_dir = Path(job["job_dir"])
        logs = list(job.get("logs", []))
    result_zip = job_dir / "result.zip"
    if result_zip.exists():
        result_zip.unlink()
    logs_path = job_dir / "logs.txt"
    logs_path.write_text("\n".join(logs), encoding="utf-8")

    include_roots = [job_dir / "output", job_dir / "config", logs_path]
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root in include_roots:
            if not root.exists():
                continue
            if root.is_file():
                zf.write(root, arcname=str(root.relative_to(job_dir)))
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(job_dir)))
    return result_zip


def _validate_scripts() -> None:
    if not TRAIN_SCRIPT.is_file():
        raise HTTPException(500, f"Train script not found: {TRAIN_SCRIPT}")
    if not PREDICT_SCRIPT.is_file():
        raise HTTPException(500, f"Predict script not found: {PREDICT_SCRIPT}")


def _run_unified_job(job_id: str, req: FullJobRequest, train_output_root: Path, predict_output_tif: Path, job_dir: Path) -> None:
    try:
        _set_status(job_id, "running")
        _write_json(job_dir / "config" / "full_request.json", req.model_dump())

        train_cmd = _build_train_command(req.train, train_output_root)
        rc_train = _run_command(job_id, train_cmd, cwd=PROJECT_ROOT)
        if rc_train != 0:
            raise RuntimeError(f"Training command failed with code {rc_train}")

        model_path = train_output_root / "models" / f"{req.train.model_name}{_model_artifact_suffix(req.train.algorithm)}"
        preproc_path = train_output_root / "models" / "preproc.joblib"
        predict_req = PredictJobRequest(
            feature_dir=req.predict_feature_dir,
            research_shp=req.predict_research_shp,
            model_path=str(model_path),
            preproc_path=str(preproc_path),
            output_tif=str(predict_output_tif),
            batch_size=req.predict_batch_size,
        )
        predict_cmd = _build_predict_command(predict_req, output_tif=predict_output_tif)
        rc_predict = _run_command(job_id, predict_cmd, cwd=PROJECT_ROOT)
        if rc_predict != 0:
            raise RuntimeError(f"Predict command failed with code {rc_predict}")

        _set_status(job_id, "packaging")
        result_zip = _package_job(job_id)
        artifacts = _collect_artifacts(job_dir)
        with _LOCK:
            job = _JOBS[job_id]
            job["status"] = "done"
            job["finished_at"] = _now_iso()
            job["result_zip"] = result_zip
            job["artifacts"] = artifacts
    except Exception as exc:
        _set_error(job_id, repr(exc))
        _set_status(job_id, "failed")
        with _LOCK:
            _JOBS[job_id]["finished_at"] = _now_iso()


def _init_job(job_type: str, meta: dict[str, Any] | None = None) -> tuple[str, Path]:
    job_id = str(uuid4())
    job_dir = RUNS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input").mkdir(parents=True, exist_ok=True)
    (job_dir / "output").mkdir(parents=True, exist_ok=True)
    (job_dir / "config").mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})

    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "type": job_type,
            "status": "queued",
            "created_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "logs": [],
            "error": None,
            "result_zip": None,
            "job_dir": job_dir,
            "artifacts_root": job_dir,
            "artifacts": [],
            "algorithm": meta.get("algorithm"),
            "optimizer": meta.get("optimizer"),
            "model_name": meta.get("model_name"),
            "display_name": meta.get("display_name"),
        }
    return job_id, job_dir


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_load_index_html())


@app.get("/jobs")
def list_jobs() -> dict[str, Any]:
    with _LOCK:
        jobs = [_serialize_job(v) for v in _JOBS.values()]
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    return {"count": len(jobs), "jobs": jobs}


@app.post("/jobs/upload/unified", response_model=JobResponse)
@app.post("/jobs/upload/full", response_model=JobResponse)
def create_unified_job_upload(
    factors_zip: UploadFile = File(...),
    samples_zip: UploadFile = File(...),
    research_zip: UploadFile = File(...),
    priors_zip: UploadFile = File(...),
    use_dome: bool = Form(True),
    use_fault: bool = Form(True),
    use_strata: bool = Form(True),
    use_singularity: bool = Form(False),
    dome_pattern: str = Form("dome"),
    fault_pattern: str = Form("fault,thrust"),
    strata_pattern: str = Form("strata,fav"),
    singularity_pattern: str = Form("singularity,fractal"),
    label_col: str = Form("Class"),
    output_root: str | None = Form(None),
    model_name: str = Form("mineral_model_multi_phy_final"),
    algorithm: Literal[
        "ginn", "pinn", "rf", "lightgbm", "catboost", "xgboost", "mlp", "gnn", "vae", "transunet", "gcn_transformer"
    ] = Form("ginn"),
    optimizer: Literal["none", "bayes"] = Form("none"),
    bayes_trials: int = Form(25),
    seed: int = Form(2025),
    test_size: float = Form(0.2),
    val_size: float = Form(0.2),
    epochs: int = Form(800),
    patience: int = Form(60),
    lr: float = Form(1e-3),
    weight_decay: float = Form(1e-4),
    physics_weight: float = Form(1.0),
    grad_clip_norm: float = Form(2.0),
    batch_size_cuda: int = Form(4096),
    batch_size_cpu: int = Form(512),
    num_workers: int = Form(0),
    threshold: float = Form(0.5),
    disable_shap: bool = Form(False),
    shap_background_size: int = Form(64),
    shap_eval_size: int = Form(256),
    shap_nsamples: int = Form(200),
    predict_output_tif: str | None = Form(None),
    predict_batch_size: int = Form(262144),
) -> JobResponse:
    _validate_scripts()
    job_id, job_dir = _init_job(
        "upload-unified",
        meta={
            "algorithm": algorithm,
            "optimizer": optimizer,
            "model_name": model_name,
            "display_name": _make_job_display_name("upload-unified", algorithm, model_name),
        },
    )
    input_root = job_dir / "input"
    upload_root = input_root / "uploads"
    extract_root = input_root / "extracted"
    upload_root.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    factors_zip_path = upload_root / "factors.zip"
    samples_zip_path = upload_root / "samples.zip"
    research_zip_path = upload_root / "research.zip"
    priors_zip_path = upload_root / "priors.zip"
    _save_upload(factors_zip, factors_zip_path)
    _save_upload(samples_zip, samples_zip_path)
    _save_upload(research_zip, research_zip_path)
    _save_upload(priors_zip, priors_zip_path)

    try:
        factors_dir, samples_shp, dome_tif, fault_tif, strata_tif, singularity_tif = _prepare_upload_train_inputs(
            input_root=extract_root,
            factors_zip_path=factors_zip_path,
            samples_zip_path=samples_zip_path,
            priors_zip_path=priors_zip_path,
            dome_pattern=dome_pattern,
            fault_pattern=fault_pattern,
            strata_pattern=strata_pattern,
            singularity_pattern=singularity_pattern,
            use_dome=use_dome,
            use_fault=use_fault,
            use_strata=use_strata,
            use_singularity=use_singularity,
            require_priors=(algorithm in {"ginn", "pinn"}),
        )
        research_dir = extract_root / "research"
        _safe_extract(research_zip_path, research_dir)
        research_shp = _find_single_shp(research_dir)
    except Exception as exc:
        raise HTTPException(400, str(exc))

    train_req = TrainJobRequest(
        shapefile=str(samples_shp),
        feature_dirs=[str(factors_dir)],
        dome_tif=str(dome_tif) if dome_tif else None,
        fault_tif=str(fault_tif) if fault_tif else None,
        strata_tif=str(strata_tif) if strata_tif else None,
        singularity_tif=str(singularity_tif) if singularity_tif else None,
        label_col=label_col,
        output_root=output_root,
        model_name=model_name,
        algorithm=algorithm,
        optimizer=optimizer,
        bayes_trials=bayes_trials,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        physics_weight=physics_weight,
        grad_clip_norm=grad_clip_norm,
        batch_size_cuda=batch_size_cuda,
        batch_size_cpu=batch_size_cpu,
        num_workers=num_workers,
        threshold=threshold,
        disable_shap=disable_shap,
        shap_background_size=shap_background_size,
        shap_eval_size=shap_eval_size,
        shap_nsamples=shap_nsamples,
    )
    _validate_train_priors(train_req)
    req = FullJobRequest(
        train=train_req,
        predict_feature_dir=str(factors_dir),
        predict_research_shp=str(research_shp),
        predict_output_tif=predict_output_tif,
        predict_batch_size=predict_batch_size,
    )
    train_output_root = _normalize_output_root(req.train.output_root, job_dir / "output" / "train")
    train_output_root.mkdir(parents=True, exist_ok=True)
    predict_output_tif_path = (
        Path(req.predict_output_tif).expanduser().resolve()
        if req.predict_output_tif
        else job_dir / "output" / "predict" / "probability.tif"
    )
    predict_output_tif_path.parent.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        _JOBS[job_id]["started_at"] = _now_iso()
        _JOBS[job_id]["output_root"] = str(train_output_root)
        _JOBS[job_id]["output_tif"] = str(predict_output_tif_path)

    t = threading.Thread(
        target=_run_unified_job,
        args=(job_id, req, train_output_root, predict_output_tif_path, job_dir),
        daemon=True,
    )
    t.start()
    return JobResponse(job_id=job_id)


@app.post("/jobs/unified", response_model=JobResponse)
@app.post("/jobs/full", response_model=JobResponse)
def create_unified_job(req: FullJobRequest) -> JobResponse:
    _validate_scripts()
    _validate_train_priors(req.train)
    job_id, job_dir = _init_job(
        "unified",
        meta={
            "algorithm": req.train.algorithm,
            "optimizer": req.train.optimizer,
            "model_name": req.train.model_name,
            "display_name": _make_job_display_name("unified", req.train.algorithm, req.train.model_name),
        },
    )
    train_output_root = _normalize_output_root(req.train.output_root, job_dir / "output" / "train")
    train_output_root.mkdir(parents=True, exist_ok=True)
    predict_output_tif = (
        Path(req.predict_output_tif).expanduser().resolve()
        if req.predict_output_tif
        else job_dir / "output" / "predict" / "probability.tif"
    )
    predict_output_tif.parent.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        _JOBS[job_id]["started_at"] = _now_iso()
        _JOBS[job_id]["output_root"] = str(train_output_root)
        _JOBS[job_id]["output_tif"] = str(predict_output_tif)

    t = threading.Thread(
        target=_run_unified_job,
        args=(job_id, req, train_output_root, predict_output_tif, job_dir),
        daemon=True,
    )
    t.start()
    return JobResponse(job_id=job_id)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return _serialize_job(job)


@app.post("/jobs/{job_id}/report/generate")
def generate_job_report(job_id: str) -> dict[str, Any]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        job_status = str(job.get("status") or "")
        job_dir = Path(job["job_dir"]).resolve()
        job_copy = dict(job)

    if job_status != "done":
        raise HTTPException(400, f"job status is {job_status}, report can be generated only after completion")

    artifacts = _collect_artifacts(job_dir)
    markdown, html, source_paths = _build_report_documents(job_id, job_copy, job_dir, artifacts)

    report_dir = job_dir / "output" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_md = report_dir / "auto_report.md"
    report_html = report_dir / "auto_report.html"
    report_md.write_text(markdown, encoding="utf-8")
    report_html.write_text(html, encoding="utf-8")

    updated_artifacts = _collect_artifacts(job_dir)
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["artifacts"] = updated_artifacts

    result_zip_path: str | None = None
    try:
        result_zip = _package_job(job_id)
        result_zip_path = str(result_zip)
        with _LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["result_zip"] = result_zip
    except Exception as exc:
        _append_log(job_id, f"[report] refresh result.zip failed: {exc!r}")

    _append_log(job_id, f"[report] generated: {report_md.relative_to(job_dir).as_posix()}")

    md_rel = report_md.relative_to(job_dir).as_posix()
    html_rel = report_html.relative_to(job_dir).as_posix()
    return {
        "job_id": job_id,
        "report_md_path": md_rel,
        "report_md_url": _artifact_url(job_id, md_rel),
        "report_html_path": html_rel,
        "report_html_url": _artifact_url(job_id, html_rel),
        "source_paths": source_paths,
        "result_zip": result_zip_path,
    }


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str) -> FileResponse:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        if job["status"] != "done":
            raise HTTPException(400, f"job status is {job['status']}")
        result_zip = Path(job["result_zip"]) if job.get("result_zip") else None

    if not result_zip or not result_zip.is_file():
        raise HTTPException(404, "result zip not found")
    return FileResponse(str(result_zip), filename=f"{job_id}.zip")


@app.get("/jobs/{job_id}/prediction/summary")
def get_prediction_summary(job_id: str) -> dict[str, Any]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        tif_path = _resolve_prediction_tif(job)

    if not tif_path:
        raise HTTPException(404, "prediction raster not found")
    return {
        "job_id": job_id,
        "summary": _prediction_summary(tif_path),
        "preview_url": f"/jobs/{job_id}/prediction/preview.png",
    }


@app.get("/jobs/{job_id}/prediction/preview.png")
def get_prediction_preview(
    job_id: str,
    max_size: int = 900,
    colormap: str = "turbo",
) -> Response:
    if max_size < 128 or max_size > 4096:
        raise HTTPException(400, "max_size must be in [128, 4096]")
    if colormap not in _VALID_COLORMAPS:
        raise HTTPException(400, f"unsupported colormap: {colormap}")

    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        tif_path = _resolve_prediction_tif(job)

    if not tif_path:
        raise HTTPException(404, "prediction raster not found")
    png = _render_prediction_preview(tif_path, max_size=max_size, colormap=colormap)
    return Response(content=png, media_type="image/png")


@app.get("/jobs/{job_id}/artifact-raster/summary")
def get_artifact_raster_summary(job_id: str, artifact_path: str) -> dict[str, Any]:
    _, target = _resolve_job_artifact_path(job_id, artifact_path)
    if target.suffix.lower() not in {".tif", ".tiff"}:
        raise HTTPException(400, "artifact must be a tif/tiff raster")
    encoded = quote(artifact_path, safe="")
    return {
        "job_id": job_id,
        "artifact_path": artifact_path,
        "summary": _prediction_summary(target),
        "preview_url": f"/jobs/{job_id}/artifact-raster/preview.png?artifact_path={encoded}",
    }


@app.get("/jobs/{job_id}/artifact-raster/preview.png")
def get_artifact_raster_preview(
    job_id: str,
    artifact_path: str,
    max_size: int = 900,
    colormap: str = "viridis",
) -> Response:
    if max_size < 128 or max_size > 4096:
        raise HTTPException(400, "max_size must be in [128, 4096]")
    if colormap not in _VALID_COLORMAPS:
        raise HTTPException(400, f"unsupported colormap: {colormap}")

    _, target = _resolve_job_artifact_path(job_id, artifact_path)
    if target.suffix.lower() not in {".tif", ".tiff"}:
        raise HTTPException(400, "artifact must be a tif/tiff raster")

    png = _render_prediction_preview(target, max_size=max_size, colormap=colormap)
    return Response(content=png, media_type="image/png")


@app.get("/jobs/{job_id}/artifact-vector/geojson")
def get_artifact_vector_geojson(
    job_id: str,
    artifact_path: str,
    max_features: int = 10000,
) -> dict[str, Any]:
    if max_features < 1 or max_features > 200000:
        raise HTTPException(400, "max_features must be in [1, 200000]")

    _, target = _resolve_job_artifact_path(job_id, artifact_path)
    if target.suffix.lower() != ".shp":
        raise HTTPException(400, "artifact must be a .shp vector file")

    try:
        payload = _read_shp_as_geojson(target, max_features=max_features)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"failed to parse shapefile: {exc!r}")

    features = payload["features"]
    sidecar = _find_projection_sidecar(target)
    if sidecar is not None:
        try:
            wkt = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
            if wkt:
                source_crs = CRS.from_wkt(wkt)
                transformed: list[dict[str, Any]] = []
                if source_crs and source_crs.to_epsg() != 4326:
                    ok_count = 0
                    transformed: list[dict[str, Any]] = []
                    for feat in features:
                        geom = feat.get("geometry")
                        if not geom:
                            transformed.append(feat)
                            continue
                        g2 = _transform_geometry_to_wgs84(geom, source_crs)
                        if _geometry_is_lonlat(g2):
                            ok_count += 1
                        transformed.append(
                            {
                                "type": "Feature",
                                "properties": dict(feat.get("properties", {})),
                                "geometry": g2,
                            }
                        )
                    if ok_count > 0:
                        features = transformed

                # Fallback for UTM shapefiles where CRS transform is unavailable/failed.
                if features and not _geometry_is_lonlat(features[0].get("geometry", {})):
                    utm = _extract_utm_zone_from_wkt(wkt)
                    if utm is not None:
                        zone, is_south = utm
                        transformed2: list[dict[str, Any]] = []
                        ok2 = 0
                        for feat in features:
                            geom = feat.get("geometry")
                            if not geom:
                                transformed2.append(feat)
                                continue
                            g3 = _map_geometry_xy(geom, lambda x, y: _utm_xy_to_lonlat(x, y, zone, is_south))
                            if _geometry_is_lonlat(g3):
                                ok2 += 1
                            transformed2.append(
                                {
                                    "type": "Feature",
                                    "properties": dict(feat.get("properties", {})),
                                    "geometry": g3,
                                }
                            )
                        if ok2 > 0:
                            features = transformed2
        except Exception:
            # Keep raw coordinates when projection parsing fails.
            pass

    bounds = _bounds_from_feature_list(features)
    summary = dict(payload.get("summary", {}))
    summary["bounds"] = bounds

    return {
        "job_id": job_id,
        "artifact_path": artifact_path,
        "geojson": {
            "type": payload["type"],
            "features": features,
        },
        "summary": summary,
    }


@app.get("/jobs/{job_id}/prediction/sample")
def get_prediction_sample(job_id: str, lon: float, lat: float) -> dict[str, Any]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        tif_path = _resolve_prediction_tif(job)
    if not tif_path:
        raise HTTPException(404, "prediction raster not found")
    sample = _sample_raster_lonlat(tif_path, lon=lon, lat=lat)
    return {
        "job_id": job_id,
        "layer": "prediction_result",
        "lon": float(lon),
        "lat": float(lat),
        "sample": sample,
    }


@app.get("/jobs/{job_id}/artifact-raster/sample")
def get_artifact_raster_sample(job_id: str, artifact_path: str, lon: float, lat: float) -> dict[str, Any]:
    _, target = _resolve_job_artifact_path(job_id, artifact_path)
    if target.suffix.lower() not in {".tif", ".tiff"}:
        raise HTTPException(400, "artifact must be a tif/tiff raster")
    sample = _sample_raster_lonlat(target, lon=lon, lat=lat)
    return {
        "job_id": job_id,
        "artifact_path": artifact_path,
        "lon": float(lon),
        "lat": float(lat),
        "sample": sample,
    }


@app.get("/jobs/{job_id}/artifacts")
def get_artifacts(job_id: str) -> dict[str, Any]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        job_dir = Path(job["job_dir"]).resolve()
        cached = list(job.get("artifacts", []))

    artifacts = cached if cached else _collect_artifacts(job_dir)
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["artifacts"] = artifacts
    return {"job_id": job_id, "artifacts": artifacts}


@app.get("/jobs/{job_id}/artifacts/{artifact_path:path}")
def get_artifact_file(job_id: str, artifact_path: str) -> FileResponse:
    _, target = _resolve_job_artifact_path(job_id, artifact_path)
    return FileResponse(str(target), filename=target.name)
