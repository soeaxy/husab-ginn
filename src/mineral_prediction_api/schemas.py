from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TrainJobRequest(BaseModel):
    shapefile: str = Field(..., description="Training sample shapefile path.")
    feature_dirs: list[str] = Field(
        ..., min_length=1, description="Feature raster directories."
    )
    dome_tif: str | None = Field(None, description="Distance raster for dome prior.")
    fault_tif: str | None = Field(None, description="Distance raster for fault prior.")
    strata_tif: str | None = Field(
        None, description="Distance raster for strata prior."
    )
    singularity_tif: str | None = Field(
        None,
        description="Legacy optional singularity prior; inactive in the ORGEO-D-26-00764 experiments.",
    )
    label_col: str = Field("Class", description="Label column name.")

    output_root: str | None = Field(
        None, description="Output root directory. Defaults to per-job directory."
    )
    model_name: str = Field(
        "mineral_model_multi_phy_final", description="Output model basename."
    )
    algorithm: Literal[
        "ginn",
        "pinn",
        "rf",
        "lightgbm",
        "catboost",
        "xgboost",
        "mlp",
        "gnn",
        "vae",
        "transunet",
        "gcn_transformer",
    ] = Field(
        "ginn",
        description="Training algorithm; 'pinn' is retained as a legacy alias for manuscript GINN.",
    )
    optimizer: Literal["none", "bayes"] = Field(
        "none",
        description="Hyperparameter optimization strategy.",
    )
    bayes_trials: int = Field(
        25, ge=1, le=500, description="Bayesian optimization trial count."
    )

    seed: int = 2025
    test_size: float = 0.2
    val_size: float = 0.2
    epochs: int = 800
    patience: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    physics_weight: float = Field(
        1.0,
        description="Geological-prior loss weight; field name retained for backward compatibility.",
    )
    grad_clip_norm: float = 2.0
    batch_size_cuda: int = 4096
    batch_size_cpu: int = 512
    num_workers: int = 0
    threshold: float = 0.5

    disable_shap: bool = False
    shap_background_size: int = 64
    shap_eval_size: int = 256
    shap_nsamples: int = 200


class PredictJobRequest(BaseModel):
    feature_dir: str = Field(..., description="Feature raster directory.")
    research_shp: str = Field(..., description="Research area shapefile path.")
    model_path: str = Field(..., description="Trained model path (.pth).")
    preproc_path: str = Field(
        ..., description="Preprocessor file path (preproc.joblib)."
    )
    output_tif: str | None = Field(None, description="Output probability TIFF path.")
    batch_size: int = 262_144


class FullJobRequest(BaseModel):
    train: TrainJobRequest
    predict_feature_dir: str = Field(
        ..., description="Feature raster directory for prediction."
    )
    predict_research_shp: str = Field(
        ..., description="Research area shapefile path for prediction."
    )
    predict_output_tif: str | None = Field(
        None, description="Prediction output tif. Defaults to per-job directory."
    )
    predict_batch_size: int = 262_144


class JobResponse(BaseModel):
    job_id: str
