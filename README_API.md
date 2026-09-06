# Mineral Prediction Unified API

This service exposes a unified workflow only: training and prediction in one job.

For exact reproduction of the manuscript's fixed spatial split and repeated
benchmark, use the command-line runners documented in `README.md`. The API is an
operational wrapper and does not replace the archived experiment manifest.

## 1. Install

```powershell
cd husab-ginn
python -m pip install -e .
```

## 2. Start

```powershell
uvicorn mineral_prediction_api.api:app --host 127.0.0.1 --port 8000
```

Open:

- `http://localhost:8000/` (Unified dashboard UI)
- `http://localhost:8000/docs` (OpenAPI docs)

## 3. Unified Submit Endpoints

The dashboard uses:

- `POST /jobs/upload/unified`

Compatibility aliases kept:

- `POST /jobs/upload/full` (same as unified upload endpoint)
- `POST /jobs/unified` (JSON request)
- `POST /jobs/full` (same as unified JSON endpoint)

Removed endpoints:

- `POST /jobs/upload/train`
- `POST /jobs/upload/predict`
- `POST /jobs/train`
- `POST /jobs/predict`

## 4. Upload Fields (`POST /jobs/upload/unified`)

Required files:

These files must be supplied from authorized local sources. No research data or
sample upload archives are included in <https://github.com/soeaxy/husab-ginn>.

- `factors_zip`: feature rasters zip
- `samples_zip`: training samples shapefile zip
- `research_zip`: prediction area shapefile zip
- `priors_zip`: prior rasters zip

Prior selection for GINN (at least one must be true):

- `use_dome`: `true|false`
- `use_fault`: `true|false`
- `use_strata`: `true|false`
- `use_singularity`: `true|false` (legacy optional input; not used in the manuscript experiments)

Note:

- For `ginn` (or legacy alias `pinn`), at least one prior is required and the geological-prior loss is enabled during training.
- The manuscript configuration uses dome, fault and strata only. MLP, GCN-Transformer and the tree baselines do not use the geological-prior loss.

Optional prior matching patterns:

- `dome_pattern` (default: `dome`)
- `fault_pattern` (default: `fault,thrust`)
- `strata_pattern` (default: `strata,fav`)
- `singularity_pattern` (default: `singularity,fractal`)

Other key options:

- `label_col`
- `epochs`, `patience`
- `model_name`
- `algorithm`: `ginn|pinn|mlp|gcn_transformer|rf|lightgbm|catboost|xgboost|gnn|vae|transunet`; `pinn` is retained only as a compatibility alias for GINN
- `optimizer`: `none|bayes`
- `bayes_trials`
- `disable_shap`
- `predict_batch_size`
- `predict_output_tif`

## 5. Job Query and Downloads

- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`
- `GET /jobs/{job_id}/artifacts`
- `GET /jobs/{job_id}/artifacts/{artifact_path}`
- `GET /jobs/{job_id}/prediction/summary`
- `GET /jobs/{job_id}/prediction/preview.png`
- `POST /jobs/{job_id}/report/generate` (one-click bilingual standard report generation with traceability checklist)

`prediction/preview.png` query params:

- `max_size`: preview longest edge, range `[128, 4096]`, default `900`
- `colormap`: `turbo|viridis|plasma|magma|inferno|cividis`, default `turbo`

## 6. Output Layout

Default run folder: `runs/{job_id}`

- `input/`: uploaded and extracted inputs
- `output/`: training and prediction outputs
- `config/`: request snapshots
- `logs.txt`: merged runtime log
- `result.zip`: packaged result

Typical visualization artifacts:

- `output/train/figures/roc_curve_test.png`
- `output/train/figures/pr_curve_test.png`
- `output/train/explainability/shap_summary.png`
- `output/train/metrics/metrics.json`
- `output/report/auto_report.md` (Chinese and English, numbered figures, executive summary, traceability checklist)
- `output/report/auto_report.html` (Chinese and English, numbered figures, executive summary, traceability checklist)

## 7. Optional Environment Variables

- `MINERAL_RUNS_DIR`
- `MINERAL_TRAIN_SCRIPT`
- `MINERAL_PREDICT_SCRIPT`

Legacy aliases still supported:

- `PINN_RUNS_DIR`
- `PINN_TRAIN_SCRIPT`
- `PINN_PREDICT_SCRIPT`
