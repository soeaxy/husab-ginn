# Husab GINN spatial benchmark

Code repository: <https://github.com/soeaxy/husab-ginn>.

Research code for the manuscript **“Geology-informed neural networks for
uranium prospectivity mapping in the Rössing–Husab district, Namibia: a
spatially blocked benchmark”** (Ore Geology Reviews manuscript
`ORGEO-D-26-00764`, major revision).

The model is called a **geology-informed neural network (GINN)** in the
manuscript. Historical scripts, result folders and checkpoints use the internal
algorithm key `pinn`; that key is retained for backward compatibility and must
not be interpreted as a differential-equation physics-informed neural network.
New command-line and API requests may use `--algorithm ginn`.

## What this repository implements

- portable reconstruction of the 20 m positive-candidate grid from pre-delineated
  target polygons and deterministic 10% positive draw;
- ten spatially balanced pseudo-absence realizations;
- one fixed 5 × 5 spatial-block benchmark shared by seven models;
- GINN, an architecture-matched MLP, an inductive local GCN–Transformer, Random
  Forest, LightGBM, XGBoost and CatBoost;
- controlled prior, weight, zero-loss, kernel-scale, initialization,
  spatial-grid and training-background experiments;
- metric recomputation, paired statistics, ensemble prediction and the
  coordinate-suppressed nine-panel evidence-layer figure;
- a FastAPI wrapper for training and prediction.

The main scientific result is not a general superiority claim for GINN. Under
the archived protocol, Random Forest had the highest mean AP, CatBoost the
highest ROC-AUC and GCN–Transformer the strongest top-5% sampled-cell capture.
Removing or freezing parts of the tested geological regularizer often improved
GINN AP. See [`docs/manuscript_alignment.md`](docs/manuscript_alignment.md) for
the claim-to-code crosswalk and interpretation limits.

## Manuscript protocol

| Item | Archived setting |
| --- | --- |
| Positive candidates | 20 m × 20 m grid; 41,980 cells within target polygons |
| Positive draw | 4,198 cells (`floor(41980 / 10)`), no replacement, `random_state=1` |
| Pseudo-absences | 41,980 per realization; 10 deterministic realizations |
| Negative strata | 70% hard, 5% transition, 25% background |
| Exclusion/thinning | ≥1000 m from mapped mineralized polygons; one point per 80 m cell |
| Primary split | private fixed 5 × 5 block manifest; model/split seed 2026 |
| Primary comparison | 7 models × 10 pseudo-absence realizations = 70 runs |
| Primary metric | validation-selected AP (`pr_auc` in archived JSON) |
| Supplementary work | 76 additional real-data runs, with repeat axes kept separate |

Dataset-specific counts are verified facts, not generic assertions enforced on
other study areas. The original delineation criteria and positional accuracy of
the target and prior-source geometries are not documented. Positive
labels here are polygon-membership points, not independent assay-grade ore
evidence.

Pre-delineated target polygons are used as a label source. Within the Husab
protocol, `Z1` and `Z2` are operating known targets; other polygon names are
retained only as provided and are not asserted here as confirmed independent
deposits.

## Input scaling and preprocessing

Inputs were normalized to `[0,1]` as part of the provided pre-model raster
preparation for this study. The training stage then applies train-only
preprocessing (`SimpleImputer` + variance filter + `StandardScaler`) and reuses
fitted transforms for held-out sets.

## Repository layout

```text
.
|-- manuscript_protocol.py                # machine-readable paper/code contract
|-- rebuild_positive_samples.py           # 20 m positive-label reconstruction
|-- rebuild_negative_samples.py           # ten pseudo-absence realizations
|-- train_multi_physics_model.py           # GINN and all comparison trainers
|-- mineral_deep_models.py                 # MLP and GCN–Transformer definitions
|-- run_reconstructed_benchmarks.py        # 7 × 10 primary benchmark
|-- run_ogr_revision_experiments.py        # prior/scale/seed/grid experiments
|-- run_ogr_sampling_design.py             # train-only background probes
|-- analyze_reconstructed_benchmarks.py    # paired statistical analysis
|-- analyze_evidence_guided_ensemble.py    # exploratory ensemble; not a primary paper result
|-- predict_reconstructed_pinn_ensemble.py # GINN ensemble; legacy filename
|-- plot_manuscript_evidence_layers.py     # coordinate-free manuscript Figure 2
|-- src/mineral_prediction_api/            # optional API
|-- docs/
`-- tests/
```

## Environment

The archived experiments used Python 3.11.14. Python 3.10 or later is
supported. Install the locked environment with `uv`:

```powershell
uv sync --python 3.11
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS, activate with `source .venv/bin/activate`. Alternatively, prefix
the Python and Uvicorn commands below with `uv run` without activating a shell.

Or install from the declared direct dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the research scripts from a repository checkout. The built wheel packages
the API service; it is not a standalone bundle of the root-level experiment
runners.

## Data boundary

This is an authorized code-only publication with a new Git history. The research
data have not been authorized for public release and are not bundled:

- nine aeromagnetic/radiometric factor rasters;
- dome, fault and stratigraphic distance rasters;
- mapped deposit/structural/lithological polygons;
- labeled sample points, drilling records and point-level predictions.
- exact study bounds, field observations, site coordinates and trained models.

Place authorized local copies under `data/factors`, `data/priors` and another
local data directory, or pass explicit paths. Generated `experiments/`,
`analysis_output/`, trained models and publication figures are ignored by Git.
Do not upload controlled inputs or generated outputs to this repository.
`.gitignore` and `scripts/check_public_release.py` guard the code-only boundary.
The synthetic unit tests run without any project data. They verify implemented
behavior, not the numerical results reported for the private study dataset.

[`configs/synthetic_spatial_split.example.json`](configs/synthetic_spatial_split.example.json)
is an explicitly synthetic format example in local Cartesian coordinates. Its
bounds and block assignments do not describe the study area and must not be
used as the manuscript split. Exact study reproduction requires the authorized
private inputs, fixed split manifest and archived run metadata. The benchmark
runner can create a new fixed split for user-supplied data, which is a new run.

## Reproduce the sample chains

Positive reconstruction:

```powershell
python .\rebuild_positive_samples.py `
  --study-area path\to\study_area.gpkg `
  --known-polygons path\to\known_deposit_polygons.gpkg `
  --tailings path\to\tailings.gpkg `
  --output-dir .\data\positive_reconstruction
```

Pseudo-absence reconstruction:

```powershell
python .\rebuild_negative_samples.py `
  --candidate-grid path\to\full_candidate_grid.gpkg `
  --positive-samples .\data\positive_reconstruction\positive_samples.gpkg `
  --known-zones path\to\known_deposit_polygons.gpkg `
  --structure-zone path\to\structural_buffer.gpkg `
  --lithology-zone path\to\uraniferous_granite.gpkg `
  --lithology-zone path\to\rossing_formation.gpkg `
  --output-dir .\data\reconstructed_samples_v2
```

Detailed rules are documented in
[`docs/positive_sample_reconstruction.md`](docs/positive_sample_reconstruction.md)
and [`docs/negative_sample_reconstruction.md`](docs/negative_sample_reconstruction.md).

## Run the seven-model benchmark

```powershell
python .\run_reconstructed_benchmarks.py `
  --data-root .\data\reconstructed_samples_v2 `
  --feature-dir .\data\factors `
  --prior-dir .\data\priors `
  --output-root .\experiments\reconstructed_v2_spatial_prselected `
  --algorithms ginn,mlp,gcn_transformer,rf,lightgbm,xgboost,catboost `
  --selection-metric pr_auc `
  --device cpu
```

The runner writes one fixed split manifest and reuses it across all seven
models within each realization. GCN–Transformer builds eight-neighbor ego
graphs from training coordinates/features only; held-out labels never enter a
neighborhood. GINN uses the same nine evidence features at inference and adds
three distance-score penalties only to the training loss.

After the primary archive passes its equivalence gate, run the controlled
supplementary experiments with:

```powershell
python .\run_ogr_revision_experiments.py --phase all
python .\run_ogr_sampling_design.py
```

Repeat axes must remain separate; they are not interchangeable replications.

## Generate coordinate-suppressed evidence layers

```powershell
python .\plot_manuscript_evidence_layers.py `
  --feature-dir .\data\factors `
  --output-stem .\publication_figures\figure_2_evidence_layers `
  --overwrite
```

The script preserves native raster extent, pixels and the fixed 0–1 color
scale, while removing coordinate labels, ticks, tick marks and coordinate grid
lines. It retains the north arrow and 10 km scale bar and writes a hash-bearing
export manifest.

## Verify

```powershell
python -m unittest discover -s tests -v
python scripts/check_public_release.py
```

The real-data artifact audit is opt-in because it requires controlled local
inputs and completed experiment archives placed within this checkout. It never
reads a sibling research repository and is skipped in a clean clone:

```powershell
$env:OGR_VERIFY_ARTIFACTS = "1"
python -m unittest tests.test_ogr_revision_artifacts -v
```

## API

```powershell
uvicorn mineral_prediction_api.api:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000/docs`. See [`README_API.md`](README_API.md).

## Citation and rights

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Publication of
this code has been authorized. No open-source license is asserted; publication
does not grant additional reuse rights. Data, coordinates, checkpoints and
derived spatial products remain excluded. See
[`docs/public_release.md`](docs/public_release.md) for the publication boundary
and release-check commands.
