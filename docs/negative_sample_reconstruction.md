# Negative-sample reconstruction protocol

## Why this module exists

The archived Husab training sample set uses negatives that are spatially easy to separate from positives. This module rebuilds pseudo-absence samples so that negatives remain outside known ore zones but re-enter structurally and lithologically favorable settings as hard negatives.

## Core protocol

1. Keep the archived positive samples unchanged.
2. Use the full fishnet candidate grid as the negative candidate pool.
3. Exclude candidates whose exact point-to-polygon distance from a known mineralized zone is less than 1,000 m.
4. Keep only candidate cells with complete prior and factor coverage.
5. Derive a geological hardness score as `max(p_dome, p_fault, p_strata)`.
6. Partition eligible negatives into three mutually exclusive strata:
   - `hard`: inside the structural buffer, inside favorable lithology, or geological score >= 0.50
   - `transition`: not hard and geological score >= 0.20
   - `background`: all remaining eligible cells
7. For each realization, sample negatives at the original 1:10 positive:negative ratio using fixed quotas:
   - 70% hard
   - 5% transition
   - 25% background
8. Enforce one sample per 80 m thinning cell and balance selections across 1 km blocks.
9. Generate 10 realizations with different deterministic seeds.
10. Record SHA-256 hashes for every source shapefile sidecar, factor raster, prior raster, and generated dataset.

The 70/5/25 quotas are design weights, not claims about geological prevalence. The eligible candidate pool is approximately 78.0% hard, 3.4% transition, and 18.6% background; the selected weights retain a hard-negative majority while modestly increasing transition/background representation so training is not dominated by one stratum.

## Main files

- `rebuild_negative_samples.py`
- `analyze_reconstructed_samples.py`
- `tests/test_negative_sampling.py`

## Output structure

The rebuild step writes:

- `data/reconstructed_samples_v2/protocol.json`
- `data/reconstructed_samples_v2/realization_summary.csv`
- `data/reconstructed_samples_v2/realization_XX/negative_samples_rebuilt.*`
- `data/reconstructed_samples_v2/realization_XX/combined_samples_rebuilt.*`
- `data/reconstructed_samples_v2/realization_XX/manifest.json`

The analysis step writes:

- `analysis_output/negative_sample_reconstruction/analysis_report.md`
- `analysis_output/negative_sample_reconstruction/figure_catalog.md`
- `analysis_output/negative_sample_reconstruction/statistics_appendix.md`
- `analysis_output/negative_sample_reconstruction/figure_manifest.json`
- `analysis_output/negative_sample_reconstruction/*.csv`
- `analysis_output/negative_sample_reconstruction/figure_*.png`
- `analysis_output/negative_sample_reconstruction/figure_*.pdf`

These are module-local QA figure numbers, not manuscript figure numbers. The
coordinate-suppressed manuscript Figure 2 is generated separately by
`plot_manuscript_evidence_layers.py`.

## Field definitions

### Combined/negative sample outputs

| Field | Meaning |
| --- | --- |
| `sample_id` | Deterministic sample identifier |
| `Class` | Binary label, `1` for positive and `0` for negative |
| `source` | `known_zone` for positives, `pseudo_abs` for rebuilt negatives |
| `stratum` | `positive`, `hard`, `transition`, or `background` |
| `zone_id` | Positive known-zone identifier; `-1` for negatives |
| `realiz` | Realization index |
| `geo_score` | `max(p_dome, p_fault, p_strata)` |
| `p_dome` | Dome-distance prior |
| `p_fault` | Fault-distance prior |
| `p_strata` | Strata-distance prior |
| `in_struct` | Whether the cell is inside the structural constraint zone |
| `in_lith` | Whether the cell is inside favorable lithology |
| `mine_dist` | Distance to the nearest known mineralized zone in meters |
| `block_id` | 1 km balancing block identifier |
| `thin_id` | 80 m thinning-cell identifier |
| `geometry` | Sample point geometry |

## Rebuild command

```powershell
.\.venv\Scripts\python.exe .\rebuild_negative_samples.py `
  --candidate-grid path/to/husab_fishnet.shp `
  --known-zones path/to/known_deposit_polygons.shp `
  --structure-zone path/to/structural_buffer.shp `
  --lithology-zone path/to/uraniferous_granite.shp `
  --lithology-zone path/to/rossing_formation.shp `
  --output-dir .\data\reconstructed_samples_v2 `
  --overwrite
```

`--overwrite` only clears files matching this generator's exact output contract. If the output directory contains an unknown file or subdirectory, the command stops without deleting it.

## Analysis command

```powershell
.\.venv\Scripts\python.exe .\analyze_reconstructed_samples.py `
  --reconstructed-root .\data\reconstructed_samples_v2 `
  --known-zones path/to/known_deposit_polygons.shp
```

## Training usage

To swap one rebuilt realization into the current training pipeline, point `train_multi_physics_model.py` to a combined shapefile:

```powershell
.\.venv\Scripts\python.exe .\train_multi_physics_model.py `
  --shapefile .\data\reconstructed_samples_v2\realization_00\combined_samples_rebuilt.shp `
  --dome-tif .\data\priors\dome.tif `
  --fault-tif .\data\priors\fault.tif `
  --strata-tif .\data\priors\strata.tif `
  --split-mode spatial_block `
  --spatial-block-grid 5
```

The current loader only requires a valid shapefile with a `Class` field and point geometry, so the rebuilt samples are backward-compatible.

The supplied one-epoch run under `analysis_output/negative_sample_reconstruction/training_smoke` is compatibility evidence only. It must not be reported as manuscript performance. Final comparative experiments should repeat training across the ten realizations and multiple model seeds.

## Current limitations

- The protocol still depends on the archived positive set; it does not redesign positive labeling.
- Hardness is prior-driven and therefore limited by the quality of the dome, fault, and strata distance rasters.
- The analysis report is a sampling QA package, not a substitute for downstream ablation and cross-seed model evaluation.
