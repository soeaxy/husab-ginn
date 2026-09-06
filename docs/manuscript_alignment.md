# Manuscript–code alignment record

Scope: Ore Geology Reviews manuscript `ORGEO-D-26-00764` and the code publication
at <https://github.com/soeaxy/husab-ginn>. Internal experiment keys and archived filenames are not
renamed because doing so would break result traceability.

## Terminology contract

- **GINN** is the public/manuscript name: geology-informed neural network.
- `pinn` is the legacy internal algorithm key in archived configurations,
  folders and checkpoints. The CLI and API accept `ginn` and normalize it to
  that stable key.
- The model has no governing-equation residual. Its extra objective is an
  uncertainty-weighted agreement penalty against heuristic geological scores.
- Dome, fault and stratigraphic scores are the three active manuscript priors.
  The registered singularity scale is a legacy optional checkpoint parameter
  and was inactive in all reported runs.

`manuscript_protocol.py` is the machine-readable source for public labels,
sample constants, archived block IDs, scheduler patience values and graph
dimensions. Real spatial bounds and the georeferenced manifest are excluded.

The private experiment manifests retain the SHA-256 hashes of the exact files
used when each run was executed. This synchronized working tree is a later
code/documentation revision: it preserves numerical defaults and archived
compatibility, but it is not represented as byte-identical to every historical
runner. Do not overwrite the original manifests; use their hashes together with
the current regression and real-artifact audits.

## Claim-to-code crosswalk

| Manuscript claim | Code implementation | Verification boundary |
| --- | --- | --- |
| 20 m × 20 m positive grid anchored at the study minimum bounds | `rebuild_positive_samples.build_positive_candidates` | Strictly within study and mapped deposit polygons; x outer/y inner ordering |
| 41,980 candidates; 4,198 sampled positives; `random_state=1` | `rebuild_positive_samples.sample_positive_candidates` | Counts and the coordinate set are verified; row order/new IDs are not claimed as historical identifiers; generic code uses `floor(n/10)` without hard-coding counts |
| Tailings exclusion after the positive draw | `sample_positive_candidates` | Uses `intersects`, so boundary points are also excluded |
| Ten pseudo-absence realizations | `rebuild_negative_samples.SamplingConfig` | Seeds are 2025 + 1009 × realization index |
| 70/5/25 hard/transition/background design | `compute_quotas` and `assign_strata` | Design weights, not estimated geological prevalence |
| ≥1000 m inventory exclusion; 80 m thinning | `SamplingConfig` and `select_spatially_balanced` | Applies to reconstructed pseudo-absences |
| Fixed 5 × 5 spatial benchmark | `fixed_spatial_split.json`, `spatial_block_groups`, `indices_from_block_partitions` | Training: 6, 7, 10, 12, 14, 15, 16, 19, 20, 21, 23, 24; validation: 0, 1, 11, 13; test: 5, 17, 18, 22 |
| Seven-model comparison | `run_reconstructed_benchmarks.DEFAULT_ALGORITHMS` | GINN, MLP, GCN–Transformer, RF, LightGBM, XGBoost, CatBoost |
| GINN 9–64–32–2 backbone | `GeologyInformedClassifier` | 2,790 registered parameters; 2,789 participate when the three priors are active |
| Gaussian dome/strata and exponential fault scores | `make_geological_prior` | Default width/rate: 1000 m, 800 m and 0.002 m⁻¹ |
| Eight-neighbor inductive local GCN–Transformer | `build_spatial_ego_graphs`, `GCNTransformerClassifier` | 64 dimensions, four heads, two pre-normalized layers, feed-forward width 128, dropout 0.1, 76,610 parameters; no neighbor labels |
| Validation AP checkpoint selection | `validation_selection_score` and the primary benchmark runner | Archived JSON calls average precision `pr_auc`; it is not trapezoidal PR integration |
| 76 supplementary real-data runs | `run_ogr_revision_experiments.py` and `run_ogr_sampling_design.py` | 72 regularizer/scale/seed/grid runs plus four train-background runs; repeat axes remain separate |
| Coordinate-free manuscript Figure 2 | `plot_manuscript_evidence_layers.py` | Nine native rasters, fixed 0–1 scale, no coordinate labels/ticks/grid, north arrow and 10 km scale retained |

## Scheduler setting verified from the executed implementation

The archived architecture-matched MLP and GINN did not use the same
`ReduceLROnPlateau` patience. The executed implementation uses:

- GINN: 15;
- MLP and GCN–Transformer trainer: 10.

The manuscript must state this direction exactly. The strict zero-regularization
control remains the cleaner loss-removal comparison because it uses the GINN
trainer and changes only the geological-loss multiplier to zero.

## Results and interpretation boundary

The repository preserves unfavorable results. In the ten-realization primary
benchmark, mean AP is 0.3773 for GINN and 0.3829 for the architecture-matched
MLP; Random Forest leads mean AP at 0.5977. Fixed GINN prior weights and strict
zero regularization reach mean AP 0.3927 and 0.3898. These values are conditional
on one dataset, one fixed spatial partition family and the reconstructed
pseudo-absence protocol.

The positive labels are polygon-membership cells, not independent deposits,
assay thresholds or drill intersections. The pooled 697-cell polygon diagnostic
is descriptive and heterogeneous, not external validation. Original polygon and
prior-digitization provenance, controlled-data permission and independent
geological validation remain unresolved outside the executable code.

## Code-publication boundary

The author authorized a new code-only repository. The controlled data are not
authorized for public release: real rasters, vector labels, drilling data,
exact coordinates, field observations, point-level predictions, checkpoints,
probability TIFFs and local run manifests are excluded. The separate synthetic
split example documents the file format without disclosing the real extent.
The public tests exercise synthetic fixtures; the optional real-artifact audit
requires private records and does not substitute for independent validation.
No open-source or data license is asserted. The new repository does not inherit
the old repository's commits or deleted data objects.
