# Positive-sample reconstruction

`rebuild_positive_samples.py` reproduces the positive-label chain described in
the revised manuscript without project-specific absolute paths.

## Reconstructed rule

1. Read a polygonal study area and use its minimum x/y bounds as the grid
   anchor.
2. Generate coordinates with `numpy.arange(minimum, maximum, 20)` using x as
   the outer loop and y as the inner loop.
3. Retain only points **strictly within** the study-area union. Boundary points
   are not retained.
4. Retain points **strictly within** the mapped known-deposit polygon union and
   assign `Class=1`. Polygon-boundary points are not positive candidates.
5. Set the draw size to `floor(candidate_count / 10)` and call pandas
   `sample(n=..., random_state=1, replace=False)` on the combined ordered
   candidate table.
6. After the random draw, remove selected points that intersect the optional
   tailings polygon union. `intersects` also excludes points on a tailings
   boundary.

This is a polygon-membership label, not an assay-grade or drill-intersection
label. The script does not infer the geological authority or status of the
input polygons.

## Command

```powershell
python rebuild_positive_samples.py `
  --study-area path/to/study_area.gpkg `
  --known-polygons path/to/known_deposits.gpkg `
  --tailings path/to/tailings.gpkg `
  --output-dir path/to/positive_reconstruction
```

`--tailings` is optional. The manuscript defaults (20 m, divisor 10 and random
state 1) can be changed explicitly with `--spacing-m`, `--sample-divisor` and
`--random-state`; changing them creates a sensitivity run rather than the
manuscript reconstruction.

Inputs without CRS metadata are rejected by default. If the source projection
is independently known, pass it explicitly, for example
`--assume-missing-crs EPSG:32733`. The assumption is recorded in the manifest;
the script never silently adopts the study-area CRS for an unreferenced layer.

The output directory contains:

- `positive_candidates.gpkg`: all ordered polygon-contained candidates;
- `positive_samples.gpkg`: the sampled positives remaining after tailings
  exclusion;
- `positive_sampling_manifest.json`: operation order, parameters, CRS, bounds,
  counts, uniqueness checks, and SHA-256 hashes for every input and output
  vector dataset.

Re-running without `--overwrite` refuses to replace generated outputs. With
`--overwrite`, only these three owned output names are replaced; unrelated files
in the directory are neither deleted nor modified.

## Husab verification result

For the audited Husab inputs, strict polygon containment produced 41,980
positive candidates. `floor(41980 / 10)` produced a 4,198-row draw, and the
subsequent tailings exclusion removed zero selected points. These numbers are
dataset-specific verification facts and are deliberately not enforced by the
generic code.

The reconstructed 4,198-coordinate **set** matches the archived positive set.
The audit does not treat row order or newly generated `sample_id` values as
historical identifiers. Use the archived combined samples and fixed split
manifest when reproducing the reported 70-run benchmark; use this script to
reconstruct and audit the label-generation rule.

## Tests

```powershell
python -m unittest tests.test_positive_sampling -v
```

The synthetic tests cover minimum-bound anchoring, x/y ordering, strict boundary
behavior, exact pandas determinism, floor division, sampling without
replacement, post-draw tailings exclusion (including boundary intersection),
manifest hashes, and safe overwrite behavior.
