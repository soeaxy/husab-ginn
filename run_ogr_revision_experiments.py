"""Scoped, resumable OGR revision experiments on the real reconstructed sample files.

Repeat axes are deliberately separate: negative-sample realizations, initialization
seeds, and spatial-grid sensitivity are not pooled into a single repeated test.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from run_reconstructed_benchmarks import (
    build_command,
    build_fixed_split_manifest,
    metrics_complete,
    realization_paths,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiments" / "ogr_revision_20260905"
ANALYSIS = ROOT / "analysis_output" / "ogr_revision_20260905"
ARCHIVE = ROOT / "experiments" / "reconstructed_v2_spatial_prselected"
METRICS = ("pr_auc", "roc_auc", "mcc", "recall", "precision", "f1", "brier_score")


@dataclass(frozen=True)
class Run:
    axis: str
    variant: str
    realization: int = 0
    seed: int = 2026
    algorithm: str = "pinn"
    grid: int = 5
    omit_prior: str | None = None
    fixed_weights: bool = False
    scale_argument: str | None = None
    scale_value: float | None = None
    physics_weight: float = 0.1

    @property
    def key(self) -> str:
        return f"{self.axis}/{self.variant}/realization_{self.realization:02d}/seed_{self.seed}"

    @property
    def output(self) -> Path:
        return OUTPUT / self.key


BASELINE = Run("baseline", "full")


def experiment_matrix() -> list[Run]:
    runs = []
    for realization in range(10):
        for prior in ("dome", "fault", "strata"):
            runs.append(Run("negative_realization", f"without_{prior}", realization, omit_prior=prior))
        runs.append(Run("negative_realization", "fixed_weights", realization, fixed_weights=True))
    for argument, default in (("dome-width", 1000.0), ("fault-rate", 0.002), ("strata-width", 800.0)):
        for factor in (0.5, 2.0):
            runs.append(Run("prior_scale", f"{argument}_x{factor:g}", scale_argument=argument, scale_value=default * factor))
    for seed in range(2027, 2036):
        runs.append(Run("initialization", "full", seed=seed))
    for grid in (4, 10):
        for algorithm in ("pinn", "mlp", "rf"):
            runs.append(Run("spatial_grid", f"grid{grid}_{algorithm}", algorithm=algorithm, grid=grid))
    return runs


def strict_zero_matrix() -> list[Run]:
    """Keep the exact GINN initialization, RNG and scheduler; only set lambda=0."""
    return [Run("negative_realization", "strict_zero_regularization", realization, physics_weight=0.0) for realization in range(10)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def manifest_path(grid: int) -> Path:
    return ARCHIVE / "fixed_spatial_split.json" if grid == 5 else OUTPUT / f"fixed_spatial_split_grid{grid}.json"


def prepare() -> None:
    samples = realization_paths(ROOT / "data" / "reconstructed_samples_v2", list(range(10)))
    for grid in (4, 10):
        if not manifest_path(grid).exists():
            build_fixed_split_manifest(samples, manifest_path(grid), grid, 2026, 0.2, 0.2)
    if not manifest_path(5).exists():
        raise FileNotFoundError("The authoritative 5x5 split manifest is required.")
    input_paths = [
        path for sample in samples for path in sample.parent.glob(sample.stem + ".*")
    ] + list((ROOT / "data" / "factors").glob("*.tif")) + list((ROOT / "data" / "priors").glob("*.tif"))
    write_json(OUTPUT / "experiment_manifest.json", {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "repeat_axes": {
            "negative_realization": "10 reconstructed negative-sample sets; model and spatial split seeds fixed at 2026",
            "initialization": "realization_00; model seeds 2026-2035; same fixed 5x5 split; 2026 reuses verified baseline",
            "prior_scale": "realization_00; one parameter at a time x0.5 or x2, including fault decay RATE, not length",
            "spatial_grid": "realization_00; grid4/grid10 GINN, architecture-matched MLP and RF; grid5 uses archive references",
        },
        "training": {"epochs": 80, "patience": 20, "selection_metric": "pr_auc", "physics_weight": 0.1, "batch_size_cpu": 2048, "device": "cpu", "split_seed": 2026},
        "primary_metric": "test_pr_auc",
        "test_metric_used_for_selection": False,
        "sources_sha256": {name: sha256_file(ROOT / name) for name in ("train_multi_physics_model.py", "physics_informed_model.py", "mineral_deep_models.py", "run_ogr_revision_experiments.py")},
        "inputs_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(input_paths)},
        "splits_sha256": {str(grid): sha256_file(manifest_path(grid)) for grid in (4, 5, 10)},
        "strict_zero_comparator": "algorithm=pinn (legacy key for GINN); all three priors retained; same GINN trainer and scheduler; only physics_weight=0. Archived MLP is architecture-matched but its scheduler patience is 10 instead of GINN 15.",
        "runs": [asdict(run) | {"key": run.key} for run in [BASELINE] + experiment_matrix() + strict_zero_matrix()],
    })


def command_for(run: Run, threads: int) -> list[str]:
    command = build_command(
        train_script=ROOT / "train_multi_physics_model.py",
        sample_path=ROOT / "data" / "reconstructed_samples_v2" / f"realization_{run.realization:02d}" / "combined_samples_rebuilt.shp",
        feature_dir=ROOT / "data" / "factors", prior_dir=ROOT / "data" / "priors",
        output_dir=run.output, split_manifest=manifest_path(run.grid), algorithm=run.algorithm,
        model_seed=run.seed, split_seed=2026, grid_size=run.grid, deep_epochs=80,
        patience=20, physics_weight=run.physics_weight, selection_metric="pr_auc", device="cpu",
    )
    command.extend(["--torch-threads", str(threads)])
    if run.omit_prior:
        index = command.index(f"--{run.omit_prior}-tif")
        del command[index:index + 2]
    if run.fixed_weights:
        command.append("--fixed-prior-weights")
    if run.scale_argument:
        command.extend([f"--{run.scale_argument}", str(run.scale_value)])
    return command


def execute(run: Run, threads: int) -> dict[str, Any]:
    command = command_for(run, threads)
    row = asdict(run) | {"key": run.key, "output_dir": str(run.output), "status": "pending", "duration_seconds": 0.0, "exit_code": None}
    if metrics_complete(run.output) and (run.output / "completed.json").exists():
        return asdict(run) | json.loads((run.output / "completed.json").read_text(encoding="utf-8"))
    run.output.mkdir(parents=True, exist_ok=True)
    write_json(run.output / "command.json", {"argv": command, "run": asdict(run), "threads": threads})
    environment = os.environ.copy()
    environment.update({"PYTHONUTF8": "1", "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads), "LOKY_MAX_CPU_COUNT": str(threads)})
    started = time.perf_counter()
    print(f"[start] {run.key}", flush=True)
    try:
        with (run.output / "training.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        row["exit_code"] = result.returncode
        row["status"] = "completed" if result.returncode == 0 and metrics_complete(run.output) else "failed"
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
    row["duration_seconds"] = round(time.perf_counter() - started, 3)
    if row["status"] == "completed":
        metrics = json.loads((run.output / "metrics" / "metrics.json").read_text(encoding="utf-8"))
        row.update({f"test_{name}": metrics["test_metrics"][name] for name in METRICS})
        write_json(run.output / "completed.json", row)
    else:
        write_json(run.output / "failed.json", row)
    print(f"[{row['status']}] {run.key} {row['duration_seconds']}s", flush=True)
    return row


def verify_baseline() -> dict[str, Any]:
    old = ARCHIVE / "realization_00" / "pinn" / "seed_2026" / "metrics"
    new = BASELINE.output / "metrics"
    differences = {}
    for filename in ("metrics.json",):
        a, b = [json.loads((root / filename).read_text(encoding="utf-8")) for root in (old, new)]
        for partition in ("validation_metrics", "test_metrics"):
            for metric in METRICS:
                differences[f"{partition}.{metric}"] = b[partition][metric] - a[partition][metric]
    old_split = pd.read_csv(old / "split_assignments.csv")
    new_split = pd.read_csv(new / "split_assignments.csv")
    same_split = old_split.equals(new_split)
    a, b = [pd.read_csv(root / "test_predictions.csv") for root in (old, new)]
    numeric = a.select_dtypes(include="number").columns
    prediction_max_difference = float(np.max(np.abs(a[numeric].to_numpy() - b[numeric].to_numpy())))
    result = {
        "passed": same_split and max(abs(v) for v in differences.values()) < 1e-5 and prediction_max_difference < 5e-5,
        "split_assignments_exact": same_split,
        "metric_differences": differences,
        "max_numeric_test_prediction_difference": prediction_max_difference,
        "metric_tolerance": 1e-5, "prediction_tolerance": 5e-5,
        "archive": str(old), "rerun": str(new),
    }
    write_json(ANALYSIS / "baseline_equivalence.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def save_ledger(rows: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUTPUT / "run_ledger.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["key"]))


def summarize() -> None:
    rows = []
    references = []
    for realization in range(10):
        references.append((Run("negative_realization", "full_archive", realization), ARCHIVE / f"realization_{realization:02d}" / "pinn" / "seed_2026"))
    references.append((Run("initialization", "full", seed=2026), BASELINE.output))
    for algorithm in ("pinn", "mlp", "rf"):
        references.append((Run("spatial_grid", f"grid5_{algorithm}", algorithm=algorithm), ARCHIVE / "realization_00" / algorithm / "seed_2026"))
    for run, output in [(run, run.output) for run in [BASELINE] + experiment_matrix() + strict_zero_matrix()] + references:
        path = output / "metrics" / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = asdict(run) | {"source_output": str(output), "is_reference": output != run.output}
        row.update({f"test_{name}": metrics["test_metrics"][name] for name in METRICS})
        row.update({f"validation_{name}": metrics["validation_metrics"][name] for name in METRICS})
        row["n_test"] = metrics["test_metrics"]["n_samples"]
        rows.append(row)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).rename(columns={"axis": "repeat_axis"})
    frame.to_csv(ANALYSIS / "all_run_metrics.csv", index=False, encoding="utf-8-sig")
    for axis in ("negative_realization", "initialization", "prior_scale", "spatial_grid"):
        frame[frame.repeat_axis == axis].to_csv(ANALYSIS / f"{axis}_metrics.csv", index=False, encoding="utf-8-sig")
    summary = frame.groupby(["repeat_axis", "variant"])[[f"test_{name}" for name in METRICS]].agg(["count", "mean", "std", "min", "max"])
    summary.to_csv(ANALYSIS / "summary_by_axis_variant.csv", encoding="utf-8-sig")
    write_json(ANALYSIS / "completion_counts.json", {"new_runs_complete": sum((r.output / "completed.json").exists() for r in [BASELINE] + experiment_matrix() + strict_zero_matrix()), "new_runs_expected": 72, "logical_initialization_runs": len(frame[frame.repeat_axis == "initialization"]), "axis_rows": frame.repeat_axis.value_counts().to_dict()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "matrix", "strict-zero", "summarize", "all"), default="all")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4, 8), default=2)
    args = parser.parse_args()
    if args.phase == "summarize":
        summarize()
        return
    prepare()
    if args.phase in ("baseline", "all"):
        row = execute(BASELINE, args.threads)
        save_ledger([row])
        if row["status"] != "completed" or not verify_baseline()["passed"]:
            raise RuntimeError("Baseline equivalence gate failed. Do not run the matrix until diagnosed.")
    if args.phase in ("matrix", "strict-zero", "all"):
        gate = ANALYSIS / "baseline_equivalence.json"
        if not gate.exists() or not json.loads(gate.read_text(encoding="utf-8"))["passed"]:
            raise RuntimeError("A passing baseline equivalence gate is required before the matrix.")
        selected_runs = strict_zero_matrix() if args.phase == "strict-zero" else experiment_matrix()
        if args.phase == "all":
            selected_runs += strict_zero_matrix()
        rows = [asdict(r) | json.loads((r.output / "completed.json").read_text(encoding="utf-8")) for r in [BASELINE] + experiment_matrix() + strict_zero_matrix() if (r.output / "completed.json").exists() and r not in selected_runs]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(execute, run, args.threads) for run in selected_runs]
            for future in as_completed(futures):
                rows.append(future.result())
                save_ledger(rows)
                summarize()
        if any(row["status"] != "completed" for row in rows):
            raise RuntimeError("Some runs failed; inspect the ledger and their individual logs.")
    summarize()


if __name__ == "__main__":
    main()
