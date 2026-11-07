# Historical experiment implementation

This directory preserves the implementation and records used to run the batched annotation
study. For reproducing the paper's statistics from the included error arrays, start with
[`../analysis/`](../analysis/). That package has pinned dependencies and a single `reproduce`
command, and needs no model inference.

The code here spans several successive designs. Names such as Step 0, Freeze B, X1R4, and
B3–B5 refer to those recorded stages. Older status files describe their own dates and are not
a current job queue. Preparation files and launch instructions do not indicate that a run is
currently active.

## Code map

| Location | Purpose |
|---|---|
| [`src/step0/`](src/step0/) | Initial protocol freeze, data selection, assignments, hashes, integrity checks, and simulations |
| [`src/step1/`](src/step1/) | Initial solo screening, runtime measurements, call ledgers, and runner |
| [`src/step2/`](src/step2/) | Historical replication bridge and its analysis |
| [`src/freezeb/`](src/freezeb/) | Later screening, cross-family extension, repeated sampling, batch-size study, and recovery |
| [`tests/`](tests/) | Tests for those historical stages and their execution contracts |
| [`pyproject.toml`](pyproject.toml), [`uv.lock`](uv.lock) | Historical package and dependency environment, separate from the publication analysis package |

## Entry points by stage

The scripts at the directory root are grouped below by their role. `modal_*.py`, `run_*.py`,
and Runpod drivers can launch remote inference; they are not needed for CPU reproduction.

| Stage | Entry points | Related records |
|---|---|---|
| Initial solo screen | `modal_step1.py` | `step1-out/`, `STEP1_STATUS.json` |
| Historical replication bridge | `modal_bridge.py`, `analyze_step2_bridge.py` | `step2-out/bridge/` |
| Freeze B screening and calibration | `modal_freezeb_benchmark.py`, `modal_freezeb_screen.py` | `freezeb-screen/`, `runtime-lock/` |
| Freeze B matched comparison | `modal_freezeb_e2.py`, `analyze_freezeb_e2.py` | `e2-out/` |
| Extension frames and runtime checks | `build_extension_frames.py`, `modal_extension_stage.py`, `modal_extension_batch_invariance.py` | `extension-staging/`, `extension-runtime/`, `extension-batch-invariance/` |
| Extension solo screen | `modal_extension_m1.py` | `admission-lock/` |
| Full 16-cell grid (B3) | `modal_extension_b3.py`, `analyze_extension_b3.py` | `extension-b3/`, `full-grid-lock/`, `execution-release/` |
| Repeated sampling (B4) | `prepare_extension_b4_budget.py`, `modal_extension_b4.py`, `runpod_extension_b4.py`, `analyze_extension_b4.py` | `extension-b4/` |
| B4 serving and transport | `b4_prepare_cache.py`, `b4_supervised_smoke_request.py`, `b4_runpod_worker.py`, `extension_b4_runpod.py`, `extension_b4_transport.py`, `reconcile_extension_b4_meter.py` | B4 preparation and metering records |
| Output recovery | `prepare_output_recovery.py`, `run_output_recovery.py`, `output_recovery_worker.py` | `output-recovery/`, `amendments/` |
| Recovered sensitivity analyses | `analyze_b3_repaired.py`, `analyze_b4_revised.py` | Timestamped repaired/revised analysis directories under `extension-b3/` and `extension-b4/` |
| Five-size experiment (B5) | `prepare_b5.py`, `seal_b5.py`, `run_b5.py`, `b5_worker.py`, `analyze_b5.py` | `extension-b5/`, `B5_PROTOCOL.md` |
| Separate endpoint pilot | `prepare_b5_endpoint.py`, `seal_b5_endpoint.py`, `run_b5_endpoint.py`, `b5_endpoint_worker.py`, `analyze_b5_endpoint.py` | `extension-b5-endpoint-pilot/`, `B5_ENDPOINT_PILOT_PROTOCOL.md` |
| Post-hoc CPU analyses | `analyze_a1_a2.py`, `analyze_cpu_followups.py`, `analyze_decision_audit.py` | `cpu-followups/`, `decision-audit/` |
| Protocol sealing | `seal_freezeb.py`, `seal_freezeb_part2.py`, `seal_freezeb_runtime.py`, `seal_extension.py`, `seal_extension_admission.py`, `seal_extension_full_grid.py`, `seal_extension_full_grid_budget.py`, `seal_extension_full_grid_execution.py` | Protocols, seals, admission and execution locks |
| Historical cloud watchdog | `runpod_watchdog.py` | Original lifecycle and termination checks |

## Evidence and interpretation

| Evidence | Main location | Interpretation |
|---|---|---|
| Primary covariance result | [`extension-b3/analysis/`](extension-b3/analysis/) | Original sealed 16-cell analysis with all prespecified malformed-output treatments |
| Recovered covariance sensitivity | [`extension-b3/repaired-analysis-20260918T041906Z/`](extension-b3/repaired-analysis-20260918T041906Z/) | Separate linked-recovery sensitivity; does not replace the primary result |
| Recovered coverage sensitivity | [`extension-b4/revised-analysis-20260918T034636Z/`](extension-b4/revised-analysis-20260918T034636Z/) | Secondary post-execution analysis; the original status remains `coverage_unavailable_original` |
| Batch-size results | [`extension-b5/run-01/analysis-final-a/`](extension-b5/run-01/analysis-final-a/) | Four selected cells and sizes 2–32; use the paper's descriptive interpretation of trends |
| Endpoint pilot | [`extension-b5-endpoint-pilot/`](extension-b5-endpoint-pilot/) | Separate experiment, never pooled into the five-size B5 study |
| Protocol changes | [`amendments/`](amendments/) | Dated changes and their scope |

Large raw outputs, private dataset frames, and some sealed bundles are excluded from the
working tree. Earlier run archives are attached to
[GitHub releases](https://github.com/dk2lone/batched-llm-annotation/releases). The historical
event-level analysis scripts therefore need more inputs than a source-only clone provides.

Keep timestamped results, seals, locks, and linked retries intact. Renaming their paths or
editing sealed source snapshots can break hash verification and provenance.

## Publication outputs

There are two output routes with different purposes:

| Route | Inputs | Outputs |
|---|---|---|
| `analysis/`: `uv run reproduce all` | Included error arrays, saved replays, and protocol | Recomputed statistics and reproduction plots under `analysis/outputs/` |
| `manuscript/build_tables.py` | Included historical summaries at its recorded paths | Camera-ready tables, CSVs, and a hash record under `manuscript/tables/` |

`manuscript/make_figure.py` uses `manuscript/figure_source_data.csv` for the manuscript's
result figures. `manuscript/make_study_overview.py` draws the schematic and uses the included
`manuscript/figures/assets/annotation_robot_flat.png` illustration. The reproduction plots
and manuscript figures have different layouts; generating `analysis/outputs/` does not
overwrite the camera-ready figures.

To reproduce the statistics and run the publication tests, use these commands from the
repository root:

```sh
cd analysis
uv sync --locked --extra dev
uv run reproduce all
uv run pytest
uv run ruff check .
```

The commands write only local analysis outputs and caches. Success produces result JSONs
and figures under `analysis/outputs/`, a passing test suite, and no Ruff errors. They do not
launch GPU work. See the [analysis README](../analysis/README.md) for individual commands
and the limits of reproduction from stored arrays.
