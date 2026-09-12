# Step 0: freeze the 120-hour batched-annotation study

This package performs Step 0 of `../plan-120h-a100.md`. It freezes the protocol, the item
frames, the assignment generator, the analysis tests, the model and engine pins, and the
venue route, before any new model output exists.

No new experimental model labels exist. Step 0 runs on CPU only. It loads no model, uses no
GPU, and produces no joint, exact-context, solo, filler, fake-batch, or repeated-sampling
output. Every threshold in the freeze is a planned decision rule, not a finding.

## What each part does

| Path | Role |
|---|---|
| `src/step0/lock.py` | the frozen protocol; every other file renders or checks it |
| `src/step0/freeze.py` | writes the timestamped freeze, `PROTOCOL.md`, and both locks |
| `src/step0/datafreeze.py` | pins dataset revisions, audits licenses, selects the item frames |
| `src/step0/assign.py` | the length-stratified, position-balanced assignment generator |
| `src/step0/stats.py` | the frozen estimators, the randomization test, the interval methods |
| `src/step0/simulate.py` | the null and power simulations that resolve five protocol fields |
| `src/step0/repro_gatea.py` | reruns the old Gate A analysis and compares its key metrics |
| `src/step0/venue.py` | retrieves both live call pages and records the eligibility text |
| `src/step0/integrity.py` | seals the freeze with SHA-256 and verifies it |
| `src/step0/cli.py` | the `step0` command and the exit-gate check |

## Install

```bash
cd /Users/dz/bigdata/study-120h
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[data,dev]"
```

Success: `uv pip install` ends with a list of installed packages.
Failure: a resolver error names the package it cannot install. Fix the network or the
Python version, then repeat. Python 3.11, 3.12, or 3.13 is required.

## Run the whole of Step 0

Run the steps in this order. Each step writes into the freeze directory that step 2
creates. `freeze/LATEST` points at that directory, so later steps need no `--stamp`.

1. Run the null and power simulations.

```bash
.venv/bin/python -m step0.cli simulate --out simulations
```

Success: the command prints `frozen_conclusions` and writes `simulations/results.json` and
`simulations/REPORT.md`. It takes about 35 minutes on one CPU core.
Failure: a traceback, or `error_information_threshold_error_rate: null`. A null threshold
means no error rate on the grid reached the power target. Read section 4 of the report
before you change anything, and never redefine the effect to make the target pass.

2. Copy the resolved simulation values into `src/step0/lock.py`, then freeze the protocol.

The simulation fixes five protocol fields: the solo-screen minimum error count and rate,
the high-error-rate finding, the matching accuracy ceiling, and the Monte Carlo draw-count
rationale. Write them into `src/step0/lock.py`, then run:

```bash
.venv/bin/python -m step0.cli freeze --simulations simulations
```

Success: the command prints the content hash and the copied simulation hashes, and creates
`freeze/<UTC stamp>/` with `PROTOCOL.md`, `protocol.lock.json`, `protocol.lock.toml`,
`DECISIONS.md`, `decisions.json`, and `simulations/`.
Failure: `already exists` means a freeze with that stamp is present. A freeze is never
edited. Create a new one.

3. Record the live model and engine pin evidence.

```bash
.venv/bin/python -m step0.cli audit-pins
```

Success: each model prints `gated=False revision_matches=True`, and the engine prints
`wheel found True`. The files are `pins/model_pins.json` and `pins/engine_pin.json`.
Failure: exit code 2 and a `BLOCKER-pins.json` file. A gated, private, or disabled model
repo stops Step 1. A human must accept the repository terms first.

4. Freeze the three item frames.

```bash
.venv/bin/python -m step0.cli freeze-data
```

Success: three lines, one per task, each reporting 2,048 items and 4,096 filler records.
Civil Comments also reports 6,000 excluded pilot ids.
Failure: exit code 2 and a `BLOCKER-data-<task>.json` file holding the exact retrieval or
license problem. No frame is written when a task is blocked. This is the machine-readable
incomplete state; do not treat it as a pass.

5. Generate the assignments and the job skeletons.

```bash
.venv/bin/python -m step0.cli assign
```

Success: four lines, one per cell, each reporting the planned call count by condition and
an equal number of unique call ids.
Failure: `frame is missing` means step 4 did not finish. Run step 4 first.

6. Reproduce the old Gate A analysis.

```bash
.venv/bin/python -m step0.cli repro-gatea
```

Success: `verdict: PASS`, `historical artifact unchanged: True`, and `metrics matched:
24/24`.
Failure: exit code 1. Read `repro/REPRO_REPORT.md`. A mismatch means the preserved Gate A
report cannot be regenerated from its own saved rows. Do not cite the pilot until the cause
is found. The rerun reads the historical artifact and writes only into the freeze.

7. Build the venue and authorship record.

```bash
.venv/bin/python -m step0.cli venue
```

Success: all four consistency checks print `true`, and the record names the 6-page
Undergraduate and REU Consortium route.
Failure: exit code 2 and a `BLOCKER-venue.json` file. The call pages must be read live.

8. Run the tests, the linter, and the type checker into the freeze.

```bash
.venv/bin/python -m step0.cli checks
```

Success: `pytest: ok`, `ruff: ok`, `mypy: ok`, and `checks/RESULTS.json` records each exit
code.
Failure: exit code 1. Read the `tail` field of the failing check in that file.

9. Seal the freeze.

```bash
.venv/bin/python -m step0.cli seal
```

Success: the command prints the file count and the manifest hash. Every sealed file becomes
read-only.
Failure: `A sealed freeze is never resealed`. Make a new freeze instead.

10. Verify the seal.

```bash
.venv/bin/python -m step0.cli verify
```

Success: `"ok": true` and an empty `problems` list.
Failure: exit code 1 and one problem line per changed, added, removed, or writable file.
A mismatch voids the freeze. Create a new freeze; never repair a sealed one.

11. Check the exit gate.

```bash
.venv/bin/python -m step0.cli status
```

Success: `Verdict: **COMPLETE**` and a table where every item passes. The command writes
`STATUS.json` and `STEP0_STATUS.md` beside this README, outside the sealed freeze.
Failure: `Verdict: **INCOMPLETE**` and the failing item named in the table. That file is
the machine-readable Step 0 state. Keep it; do not overwrite it with a claim of success.

## Rules this package enforces

1. A frozen file is never overwritten. Every write refuses an existing path.
2. A sealed freeze is immutable. `verify` fails on any change, including a new file.
3. The historical Gate A artifacts are hashed before and after the rerun and must not change.
4. A live retrieval failure writes a blocker and stops. No row is ever fabricated.
5. No repository is called ungated and no license is called redistributable without the
   stored evidence in `pins/` and `data/<task>/provenance.json`.
6. No dataset text is published. The release carries document ids, row indices, character
   counts, and SHA-256 digests.
7. No credential is written into any file, command, log, or report.

## What Step 0 does not decide

The second task stays a Step 1 decision made by the solo-only screen. Step 0 freezes both
candidate frames, AG News and MultiNLI, with their provenance, so neither can be designed
after a screen outcome is known. Step 0 also does not record the CUDA version, the driver
version, the GPU, or the torch version. Those are Step 1 host facts. The serving engine and
its version are already pinned here.
