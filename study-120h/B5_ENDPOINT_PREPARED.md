# B5 endpoint pilot preparation

Project: IEEE BigData 2026 annotation dependence, branch `main`. This is the separate exploratory `b5-endpoint-pilot-v1` study. The existing five-size B5 seal and smoke remain unchanged, and their scientific outputs will not be pooled with this pilot.

The CPU schedule has four cells, one independently seeded assignment per cell, 8,192 items per cell, and sizes 2 and 32 only. It has exactly 17,408 joint and 65,536 exact-context calls, for **82,944 total**. Every size-32 group is the union of 16 size-2 groups. The source frames, structured schema policy, and context audit pass. The fitted pre-smoke point runtime is 3.3243 A100 hours; applying the B5 smoke's observed 1.5504 generation ratio suggests roughly 4–5 hours. The no-cache bound is 13.0779 hours, so execution must enforce the 7.0-hour stop and live forecast. The secure A100 PCIe price is $1.59/hour, and at least $13.00 live balance is required before any Pod creation.

Preparation artifacts: `extension-b5-endpoint-pilot/preparation/v2/`. The seal and local commit identify the exact live code and protocol. The output directory for a future run must be under `extension-b5-endpoint-pilot/`.

## Commands

Run from `/Users/dz/bigdata/study-120h`.

1. Verify the seal: `.venv/bin/python seal_b5_endpoint.py --verify <seal-directory>`. Success reports matching live and snapshot hashes. Any changed covered file fails.
2. Dry run: `.venv/bin/python run_b5_endpoint.py --dry-run --output extension-b5-endpoint-pilot/run-01 --seal <seal-directory>`. Success reports 82,944 calls and `pod_created: false`. Count or digest mismatch fails.
3. Read-only live preflight: `.venv/bin/python run_b5_endpoint.py --preflight --output extension-b5-endpoint-pilot/run-01 --seal <seal-directory> --confirm <seal-token>`. Success requires at least $13.00 available balance, no running Pod, the approved GPU at no more than $1.59/hour, model volumes in CA-MTL-3, and a valid Runpod CLI API key. Failure creates no Pod.
4. Launch only after preflight passes: `.venv/bin/python run_b5_endpoint.py --launch --output extension-b5-endpoint-pilot/run-01 --seal <seal-directory> --confirm <seal-token>`. Success has 82,944 completed calls. It arms the watchdog before Pod setup, stores durable receipts, enforces the 7.0-hour limit, and deletes each Pod after its model block.
5. Read progress: `.venv/bin/python run_b5_endpoint.py --status --output extension-b5-endpoint-pilot/run-01 --seal <seal-directory>`. Success shows rising completed calls and settled meter phases. A stalled run with an open meter phase requires Pod reconciliation before resume.
6. Resume ordinary interruptions: `.venv/bin/python run_b5_endpoint.py --resume --output extension-b5-endpoint-pilot/run-01 --seal <seal-directory> --confirm <seal-token>`. Receipt reconciliation prevents repeating completed calls. A failed safety gate stops the resume.
7. Analyze after all calls have terminal records: `.venv/bin/python analyze_b5_endpoint.py --events extension-b5-endpoint-pilot/run-01/events.jsonl --output extension-b5-endpoint-pilot/run-01/analysis-a --seal <seal-directory>`. Repeat independently with `analysis-b`; the two `result_sha256` values must match. Missing or foreign calls stop analysis.

The full five-size B5 experiment is outside this pilot authorization. No pilot verdict launches it automatically.
