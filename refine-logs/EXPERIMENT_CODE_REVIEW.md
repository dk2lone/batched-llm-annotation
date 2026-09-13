# Experiment code review

Date: 2026-09-13
Scope: second sealed E2 cell, `ibm-granite/granite-3.3-8b-instruct::civil_comments`
Verdict: PASS; no remaining launch blockers

## Reviewed implementation

- `study-120h/modal_freezeb_e2.py`
- `study-120h/src/freezeb/e2.py`
- `study-120h/src/freezeb/e2_amendment.py`
- `study-120h/tests/test_freezeb_e2.py`

The review checked the sealed frame and queue, model revision and checkpoint manifest, task prompt
and parser selection, call counts and identifiers, result namespace isolation, execution-block
ownership, retries, restart behavior, first-1,000-attempt release gate, and all-worker A100
metering.

## Resolved findings

1. Checkpoint verification initially used a Qwen-specific validator. It now selects the Granite
   manifest, verifies its pinned revision and all 15 files, and rejects Qwen identity.
2. A saved first-1,000 reforecast could initially bypass a failed continuation decision. Saved
   barriers now require the exact cell key, plan SHA256, and `may_continue: true`.
3. Cached remote chunks initially replaced session elapsed time with chunk wall time. Recovery now
   retains the original worker session and bills the unseen elapsed delta.
4. Returned payloads were initially validated before metering. Smoke and scientific payloads are
   now meter- and state-recorded first. Invalid payloads are quarantined and excluded from the
   scientific event and timing ledgers. Resume avoids duplicate billing and preserves the original
   timing allocation after an interrupted accepted write.

## Verification

- Ruff formatting and lint: pass
- Strict mypy on the four runner files: pass
- Focused E2 tests: 17 passed
- Full project CPU tests: 275 passed
- Plan-only Granite invocation: pass; no Modal function or GPU started

The reviewed deterministic plan SHA256 is
`81159b7ea6ebb3fcf9ecd5e0a5a1ea26730519c8200cda8ce54f7926c7178f31`.
