# B4 two-GPU execution amendment

Recorded: 2026-09-16T14:27:14Z

## Scope

This amendment applies to the Runpod B4/A11 execution only. Daniel authorized starting the
run with two GPUs on 2026-09-16 after the four-GPU capacity watcher did not launch it.

## Runtime change

Each four-cell reference or evaluation round is ordered by its sealed base runtime forecast
and split into two waves of at most two cells. Each wave has its own Runpod session, GitHub
watchdog, budget admission check, lifecycle cleanup, and meter settlement. All four reference
cells must still complete before any evaluation cell begins.

## What remains unchanged

The sealed protocol, models, revisions, datasets, prompts, call identities, serving and decoding
policies, four distinct model-cache volumes, Secure Cloud A100 SXM 80GB requirement, 80 A100-hour
ceiling, 10-hour reserve, 1.6 runtime guard, retry rules, output validation, and analysis plan are
unchanged. This amendment changes only the maximum live GPU count and execution ordering.
