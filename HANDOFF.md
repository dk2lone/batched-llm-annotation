# IEEE BigData 2026 handoff

Updated: 2026-09-15 America/Toronto (2026-09-16 UTC).

## Current work

Project: dependence in batched LLM text annotation. Repository:
`dk2lone/batched-llm-annotation`, private, branch `main`. The latest implementation
commit is `ed9c633` (`guard runpod provisioning and resolve saved login`).
The previous handoff is preserved at `git show 624c708:HANDOFF.md`.

The current task is preparing the B4 / A11 repeated-annotation experiment for
Runpod. It tests whether correlated errors within shared annotation calls make
ordinary confidence intervals too narrow. It compares item-independent intervals,
call-level intervals, and a whole-call bootstrap. Planned comparisons are not findings.

The initial checkpoint is 250 reference and 400 evaluation replicates per cell,
83,200 first calls across four cells. Each evaluation replicate contains 512 item
appearances in 32 size-16 calls. The 1,000 and 2,500 evaluation prefixes remain
nested extensions, not the current initial launch. Preserve the 80-A100-hour
extension ceiling, 10-hour reserve, and 1.6 forecast guard. The separate historical
120-hour study is not a replacement budget for this extension.

The selected execution configuration is four parallel, separately terminated,
single-GPU Secure Cloud A100 SXM 80GB Pods, one per cell:

- Granite 3.3 8B Instruct / IMDb.
- Llama 3.1 8B Instruct / Civil Comments.
- Mistral 7B Instruct v0.3 / MultiNLI.
- Qwen3 8B / Civil Comments.

All four reference streams finish before evaluation starts. This is not a
four-GPU tensor-parallel model. No fallback to Community Cloud or 40GB is configured.

## Implementation and preparation

The Runpod runner, SSH transport, offline inference worker, budget checks,
durable raw receipts, recovery, independent lane cleanup, and CPU plan preparation
are implemented. Local verification passed 509 regression tests, 64 focused B4
tests, scoped Ruff, and strict mypy across 62 source files. No Runpod scientific
annotation calls or GPU smoke tests have run.

The active CPU-only plan is
`study-120h/extension-b4/preparation/20260915-runpod-cli-guard-cpu/PLAN-400.json`.
Its plan SHA-256 is
`24f95bcd628cd9f90325169cb4961e11dc22279ca010b18eeb999a997a19e56d`.
Earlier timestamped preparation plans remain untouched. See
`study-120h/extension-b4/VALIDATION.json` for implementation hashes and checks.

The pinned Linux runtime was built, CPU-version checked, and published successfully by
`.github/workflows/build-b4-runtime.yml` in
[GitHub Actions run 35041546684](https://github.com/dk2lone/batched-llm-annotation/actions/runs/35041546684).
The immutable image is
`ghcr.io/dk2lone/b4-a100-runtime@sha256:1cf3be1dfb50dbb800a658b371e8ec58f1ea485b55bae5b92ee281af2feba898`
from commit `8bcd697355182c81a32f46079054a0fea53bae0e`.
The image contains public runtime dependencies only, without project code,
private data, model weights, or credentials. The GHCR package is private: an
anonymous registry pull returned HTTP 401, and the Runpod account has no configured
container-registry authentication. Runpod pull access therefore remains unverified.
The validation JSON predates completion of the image-build step; its
`runtime_image_built: false` is the earlier validation snapshot.

Runpod browser login succeeded. The transport reads Flash's saved default
credential profile in memory without rewriting the credential file. A live read-only
check on 2026-09-15 America/Toronto showed $17.9771 balance, $0/hour active spend,
no Pods, no network volumes, and two registered SSH keys. Secure A100 SXM 80GB
pricing was $1.59/GPU-hour with low stock in several data centers.
Recheck balance, rate, and four separate one-GPU allocations before paid work.
These checks do not establish that the full experiment is funded.

All four sealed Hugging Face revisions resolved exactly through the saved access,
including the gated Llama revision. This verifies metadata access, not downloaded
weights or an offline cache. No Runpod network volumes or model caches have been
created by this preparation.

## Blocker and authority

Runpod CLI 2.14.0 removed `--terminate-after`, which the earlier runner required as a
server-side cost guard. Live CLI help and the REST v2 OpenAPI schema have no
automatic-termination field. Runpod's merged removal PR documents that the old
GraphQL fields were accepted but ignored while Pods continued billing; the restore
PR remains a draft pending backend enforcement. There is currently no supported,
verifiable server-side Pod deadline to substitute. The current working tree replaces
that rejected path with a pre-armed GitHub Actions watchdog, normal local deletion and
a separately verified prepaid-balance ceiling. The new path is not live-validated yet.

Read-only 2.14.0 response shapes were checked for user, Pod list, volume list, SSH
key list, GPU catalog, and registry list. User is an object; Pod, volume, and GPU
lists are arrays; SSH keys use a `{"keys": [...]}` envelope; an empty registry list
is JSON `null`. A live create response cannot be verified without violating the
cost guard.

Daniel authorized preparation and a bounded paid synthetic smoke test, not
the full scientific experiment. Paid work is paused at this compatibility
blocker. Do not remove the guard, fabricate passed preflight checks, or launch
the full runner to bootstrap its own smoke-test requirement.

Two CPU-only lifecycle probes on 2026-09-15 America/Toronto tested whether a
Pod could use container exit as the missing guard. An Alpine Pod with a
15-second command and a one-shot `hello-world` Pod both remained `RUNNING`
after their application command should have exited. Both were deleted, a live
list then showed no active Pods and active spend returned to $0/hour. Their
combined compute charge is bounded above by $0.003 at the returned $0.06/hour
CPU rate. Container exit is therefore not an acceptable Pod spending guard.

## Next action

Keep the full experiment on the selected Secure Cloud A100 SXM 80GB Pod/SSH
path. Do not use a Serverless-only smoke test. Replace the unavailable Runpod
deadline with three independent controls before any GPU allocation:

1. Pre-arm a GitHub Actions watchdog and verify its deletion step is already running before
   creating a Pod. It watches one unique run prefix, records the exact Pod IDs and
   deletes them at the shutdown target. Limit each target to five hours so it fits
   within the six-hour GitHub-hosted job limit. This is not a guaranteed lease.
2. Keep normal runner teardown and add a scheduled reconciliation workflow as a
   backup. Durable receipts on the four network volumes allow an incomplete lane
   to resume under a new lease without repeating accepted scientific calls.
3. Verify auto-pay is disabled and restrict prepaid Runpod credit to the authorized
   phase plus its storage reserve. The balance is the final dollar ceiling if both
   deletion paths fail; Runpod documents that low balance stops Pods.

The watchdog needs a newly rotated, least-privilege Runpod key stored only as a
GitHub Actions secret. The previously exposed key must not be reused. Test the
complete lease path on CPU first, then use the already authorized bounded A100
synthetic smoke test to verify image pull, exact hardware, offline inference,
receipt recovery, watchdog deletion and zero remaining Pods. Only that same path
is eligible for the full experiment. Fresh financial evidence and separate
authorization are required before full scientific execution.

Separately configure least-privilege GHCR pull access because the published package
is private. Exact offline model caches remain uncreated.

Read `study-120h/extension-b4/runpod/OPERATIONS.md` and
`study-120h/EXTENSION_B4_OPERATIONS.md` for the execution gates. Consult
`refine-logs/EXPERIMENT_PLAN.md` for the extension design and
`plan-120h-a100-rebased.md` for the separate budgeted study.

Do not rerun completed stages, edit sealed protocols, or overwrite timestamped
data, seals, locks, or plans. Keep the Gate A pilot separate from fresh
confirmation and coverage results. Preserve the unrelated untracked `.flash/`
directory. Never copy credentials into code, commands, handoffs, or reports.
The Runpod key previously pasted in chat should be rotated. Do not invoke
Claude CLI unless Daniel explicitly requests it.
