# Cross-family extension tracker

Status: the active X1R2 CPU protocol is sealed at `study-120h/freeze/20260913T181826Z-X1R2/`,
manifest SHA256 `0d1e70e26d1a54c98ff708c82452678a262c7586aa62ef3fa3c25fe190daa9cc`.
X1 and X1R1 are preserved but superseded. Zero extension model outputs
exist. No run below is authorized.

| Run ID | Milestone | Purpose | System / variant | Split | Decisive output | Priority | Status |
|---|---|---|---|---|---|---|---|
| X001 | M0 | Verify model access, revisions, licences, file hashes, and chat templates | Qwen3-8B, Granite-8B, Llama-3.1-8B, Mistral-7B-v0.3, Phi-4-mini | Hub metadata only | `CANDIDATES.json` in the X1R2 seal | MUST | DONE (CPU); Llama gated |
| X002 | M0 | Freeze untouched screen, confirmation, and filler frames | Civil, MultiNLI, IMDb train, AG News | New disjoint rows | Frame and ID hashes in the X1R2 seal | MUST | DONE |
| X003 | M0 | Freeze uniform prompts, parsers, seeds, budget, exact schedules, and claim tests | All candidates | CPU fixtures | X1R2 seal plus 55 focused tests | MUST | DONE |
| X003b | M0 | Synthetic runtime lock: context fit, engine, precision, measured timing | Runnable candidates | Synthetic only | Runtime lock and guarded forecast | MUST | TODO (needs GPU authorization) |
| X004 | M1 | Blinded solo capability screen, 40,960 calls in the sealed order | Five models x four tasks | 2,048 rows per cell | Parse, accuracy, class metrics, disagreement | MUST | BLOCKED ON X003b |
| X005 | M1 | Seal admitted cells, the family core, and one E5 cell per family | Deterministic task order | Solo evidence only | X1 admission lock and queue | MUST | BLOCKED ON X004 |
| X006 | M2 | B3 family core, one admitted cell per family | One cell per family | Fresh confirmation rows | Family-equal joint-minus-exact covariance | MUST | BLOCKED ON X005 |
| X007 | M3 | Remaining admitted B3 cells in balanced order | Remaining admitted cells | Fresh confirmation rows | Full signed cell table | MUST | BLOCKED ON X006 |
| X008 | M4 | Establish reference targets | One frozen cell per family | 250 x 32 fresh calls | Reference mean and SE | MUST | BLOCKED ON X005 |
| X009 | M4 | Measure interval coverage | One frozen cell per family | 400 x 32 fresh calls | Coverage, width, variance ratio | MUST | BLOCKED ON X008 |
| X010 | M5 | Increase coverage precision | Same cells | 1,000 then 2,500 evaluations | Tighter coverage intervals | NICE | BLOCKED ON X009 |
| X011 | M5 | Test batch-size and prompt boundaries | One cell per family | k=2,4,8,16,32 | Dose slope and prompt contrasts | NICE | BLOCKED ON X009 |
| X012 | M5 | Test position and composition boundaries | Eligible task cells | Frozen intervention arms | Position/composition contrasts | NICE | BLOCKED ON X009 |

## Open blockers

1. **Llama weight access.** `meta-llama/Llama-3.1-8B-Instruct` is manually gated. Its config and
   chat template return HTTP 401, so its context fit is unverified. It remains in the exact M1
   schedule; the signed runtime/access lock must mark it attempted or technically unavailable.
2. **Synthetic runtime lock (X003b).** No candidate has a measured context fit, engine check, or
   A100 timing under this extension. X004 cannot be forecast or launched until X003b completes.
3. **Budget ceiling.** All 20 cells at 8,192 items project about 86 guarded A100-hours against the
   80-hour stop. The block meter narrows the panel before the ceiling is crossed.

## Extension meter

| Quantity | Value |
|---|---:|
| Metered A100-hours used | 0.000000 |
| Hard stop | 80.000000 |
| Protected reserve | 10.000000 |
| Start guard | 1.6x |
