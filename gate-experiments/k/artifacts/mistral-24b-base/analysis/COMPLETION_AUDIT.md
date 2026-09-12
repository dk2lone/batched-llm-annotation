# Gate K completion audit

Generated 2026-09-11 after Modal disabled the workspace during the final resume.

## Decision

Stop Gate K. The tested mechanism is unsupported for
`mistralai/Mistral-Small-3.1-24B-Base-2503` at revision
`ba6496e3dce1d0bdc93848804b1d4b9d5f3c57bc`.

The run preserved 1,266 unique results from the frozen 1,296-job plan. The 30 missing jobs do
not prevent the stop decision:

- 28 are sparse 8-dimensional base-order jobs. The frozen gate uses dimensions 32 and 64.
- One is a sparse 64-dimensional nearest-first ordering control. Ordering is descriptive and
  never decides the gate.
- One is a sparse 64-dimensional random-removal job. All primary representation and accuracy
  cells are complete. This missing result can move the already-passing nearest-removal
  selectivity from 0.188 to no lower than 0.177, still above its 0.05 threshold.

The conditions that reject K therefore use complete data and cannot be rescued by the missing
jobs:

| Condition | Observed | Required |
|---|---:|---:|
| Representation contrast, sparse over dense | 0.305 | at least 1.200 |
| Representation contrast, sparse over nuisance | 0.578 | at least 1.200 |
| Accuracy, sparse minus dense | -0.125 | at least 0.050 |

The model-capability anchor passed at 0.896 accuracy with no parse failures. The negative result
is therefore not explained by inability to solve the two-dimensional task.

## Execution notes

`Qwen/Qwen3-8B` was unsuitable for this frozen gate because 576 prompts exceeded its 32,768-token
context and its 48-job anchor produced 79.2% off-label first tokens. The Mistral 24B Instruct
checkpoint solved the anchor under restricted A/B scoring but produced 47 off-label first tokens
out of 48. The Base checkpoint continued directly with A or B, passed the unchanged anchor, and
was used for the mechanism test.

The first 265 Base results captured `hidden_states[27]` from the full model output. The remaining
1,001 captured the input to decoder layer 27 with a forward pre-hook to reduce GPU memory. These
are the same layer boundary: the hidden-state tuple includes the embedding output at index zero,
so index 27 is the output of layer 26 and input to layer 27.

The final run stopped because Modal reported that the workspace was disabled. No credentials are
stored in this artifact or the experiment configuration.
