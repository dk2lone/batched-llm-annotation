# Step 1 solo-only screen

Freeze: `20260912T041633Z`
Written: 2026-09-12T20:55:25.831774+00:00

Only solo first responses, pinned-tokenizer counts, and parse statuses reach this report.
No joint, exact-context, filler, fake-batch, or repeated-sampling value exists in Step 1.
Throughput and cost live in the separate benchmark report and are not screen evidence.

| cell | model | frame | outcome | parsable rate | accuracy | solo errors |
|---|---|---|---|---:|---:|---:|
| `qwen_civil` | `Qwen/Qwen3-8B` | civil_comments | **PASS** | 0.9897 | 0.8379 | 311 |
| `qwen_secondtask` | `Qwen/Qwen3-8B` | ag_news | **FAIL_CAPABILITY** | 0.9995 | 0.6270 | 763 |
| `mistral_civil` | `mistralai/Mistral-7B-Instruct-v0.3` | civil_comments | **FAIL_CAPABILITY** | 0.9951 | 0.6909 | 623 |
| `mistral_secondtask` | `mistralai/Mistral-7B-Instruct-v0.3` | ag_news | **FAIL_CAPABILITY** | 0.9946 | 0.6211 | 765 |
| `mistral_civil` | `allenai/OLMo-2-1124-7B-Instruct` | civil_comments | **FAIL_CAPABILITY** | 0.0000 | 0.0000 | 0 |
| `mistral_secondtask` | `allenai/OLMo-2-1124-7B-Instruct` | ag_news | **FAIL_CAPABILITY** | 0.0000 | 0.0000 | 0 |
| `qwen_secondtask` | `Qwen/Qwen3-8B` | multi_nli | **PASS** | 0.9990 | 0.7402 | 530 |
| `mistral_secondtask` | `allenai/OLMo-2-1124-7B-Instruct` | multi_nli | **FAIL_CAPABILITY** | 0.0000 | 0.0000 | 0 |

## Second-task decision

- chosen second task: `ag_news`
- reason: neither frozen task passes for both models, so AG News is kept, its failing cell is published as uninformative, and the claim narrows to the tested task
- claim narrowed: True

## Second-model replacement decision

- replace: False
- reason: keep the failed cell in the published panel labelled uninformative, narrow the claim to the tested model, and do not write 'across models'.
- replacement model: allenai/OLMo-2-1124-7B-Instruct
- information-poor cells kept in the panel: []

An INFORMATION_POOR cell is never replaced. It stays in the published panel with
uncertainty bounds and the label 'insufficient events for a precise covariance estimate'.
