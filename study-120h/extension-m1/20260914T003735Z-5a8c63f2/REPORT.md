# X1R4 A6 solo capability screen

Run `20260914T003735Z-5a8c63f2` completed the sealed M1 schedule on the
`melissa21cn2016` Modal workspace. All 40,960 unique calls in 20 model-task cells
returned `ok`; there were no retries or quarantined payloads. Five A100-40GB
workers ran after the global first-1,000-attempt gate released the remaining
calls. The runner recorded 1.383838 all-worker A100-hours in 1,281 receipts.
Modal's billing summary after completion showed $3.06 metered cost, fully offset
by credits at that time. This cost includes checkpoint staging and other
workspace activity, so it is not an A6-only charge.

The first-1,000-attempt gate recorded zero failures and released the remaining
39,960 calls. Its guarded full-screen forecast was 2.128643 A100-hours; the
80-hour start decision also protected a separate 10-hour reserve. The screen stayed
below the extension meter ceiling.

Seven cells in four families passed the frozen solo capability floors. Phi-4-mini
passed no task because its output rarely matched the shared response grammar.
These are capability-screen results, not evidence about joint-call dependence
or confidence-interval coverage. The next protocol stage is A7, a new admission
lock; no extension joint output has been produced.

| Family | Task | Parse | Accuracy | Error | Decision | Frozen gate failure |
|---|---|---:|---:|---:|---|---|
| qwen | ag_news | 0.998 | 0.690 | 0.310 | exclude | accuracy below floor |
| qwen | civil_comments | 0.991 | 0.757 | 0.243 | admit | — |
| qwen | imdb | 1.000 | 0.947 | 0.053 | admit | — |
| qwen | multi_nli | 1.000 | 0.720 | 0.280 | admit | — |
| granite | ag_news | 0.996 | 0.523 | 0.477 | exclude | accuracy below floor |
| granite | civil_comments | 0.976 | 0.828 | 0.172 | exclude | parse rate below 0.98 |
| granite | imdb | 0.986 | 0.913 | 0.087 | admit | — |
| granite | multi_nli | 0.999 | 0.365 | 0.635 | exclude | accuracy below floor |
| llama | ag_news | 0.999 | 0.615 | 0.385 | exclude | accuracy below floor |
| llama | civil_comments | 1.000 | 0.840 | 0.160 | admit | — |
| llama | imdb | 0.999 | 0.907 | 0.093 | admit | — |
| llama | multi_nli | 0.998 | 0.438 | 0.562 | exclude | accuracy below floor |
| phi | ag_news | 0.098 | 0.180 | 0.820 | exclude | parse, error-information, accuracy floors |
| phi | civil_comments | 0.018 | 0.892 | 0.108 | exclude | parse rate below 0.98 |
| phi | imdb | 0.018 | 0.973 | 0.027 | exclude | parse and error-information floors |
| phi | multi_nli | 0.000 | — | — | exclude | no parsed solo labels |
| mistral | ag_news | 0.995 | 0.579 | 0.421 | exclude | accuracy below floor |
| mistral | civil_comments | 0.979 | 0.780 | 0.220 | exclude | parse rate below 0.98 |
| mistral | imdb | 0.996 | 0.671 | 0.329 | exclude | accuracy below floor |
| mistral | multi_nli | 0.994 | 0.678 | 0.322 | admit | — |

Private first responses, document text, receipts, and exact summaries remain in
`study-120h/ext-work/extension-m1/20260914T003735Z-5a8c63f2/`, which is
gitignored. Audit hashes: `SUMMARY.json`
`01e530951262f26cfa19956e1cd39f555665ea8aa147b18075438a2bf58c041a`,
`admission-inputs.json`
`84a942b7093ceda240731ec92b00811a8fc363461c59fd67eb96c2232cb43e7a`,
`receipts.jsonl`
`ee97b7a12942e2040ae8f10a42785c0671c38080138ab2db580c9c7301305936`.
