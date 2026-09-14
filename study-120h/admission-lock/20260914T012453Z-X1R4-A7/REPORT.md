# X1R4 A7 admission decision

The completed solo screen admitted seven of twenty cells in four model families. This is an admission decision, not a joint-dependence or coverage result. No extension joint output was used.

| Family | Task | Solo status | Accuracy | Parse rate | Decision | Items |
|---|---|---|---:|---:|---|---:|
| qwen | ag_news | ok:2048 | 0.690 | 0.998 | accuracy_below_frozen_floor | — |
| qwen | civil_comments | ok:2048 | 0.757 | 0.991 | admitted | 4096 |
| qwen | imdb | ok:2048 | 0.947 | 1.000 | admitted | 2048 |
| qwen | multi_nli | ok:2048 | 0.720 | 1.000 | admitted | 4096 |
| granite | ag_news | ok:2048 | 0.523 | 0.996 | accuracy_below_frozen_floor | — |
| granite | civil_comments | ok:2048 | 0.828 | 0.976 | parse_rate_below_0.98 | — |
| granite | imdb | ok:2048 | 0.913 | 0.986 | admitted | 2048 |
| granite | multi_nli | ok:2048 | 0.365 | 0.999 | accuracy_below_frozen_floor | — |
| llama | ag_news | ok:2048 | 0.615 | 0.999 | accuracy_below_frozen_floor | — |
| llama | civil_comments | ok:2048 | 0.840 | 1.000 | admitted | 4096 |
| llama | imdb | ok:2048 | 0.907 | 0.999 | admitted | 2048 |
| llama | multi_nli | ok:2048 | 0.438 | 0.998 | accuracy_below_frozen_floor | — |
| phi | ag_news | ok:2048 | 0.180 | 0.098 | parse_rate_below_0.98, error_information_outside_0.05_to_0.80, accuracy_below_frozen_floor | — |
| phi | civil_comments | ok:2048 | 0.892 | 0.018 | parse_rate_below_0.98 | — |
| phi | imdb | ok:2048 | 0.973 | 0.018 | parse_rate_below_0.98, error_information_outside_0.05_to_0.80 | — |
| phi | multi_nli | ok:2048 | — | 0.000 | no_parsed_solo_labels | — |
| mistral | ag_news | ok:2048 | 0.579 | 0.995 | accuracy_below_frozen_floor | — |
| mistral | civil_comments | ok:2048 | 0.780 | 0.979 | parse_rate_below_0.98 | — |
| mistral | imdb | ok:2048 | 0.671 | 0.996 | accuracy_below_frozen_floor | — |
| mistral | multi_nli | ok:2048 | 0.678 | 0.994 | admitted | 4096 |

The B3 family core is Qwen/Qwen3-8B::multi_nli, ibm-granite/granite-3.3-8b-instruct::imdb, meta-llama/Llama-3.1-8B-Instruct::civil_comments, mistralai/Mistral-7B-Instruct-v0.3::multi_nli.
The B4 repeated-sampling cells are ibm-granite/granite-3.3-8b-instruct::imdb, meta-llama/Llama-3.1-8B-Instruct::civil_comments, mistralai/Mistral-7B-Instruct-v0.3::multi_nli, Qwen/Qwen3-8B::civil_comments.

Frozen selected calls: 94,336 B3 and 352,000 B4 across all prefixes. Each selected schedule and the full source files have SHA-256 bindings in ADMISSION_LOCK.json.

Four logical A100-40GB worker slots own one family each. These are slot assignments, not a record of provisioned GPUs. A8 must verify actual workers.

The conservative recorded prior use is 2.882445 A100-hours, including a synthetic-runtime proxy. The extension stop remains 80 hours with a protected 10-hour reserve and a 1.6× complete-block start guard. A8 must reforecast after each model's first 1,000 attempts.
