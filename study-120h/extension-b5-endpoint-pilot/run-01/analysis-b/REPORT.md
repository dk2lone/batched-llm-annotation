# B5 endpoint pilot

Separate exploratory spending decision; never pool with confirmatory B5.

Completed calls: 82,944. Verdict: **GO**.

Standardized contrasts: size 2 0.078822; size 32 0.040095.
Raw contrasts: size 2 0.014594; size 32 0.006707.
Endpoint difference: -0.038727; 95% cluster bootstrap interval {'low': -0.0555729156003892, 'high': -0.021600227370430675, 'width': 0.033972688229958525}.

## Cells

| Cell | Size 2 | Size 32 | Difference |
|---|---:|---:|---:|
| Qwen/Qwen3-8B::civil_comments | 0.139381 | 0.048514 | -0.090867 |
| ibm-granite/granite-3.3-8b-instruct::imdb | -0.029659 | 0.083246 | 0.112905 |
| meta-llama/Llama-3.1-8B-Instruct::civil_comments | 0.161280 | 0.005910 | -0.155370 |
| mistralai/Mistral-7B-Instruct-v0.3::multi_nli | 0.044287 | 0.022709 | -0.021578 |

## Technical outcomes

{"first_attempt_failures_by_status": {}, "first_attempt_parse_failures_by_status": {}, "retried_calls": 0, "stored_attempts": 82944, "terminal_failures_by_status": {}, "terminal_parse_failures_by_status": {}}
