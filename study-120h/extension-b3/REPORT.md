# X1R4 B3 full-grid execution

The complete four-model by four-dataset B3 grid is execution-complete. This report records
technical completion only. No joint-dependence effect, recurrence claim, or scientific-result
analysis has been calculated.

| Model | Complete cells | First attempts | Retries | Terminal malformed outputs |
|---|---:|---:|---:|---:|
| Qwen3-8B | 4/4 | 137,216 | 0 | 0 |
| Granite 3.3 8B Instruct | 4/4 | 137,216 | 99 | 99 |
| Llama 3.1 8B Instruct | 4/4 | 137,216 | 25 | 25 |
| Mistral 7B Instruct v0.3 | 4/4 | 137,216 | 3 | 3 |
| **Total** | **16/16** | **548,864** | **127** | **127** |

Every scheduled first attempt and required retry has a terminal record. No call is pending and no
irrecoverable technical failure left a cell incomplete. Outputs that remained truncated after the
single allowed retry are retained for all four sealed malformed-output treatments.

The clean Llama rerun supplies the four analysis-eligible Llama cells. Outputs from the interrupted
Llama apps remain excluded from scientific analysis and are preserved only as audit evidence under
the recorded interrupted-run amendment.

The next stage is the prospectively sealed A9 analysis over all sixteen cells. It requires no new
model calls.
