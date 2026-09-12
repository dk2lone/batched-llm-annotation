# Step 1 exit-gate status

Verdict: **COMPLETE**

Freeze directory: `/Users/dz/bigdata/study-120h/freeze/20260912T041633Z`
Step 1 has been run: True

Step 1 generates solo model responses only. No joint, exact-context, filler, fake-batch, serial, or repeated-sampling response is produced in this step.

| exit-gate item | pass | detail |
|---|---|---|
| Step 0 seal still valid | True | 70 files, manifest matches record True |
| protocol amendment recorded outside the freeze | True | 20260912T202451Z-step1-execution-budget.json |
| exact pinned-tokenizer preflight for every frozen prompt type | True | 8 reports, 0 prompts over budget, 0 inexact |
| host facts recorded for every worker | True | 3 host records, incomplete [] |
| solo ledger complete for every screened cell | True | mistral_civil/civil_comments/mistralai/Mistral-7B-Instruct-v0.3=2048/2048; mistral_secondtask/ag_news/mistralai/Mistral-7B-Instruct-v0.3=2048/2048; qwen_civil/civil_comments/Qwen/Qwen3-8B=2048/2048; qwen_secondtask/ag_news/Qwen/Qwen3-8B=2048/2048; qwen_secondtask/multi_nli/Qwen/Qwen3-8B=2048/2048 |
| no forbidden condition was generated | True | only solo responses exist |
| screen verdict, task choice, and replacement decision recorded | True | outcomes {'qwen_civil/civil_comments': 'PASS', 'qwen_secondtask/ag_news': 'FAIL_CAPABILITY', 'mistral_civil/civil_comments': 'FAIL_CAPABILITY', 'mistral_secondtask/ag_news': 'FAIL_CAPABILITY', 'qwen_secondtask/multi_nli': 'PASS', 'mistral_secondtask/multi_nli': 'FAIL_CAPABILITY'}, second task ag_news |
| throughput benchmark and remaining-ledger forecast written | True | 2 models, 5 cells |
| metered A100 time recorded | True | 4 runs, 0.190 A100 hours |
| no unresolved blocker | True | resolved by frozen screen: ['load-failure--allenai_OLMo-2-1124-7B-Instruct.json'] |

10 of 10 items pass.
