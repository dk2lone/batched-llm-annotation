# Revised B4 repeated-sampling coverage

Post-execution technical amendment using exact linked repair-v1 outputs.
The original prespecified result remains coverage_unavailable_original.

Repairs used: 1095

## ibm-granite/granite-3.3-8b-instruct::imdb

Repaired calls: 730
Reference target error: 0.198156
IID Wald coverage: 0.7775
Call-aware t coverage: 0.9300
Whole-call bootstrap coverage: 0.9175
Reference precise: True

## meta-llama/Llama-3.1-8B-Instruct::civil_comments

Repaired calls: 51
Reference target error: 0.290180
IID Wald coverage: 0.9675
Call-aware t coverage: 0.9650
Whole-call bootstrap coverage: 0.9400
Reference precise: True

## mistralai/Mistral-7B-Instruct-v0.3::multi_nli

Repaired calls: 233
Reference target error: 0.613828
IID Wald coverage: 0.9075
Call-aware t coverage: 0.9525
Whole-call bootstrap coverage: 0.9475
Reference precise: True

## Qwen/Qwen3-8B::civil_comments

Repaired calls: 81
Reference target error: 0.125023
IID Wald coverage: 0.8800
Call-aware t coverage: 0.9600
Whole-call bootstrap coverage: 0.9525
Reference precise: True

## Revised C2 gate

```json
{
  "family": [
    "ibm-granite/granite-3.3-8b-instruct::imdb",
    "meta-llama/Llama-3.1-8B-Instruct::civil_comments",
    "mistralai/Mistral-7B-Instruct-v0.3::multi_nli",
    "Qwen/Qwen3-8B::civil_comments"
  ],
  "holm_adjusted_p": {
    "Qwen/Qwen3-8B::civil_comments": 8.27789468288045e-08,
    "ibm-granite/granite-3.3-8b-instruct::imdb": 5.364719401037443e-32,
    "meta-llama/Llama-3.1-8B-Instruct::civil_comments": 0.9644940958223821,
    "mistralai/Mistral-7B-Instruct-v0.3::multi_nli": 0.000576120695389083
  },
  "min_supported_cells": 2,
  "passed": true,
  "supported_cells": [
    "ibm-granite/granite-3.3-8b-instruct::imdb",
    "Qwen/Qwen3-8B::civil_comments"
  ]
}
```

These are reference-relative results, not verified true semantic accuracy.
