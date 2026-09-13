# Operational amendment: first Freeze B E2 worker concurrency

Date: `20260913T045515Z`
Amendment: `freezeb-e2-qwen-multinli-concurrency`
Admission lock SHA256: `cd60cad6d0af27efc61ab0bc7605e352250cdad88702824bb4b2b841e882a3c8`
Plan SHA256: `d642f6c3fd5deaaefce76aadcaeeef9d577a20df513c999d715fae8e21672409`

This record is written before any E2 model output. The first sealed E2 cell may use
at most 4 independent one-A100 workers. Execution
block ownership is fixed by block number modulo worker count; blocks are never split,
and their internal call order remains unchanged.

Every worker's metered A100 seconds, including model loading and the synthetic
equivalence smoke, are summed. Parallelism changes elapsed time, not the study meter.
Scientific output remains blocked until the smoke report verifies identical engine,
checkpoint, tokenizer, serving settings, and deterministic synthetic response.

This amendment changes no cell, item, assignment, prompt, call ID, analysis, or claim
rule in the sealed Freeze B protocol.
