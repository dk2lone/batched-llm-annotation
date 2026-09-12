"""The frozen Step 0 protocol.

Every value here is a design decision that must not change after the first new
joint output is opened. `PROTOCOL` is the single source of truth; the TOML and
JSON locks and PROTOCOL.md are all rendered from it.

Nothing in this module reports a result. The study has produced no new joint,
exact-context, solo, or repeated-sampling output. Every threshold is a planned
decision rule.
"""

from __future__ import annotations

from typing import Any

PROTOCOL_VERSION = "1.0.0"
PROTOCOL_ID = "bigdata2026-batch-dependence-step0"

NO_NEW_LABELS = (
    "No new experimental model labels exist. Step 0 ran on CPU only. No model was loaded, no "
    "GPU was used, and no joint, exact-context, solo, filler, fake-batch, or repeated-sampling "
    "output has been produced."
)

# --------------------------------------------------------------------------------------
# Panel, data, and models
# --------------------------------------------------------------------------------------

PANEL: dict[str, Any] = {
    "shape": "2x2 model-task panel plus two repeated-sampling cells",
    "batch_size_k": 16,
    "records_per_task": 2048,
    "assignments_per_cell": 3,
    "decoding": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 0,
        "rule": "greedy",
        "max_tokens_base": 64,
        "max_tokens_per_label": 20,
        "max_tokens_formula": "max_tokens_base + max_tokens_per_label * n_requested_labels",
    },
    "serving": {
        "engine": "vllm",
        "engine_version_pin": "vllm==0.28.0",
        "engine_wheel_pin": "vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl",
        "engine_pin_source": "https://pypi.org/pypi/vllm/0.28.0/json",
        "engine_pin_retrieved_utc": "2026-09-11",
        "engine_pin_evidence": "pins/engine_pin.json in this freeze directory",
        "engine_pin_rule": (
            "Step 1 installs exactly this version and records the installed version string. "
            "A different engine or version voids this freeze and requires a new freeze."
        ),
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
        "max_num_seqs": 64,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "hardware": "one NVIDIA A100, one at a time",
        "recorded_from_the_step_1_host": [
            "cuda_version",
            "driver_version",
            "gpu_name",
            "gpu_uuid",
            "torch_version",
            "installed_engine_version_string",
        ],
        "note": (
            "the serving engine and its version are pinned here, before any output exists. "
            "Only the host facts listed in recorded_from_the_step_1_host are written at Step 1, "
            "because they are properties of the machine and not design choices. No other "
            "protocol field may be completed after this freeze."
        ),
    },
    "cells": [
        {
            "cell_id": "qwen_civil",
            "model_id": "Qwen/Qwen3-8B",
            "task_id": "civil_comments",
            "role": "new-record replication anchor",
            "tier": "replication",
            "in_confirmation_set": False,
            "repeated_sampling": True,
        },
        {
            "cell_id": "qwen_secondtask",
            "model_id": "Qwen/Qwen3-8B",
            "task_id": "second_task",
            "role": "fresh confirmatory cell, new task axis",
            "tier": "confirmation",
            "in_confirmation_set": True,
            "repeated_sampling": False,
        },
        {
            "cell_id": "mistral_civil",
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "task_id": "civil_comments",
            "role": "fresh confirmatory cell, new model axis",
            "tier": "confirmation",
            "in_confirmation_set": True,
            "repeated_sampling": False,
        },
        {
            "cell_id": "mistral_secondtask",
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "task_id": "second_task",
            "role": "double-new cell, both axes fresh",
            "tier": "confirmation",
            "in_confirmation_set": True,
            "repeated_sampling": True,
        },
    ],
}

MODEL_ACCESS_EVIDENCE = (
    "the license string, the gated flag, the private flag, the disabled flag, and the live main "
    "commit sha are retrieved from the HuggingFace API and stored in pins/model_pins.json. "
    "This freeze asserts no access or redistribution right that is not in that record. "
    "The study downloads weights for inference and redistributes no weights and no dataset text."
)

MODELS: dict[str, Any] = {
    "Qwen/Qwen3-8B": {
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "revision_source": "https://huggingface.co/api/models/Qwen/Qwen3-8B",
        "revision_retrieved_utc": "2026-09-11",
        "license": "apache-2.0",
        "tokenizer_repo": "Qwen/Qwen3-8B",
        "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "chat_template_source": "tokenizer_config.json of the pinned revision",
        "enable_thinking": False,
        "same_revision_as_pilot": True,
        "role": "pilot model, reused so the anchor is a record replication only",
        "access_evidence": MODEL_ACCESS_EVIDENCE,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
        "revision_source": (
            "https://huggingface.co/api/models/mistralai/Mistral-7B-Instruct-v0.3"
        ),
        "revision_retrieved_utc": "2026-09-11",
        "license": "apache-2.0",
        "tokenizer_repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "tokenizer_revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
        "chat_template_source": "tokenizer_config.json of the pinned revision",
        "enable_thinking": False,
        "same_revision_as_pilot": False,
        "role": "second model, never seen by the pilot",
        "access_evidence": MODEL_ACCESS_EVIDENCE,
    },
    "allenai/OLMo-2-1124-7B-Instruct": {
        "revision": "470b1fba1ae01581f270116362ee4aa1b97f4c84",
        "revision_source": (
            "https://huggingface.co/api/models/allenai/OLMo-2-1124-7B-Instruct"
        ),
        "revision_retrieved_utc": "2026-09-11",
        "license": "apache-2.0",
        "tokenizer_repo": "allenai/OLMo-2-1124-7B-Instruct",
        "tokenizer_revision": "470b1fba1ae01581f270116362ee4aa1b97f4c84",
        "chat_template_source": "tokenizer_config.json of the pinned revision",
        "enable_thinking": False,
        "same_revision_as_pilot": False,
        "role": "first replacement candidate for the second-model axis",
        "access_evidence": MODEL_ACCESS_EVIDENCE,
    },
    "microsoft/Phi-3.5-mini-instruct": {
        "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
        "revision_source": (
            "https://huggingface.co/api/models/microsoft/Phi-3.5-mini-instruct"
        ),
        "revision_retrieved_utc": "2026-09-11",
        "license": "mit",
        "tokenizer_repo": "microsoft/Phi-3.5-mini-instruct",
        "tokenizer_revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
        "chat_template_source": "tokenizer_config.json of the pinned revision",
        "enable_thinking": False,
        "same_revision_as_pilot": False,
        "role": "second replacement candidate for the second-model axis",
        "access_evidence": MODEL_ACCESS_EVIDENCE,
    },
}

SECOND_MODEL_REPLACEMENT_POLICY: dict[str, Any] = {
    "frozen_before_any_joint_output": True,
    "primary_second_model": "mistralai/Mistral-7B-Instruct-v0.3",
    "ordered_replacement_list": [
        "allenai/OLMo-2-1124-7B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
    ],
    "trigger_conditions_all_solo_only": [
        "the pinned revision cannot be loaded on the A100 under the frozen serving settings",
        "any frozen prompt exceeds max_model_len under the pinned tokenizer",
        "solo-only parsable first-response rate is below 0.98",
        "solo-only accuracy is below the task capability floor",
    ],
    "information_poor_is_not_a_replacement_trigger": (
        "a solo error count below the frozen information threshold leaves the model in the "
        "panel as INFORMATION_POOR. It may activate the separately frozen task fallback, but "
        "it never activates model replacement."
    ),
    "forbidden_triggers": [
        "any joint, exact-context, fake-batch, or filler outcome",
        "any observed covariance, ICC, variance ratio, or coverage value",
        "convenience, cost, or speed after outputs are opened",
    ],
    "replacement_limit": 1,
    "if_no_replacement_passes": (
        "keep the failed cell in the published panel labelled uninformative, narrow the claim "
        "to the tested model, and do not write 'across models'."
    ),
    "gated_repo_check": (
        "the live gated, private, and disabled flags of every pinned model repo are retrieved "
        "and stored in pins/model_pins.json. No repo is described as ungated without that "
        "record. A gated, private, or disabled repo is a hard blocker."
    ),
}

TASKS: dict[str, Any] = {
    "civil_comments": {
        "status": "frozen",
        "dataset": "google/civil_comments",
        "revision": "f2970eb3a55777454c94069077cc8d9b5866312d",
        "revision_source": "https://huggingface.co/api/datasets/google/civil_comments",
        "revision_retrieved_utc": "2026-09-11",
        "license": "cc0-1.0",
        "redistribution_permitted": True,
        "split": "validation",
        "split_rows": 97320,
        "text_field": "text",
        "label_field": "toxicity",
        "n_classes": 2,
        "class_names": {"0": "not toxic", "1": "toxic"},
        "reference_label_rule": "toxicity >= 0.5 is reference class 1",
        "outcome_name": "disagreement with the reference label",
        "outcome_is_not": "proof of human annotation error",
        "sampling": "natural prevalence, no class balancing",
        "sampling_rationale": (
            "the estimand is dependence under a deployed annotation policy, and such a "
            "pipeline labels records as they arrive. Class balancing would change the "
            "marginal error rate and therefore the covariance itself."
        ),
        "n_items": 2048,
        "filler_pool_size": 4096,
        "min_chars": 40,
        "max_chars": 1000,
        "scan_limit": 97320,
        "selection_seed": 20260912,
        "prefix": "cc",
        "excluded_id_sources": [
            "gate-experiments/a/artifacts/full/data/targets.jsonl",
            "gate-experiments/a/artifacts/full/data/filler_pool.jsonl",
        ],
        "excluded_id_counts_expected": {"targets": 2000, "filler_pool": 4000},
        "exclusion_rationale": (
            "the 2,000 pilot targets are mandatory exclusions. The 4,000 pilot filler "
            "records were also shown to the same model inside pilot prompts, so they are "
            "excluded as well. This is stricter than the plan requires."
        ),
        "labelling_limits": (
            "Civil Comments toxicity is a crowd-rated continuous score thresholded at 0.5. "
            "It carries rater disagreement and documented identity-term bias. Treat it as a "
            "reference label, never as ground truth."
        ),
        "handling": (
            "researcher exposure to abusive text is minimised; no text is printed in "
            "reports"
        ),
    },
    "second_task_first_candidate": {
        "status": "frozen as first candidate; final choice is a Step 1 solo-only decision",
        "task_id": "ag_news",
        "dataset": "fancyzhx/ag_news",
        "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
        "revision_source": "https://huggingface.co/api/datasets/fancyzhx/ag_news",
        "revision_retrieved_utc": "2026-09-11",
        "license": "unknown (HuggingFace card reports license:unknown)",
        "redistribution_permitted": False,
        "license_finding": (
            "the AG News card declares license:unknown and the corpus description states it "
            "is provided by the academic community for research purposes. Research use is "
            "supported. Redistribution of article text is not. The release therefore publishes "
            "document ids, row indices, character counts, and SHA-256 digests only, never text."
        ),
        "split": "train",
        "split_rows": 120000,
        "text_field": "text",
        "label_field": "label",
        "n_classes": 4,
        "class_names": {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"},
        "reference_label_rule": "the dataset topic label is the reference class",
        "outcome_name": "disagreement with the reference topic label",
        "sampling": "stratified, 512 records per class",
        "sampling_rationale": (
            "the AG News train split is already exactly balanced at 30,000 rows per class, so "
            "512 per class reproduces the natural composition of the declared frame and removes "
            "sampling noise in the class margins."
        ),
        "n_items": 2048,
        "filler_pool_size": 4096,
        "min_chars": 100,
        "max_chars": 1000,
        "scan_limit": 120000,
        "selection_seed": 20260912,
        "prefix": "ag",
        "excluded_id_sources": [],
        "labelling_limits": (
            "AG News topic labels come from the source news categories, not from adjudicated "
            "annotation. Business and Sci/Tech overlap for some records. Treat the label as a "
            "reference class."
        ),
    },
    "second_task_conditional_fallback": {
        "status": (
            "conditional fallback; the frame is materialised and frozen at Step 0 so that the "
            "fallback cannot be designed after a screen outcome is known"
        ),
        "task_id": "multi_nli",
        "dataset": "nyu-mll/multi_nli",
        "revision": "da70db2af9d09693783c3320c4249840212ee221",
        "revision_source": "https://huggingface.co/api/datasets/nyu-mll/multi_nli",
        "revision_retrieved_utc": "2026-09-11",
        "license": "cc-by-3.0, cc-by-sa-3.0, mit, other (mixed, per source genre)",
        "redistribution_permitted": False,
        "license_finding": (
            "the MultiNLI card declares four licenses that vary by source genre. Research use is "
            "supported. A single redistribution right for the whole corpus is not established, so "
            "the release publishes document ids, row indices, character counts, and SHA-256 "
            "digests only, never text."
        ),
        "split": "validation_matched",
        "split_rows": 9815,
        "text_fields": ["premise", "hypothesis"],
        "text_template": "premise: {premise}\nhypothesis: {hypothesis}",
        "text_field": "premise",
        "label_field": "label",
        "n_classes": 3,
        "class_names": {"0": "entailment", "1": "neutral", "2": "contradiction"},
        "reference_label_rule": "the dataset entailment label is the reference class",
        "outcome_name": "disagreement with the reference entailment label",
        "outcome_is_not": "proof of human annotation error",
        "sampling": "natural prevalence, no class balancing",
        "sampling_rationale": (
            "2,048 items do not divide by three classes, and the estimand is dependence under a "
            "deployed annotation policy that labels records as they arrive. Natural prevalence "
            "keeps the marginal error rate, and therefore the covariance, comparable with the "
            "other frames."
        ),
        "min_chars": 40,
        "max_chars": 1000,
        "scan_limit": 9815,
        "n_items": 2048,
        "filler_pool_size": 4096,
        "selection_seed": 20260912,
        "prefix": "mnli",
        "excluded_id_sources": [],
        "frame_size_check": (
            "2,048 items plus a 4,096 record filler pool need 6,144 candidates from a split of "
            "9,815 rows. The materialised provenance records the realised candidate count."
        ),
        "activation_rule": (
            "activated if AG News fails the frozen solo-only screen for either selected panel "
            "model. "
            "MultiNLI then runs the identical solo-only screen. A failure of both tasks narrows "
            "the claim to Civil Comments and removes the task axis."
        ),
        "labelling_limits": (
            "MultiNLI has known annotation artefacts and hypothesis-only predictability. Its "
            "mixed per-genre licensing blocks redistribution of text."
        ),
    },
}

SOLO_SCREEN: dict[str, Any] = {
    "frozen_before_any_joint_output": True,
    "inputs_allowed": ["solo calls only", "tokenizer token counts", "parse status"],
    "inputs_forbidden": [
        "joint outputs",
        "exact-context outputs",
        "fake-batch statistics",
        "filler outputs",
        "any covariance, ICC, variance, or coverage value",
    ],
    "screen_sample": {
        "n_items": 2048,
        "source": "the frozen item frame of the task under screen",
        "note": "the screen uses the same frozen items, so no extra sampling frame is created",
    },
    "criteria": {
        "context_fit": {
            "rule": "every frozen joint, exact-context, solo, and filler prompt must be at most "
            "max_model_len minus the frozen max_tokens budget under the pinned tokenizer",
            "threshold": "0 prompts over budget",
        },
        "parsable_first_response_rate": {
            "rule": "parsable solo first responses divided by solo calls",
            "threshold": 0.98,
            "comparison": ">=",
        },
        "capability_floor": {
            "civil_comments": 0.70,
            "ag_news": 0.70,
            "multi_nli": 0.55,
            "rule": "solo accuracy against the reference label must be at or above the floor",
            "rationale": (
                "the floor rejects a model that cannot do the task, while staying far below the "
                "pilot's 0.832 solo accuracy so the floor cannot be reverse-engineered from a "
                "favourable result. The MultiNLI floor is lower because three-class entailment "
                "is harder for a small instruct model."
            ),
        },
        "error_information_threshold": {
            "rule": "solo error count out of 2,048 items must be at or above the threshold",
            "min_solo_errors": 11,
            "min_solo_error_rate": 0.005,
            "source": "simulations/results.json key information_threshold",
            "criterion": (
                "detection power: the probability that the frozen randomization test rejects at "
                "the three-cell Holm-adjusted alpha when the true excess covariance equals the "
                "frozen 0.002 floor. The separate claim gate, which also requires the point "
                "estimate to be at or above the floor, is reported but is not the threshold, "
                "because an unbiased estimator clears a boundary equal to the truth about half "
                "the time by construction."
            ),
            "rationale": (
                "the pair-weighted covariance statistic is bounded by p(1-p), so a very accurate "
                "cell cannot reach the 0.002 absolute floor at all. The threshold is the smallest "
                "prespecified grid rate at which the fixed 2,048-item, k=16, three-assignment "
                "design reaches the frozen detection-power target for the frozen floor."
            ),
            "high_error_rate_finding": {
                "highest_grid_error_rate_meeting_target": 0.08,
                "error_rates_above_the_band_that_miss_the_target": [
                    0.1,
                    0.113,
                    0.15,
                    0.2,
                    0.3,
                ],
            },
        },
        "accuracy_ceiling": {
            "rule": "solo accuracy at or above this value marks the cell information-poor",
            "value": 0.995,
            "note": "this is the same constraint as the error-information threshold, restated",
        },
    },
    "outcome_labels": [
        "PASS",
        "FAIL_CONTEXT",
        "FAIL_PARSE",
        "FAIL_CAPABILITY",
        "INFORMATION_POOR",
    ],
    "information_poor_handling": (
        "an INFORMATION_POOR cell stays in the published panel with uncertainty bounds and the "
        "label 'insufficient events for a precise covariance estimate'. It is not replaced and "
        "its joint outputs are still run and reported."
    ),
    "task_fallback_rule": {
        "order": ["ag_news", "multi_nli"],
        "decision_input": "solo-only screen outcome for both panel models",
        "rule": (
            "choose AG News if it passes the screen for both models. Otherwise screen MultiNLI. "
            "Choose MultiNLI only if it passes for both models. If neither task passes for both "
            "models, keep AG News, publish the failing cell as uninformative, and narrow the "
            "claim to the tested task."
        ),
        "forbidden": (
            "choosing the task by comparing batching effects, covariance, or any joint "
            "result"
        ),
    },
}

# --------------------------------------------------------------------------------------
# Prompts and response grammars
# --------------------------------------------------------------------------------------

PROMPTS: dict[str, Any] = {
    "shared": {
        "document_wrapper": '<doc id="{doc_id}">\\n{text}\\n</doc>',
        "flat_render_for_hashing": "role: content, joined by a blank line",
        "single_target_instruction": (
            "Label only the document with id \"{target_doc_id}\". Ignore every other document. "
            "Return no other id."
        ),
        "single_target_response_shape": (
            "{\"labels\": {\"<target_doc_id>\": <one permitted class integer>}}"
        ),
        "single_target_applies_to": ["exact_context", "solo", "filler", "serial_exact"],
        "duplicate_document_in_one_prompt": "forbidden; the plan builder raises",
        "templates_per_task": 1,
        "template_replication": "out of scope for this bounded paper (plan.md F9 defers it)",
    },
    "civil_comments": {
        "system": (
            "You are a careful content moderation annotator. "
            "You answer with one JSON object and nothing else."
        ),
        "definition": (
            "A document is toxic if a reasonable reader would call it rude, disrespectful, "
            "or likely to make someone leave the conversation. Use 1 for toxic and 0 for not toxic."
        ),
        "joint_instruction": "Label each of the {n} documents below for toxicity.",
        "single_instruction": "Label one document for toxicity.",
        "response_grammar": '{"labels": {"<doc_id>": <int in {0,1}>}}',
        "identical_to_pilot": True,
        "identical_to_pilot_note": (
            "the anchor cell replicates the pilot on fresh records, so its prompt text, "
            "definition, and grammar are byte-identical to "
            "gate-experiments/a/src/batchgate/prompts.py at the pilot revision."
        ),
    },
    "ag_news": {
        "system": (
            "You are a careful news topic annotator. "
            "You answer with one JSON object and nothing else."
        ),
        "definition": (
            "Assign each document to exactly one topic. Use 0 for World, 1 for Sports, "
            "2 for Business and 3 for Sci/Tech."
        ),
        "joint_instruction": "Label each of the {n} documents below for topic.",
        "single_instruction": "Label one document for topic.",
        "response_grammar": '{"labels": {"<doc_id>": <int in {0,1,2,3}>}}',
        "parallel_to_civil": True,
    },
    "multi_nli": {
        "system": (
            "You are a careful natural language inference annotator. "
            "You answer with one JSON object and nothing else."
        ),
        "definition": (
            "For each document, decide whether the hypothesis follows from the premise. "
            "Use 0 for entailment, 1 for neutral and 2 for contradiction."
        ),
        "joint_instruction": "Label each of the {n} premise and hypothesis pairs below.",
        "single_instruction": "Label one premise and hypothesis pair.",
        "response_grammar": '{"labels": {"<doc_id>": <int in {0,1,2}>}}',
    },
    "parser": {
        "implementation_reference": "gate-experiments/a/src/batchgate/prompts.py:parse_labels",
        "strictness": "malformed output is a parse failure",
        "only_permitted_repair": "removal of a single Markdown code fence, recorded in notes",
        "duplicate_json_keys": "fail the whole call; the intended mapping is unrecoverable",
        "string_digit_labels": "accepted, recorded as note string_label",
        "boolean_labels": "rejected as bad_value",
        "unknown_ids": "recorded, never scored",
        "statuses": [
            "ok",
            "no_json",
            "json_error",
            "not_object",
            "missing_labels",
            "duplicate_ids",
            "bad_value",
            "missing_id",
        ],
        "extension_for_multiclass": (
            "the permitted integer set widens to the task's class set. No other change."
        ),
    },
}

# --------------------------------------------------------------------------------------
# Conditions, call lineage, execution
# --------------------------------------------------------------------------------------

CONDITIONS: dict[str, Any] = {
    "joint": {
        "definition": "one call shows 16 records and requests 16 labels",
        "calls_per_cell": 384,
        "calls_per_assignment": 128,
        "role": "primary treatment arm",
        "cells": "all four",
    },
    "exact_context": {
        "definition": (
            "one call shows the same 16 records in the same slot order and requests exactly "
            "one target label. Repeated 16 times per joint call, once per target."
        ),
        "calls_per_cell": 6144,
        "role": "primary control arm; holds context, peers, position, and length fixed",
        "cells": "all four",
        "trim_policy": "never trimmed; the primary contrast depends on it",
    },
    "solo": {
        "definition": "one call shows one record and requests one label",
        "calls_per_cell": 2048,
        "role": "accuracy and formatting comparator, and the only screen input",
        "cells": "all four",
        "assignment_id": 0,
    },
    "fake_batch": {
        "definition": (
            "no new calls. Independently produced exact-context labels are regrouped with the "
            "frozen assignment generator, then the identical statistic is recomputed."
        ),
        "calls_per_cell": 0,
        "role": "falsification endpoint",
        "cells": "all four",
        "monte_carlo_regroupings": 2000,
    },
    "filler": {
        "definition": (
            "one call shows the target plus 15 length-matched records drawn from the disjoint "
            "filler pool, and requests only the target label"
        ),
        "role": "pilot context-length placebo",
        "length_match_tolerance": 0.05,
        "filler_candidate_draw": 64,
        "peer_selection": (
            "the same greedy nearest-fit rule as the pilot: the target's 15 peers are drawn "
            "without replacement from the disjoint filler pool so that the total peer character "
            "count approaches the total character count of the 15 real peers of the same joint "
            "call. Each step samples filler_candidate_draw candidates with a seeded generator "
            "and takes the candidate closest to the remaining ideal length."
        ),
        "peer_selection_reference": "gate-experiments/a/src/batchgate/plan.py:choose_filler",
        "target_slot_rule": "the target keeps the slot it held in the matched joint call",
        "cells": {
            "qwen_civil": {"assignments": 3, "calls": 6144, "note": "full pilot repeat"},
            "mistral_secondtask": {
                "assignments": 1,
                "calls": 512,
                "subset_rule": "the first 32 calls of assignment 1 in frozen call order",
                "note": "prespecified small double-new subset",
            },
        },
    },
    "serial_serving": {
        "definition": (
            "the same joint and exact-context calls re-executed with serving concurrency 1, so "
            "no other request shares the engine step"
        ),
        "cells": {
            "mistral_secondtask": {
                "joint_calls": 32,
                "exact_context_calls": 512,
                "subset_rule": "the first 32 calls of assignment 1 in frozen call order",
            }
        },
        "role": "tests whether scheduler co-batching, not the shared prompt, creates covariance",
        "engine_setting": "max_num_seqs = 1",
    },
    "interleaving": {
        "rule": "arms are interleaved over execution time, never run arm by arm",
        "n_execution_blocks_per_cell": 8,
        "block_assignment": "execution_block = call_index modulo 8, computed from the frozen plan",
        "within_block_order": (
            "calls are ordered by the execution_order_key stored in each job skeleton: sha256 of "
            "the canonical JSON object containing namespace, purpose, cell_id, execution_block, "
            "and call_id; every arm appears in every block"
        ),
        "fresh_serving_session_per_block": True,
        "rationale": (
            "partial interference is assumed at the annotation-call level after conditioning on "
            "the execution block, so blocks must mix arms or the assumption is untestable."
        ),
    },
}

CALL_LINEAGE_SCHEMA: dict[str, Any] = {
    "one_row_per_requested_label": True,
    "fields": {
        "call_id": "globally unique sha256-derived id of the frozen call specification",
        "parent_call_id": "null for a first response; the retried call id for a retry",
        "retry_index": "0 for the first response, then 1, 2, ...",
        "first_response_status": "the frozen state machine value of the first response",
        "cell_id": "model and task cell",
        "model_id": "pinned model repo id",
        "model_revision": "pinned commit sha",
        "task_id": "pinned task id",
        "dataset_revision": "pinned dataset commit sha",
        "condition": "joint | exact_context | solo | filler | serial_joint | serial_exact",
        "assignment": "0 for solo, else 1, 2, 3",
        "call_index": "index of the call inside its assignment",
        "batch_id": "globally unique annotation-call id, the primary cluster",
        "doc_order": "the 16 document ids in prompt order",
        "doc_id": "the requested target for this row",
        "slot": "0-based output position of the target inside the call",
        "length_stratum": "0-15 length stratum of the target",
        "execution_block": "0-7 randomised execution block",
        "serving_session": "id of the engine process that served the call",
        "concurrency_window": (
            "observable engine batch or scheduler step id when exposed, else "
            "null"
        ),
        "prompt_sha256": "hash of the flat-rendered prompt",
        "prompt_tokens": "token count under the pinned tokenizer",
        "output_tokens": "generated token count",
        "max_tokens": "the frozen output cap for this call",
        "latency_seconds": "wall clock for the call",
        "gpu_seconds": "metered A100 seconds attributed to the call",
        "raw_text": "unedited model output",
        "output_sha256": "hash of the unedited output",
        "parse_status": "parser state for this row",
        "parse_notes": "list of parser notes",
        "pred": "parsed predicted class or null",
        "reference": "the reference class",
        "error": "1 when pred differs from reference, else 0, null when unparsed",
        "created_at": "UTC timestamp",
        "run_id": "run invocation id",
        "engine_version": "serving engine version string",
    },
    "immutability": "a row is appended once and never edited",
}

FAILURE_POLICY: dict[str, Any] = {
    "state_machine": [
        "valid_first_response",
        "malformed",
        "truncated",
        "refused",
        "missing",
        "retry_valid",
    ],
    "first_response_policy": (
        "every confirmatory endpoint uses first responses only. A retry never replaces a first "
        "response and never enters a confirmatory statistic."
    ),
    "retry_policy": {
        "max_retries": 1,
        "retry_is": "a new call with a new call_id and parent_call_id set to the failed call",
        "retry_trigger": "an infrastructure error or a truncated response only",
        "retry_use": "descriptive reporting and a clearly labelled secondary sensitivity analysis",
        "never": "mixing first responses and retries inside one endpoint",
    },
    "truncation": {
        "cap": "max_tokens_base 64 plus 20 per requested label",
        "pilot_lesson": (
            "the pilot's first joint attempt used a 256-token cap and truncated every 16-label "
            "response. Exact pinned-tokenizer counts showed at most 310 tokens were needed, so "
            "the cap became 384. The 384-token cap for 16 labels is carried forward."
        ),
        "preflight": "the frozen plan is token-checked with the pinned tokenizer before any call",
    },
    "exclusion_rules": {
        "unit": "the requested label row",
        "excluded_from_the_primary_statistic": [
            "rows whose first response is not parsable",
            "rows whose target id is missing from an otherwise parsable response",
        ],
        "never_excluded": [
            "a call because its outcome is unfavourable",
            "a cell because it has few errors",
            "an assignment because its estimate is small or negative",
            "an item because it is hard or easy",
        ],
        "incomplete_calls": (
            "a call with fewer than 2 parsable rows contributes no pair and drops out of the "
            "pair-weighted sum by construction. Its row count is reported."
        ),
        "clustered_failure_check": (
            "parse-failure clustering by call is tested with the same pair-weighted statistic on "
            "the failure indicator. Clustered failures make the drop rule non-ignorable, and then "
            "the whole-call failure bound becomes the headline sensitivity."
        ),
        "mandatory_bounds": [
            "all unresolved labels counted as errors",
            "all unresolved labels counted as correct",
            "unresolved labels treated as a separate failure outcome",
            "the whole call discarded when any of its labels is unresolved",
        ],
        "parse_failure_gate": {
            "max_parse_failure_rate_per_condition": 0.02,
            "action_if_exceeded": (
                "the cell is reported as uninformative for the primary contrast and only the "
                "bounds are reported"
            ),
        },
    },
}

# --------------------------------------------------------------------------------------
# Estimands and statistics
# --------------------------------------------------------------------------------------

ESTIMANDS: dict[str, Any] = {
    "E1_primary_dependence": {
        "statistic": "signed pair-weighted excess within-call error covariance",
        "formula": (
            "C_{c,z} = sum_b sum_{i != j in b} (E_i - Ebar_z)(E_j - Ebar_z) "
            "/ sum_b |b|(|b|-1)"
        ),
        "contrast": "Delta_exact = C_joint - C_exact",
        "sign_convention": "positive means more same-call co-error than the matched control",
        "centering": "each condition is centred on its own condition mean Ebar_z",
        "centering_rationale": (
            "centring on the condition's own mean keeps the contrast from being driven by the "
            "marginal accuracy difference between arms, which is a separate endpoint."
        ),
        "grouping_joint": "the real annotation call",
        "grouping_control": "the matched target set of the same frozen assignment matrix",
        "per_cell_aggregation": "unweighted mean of the three assignment-level contrasts",
        "per_cell_aggregation_rationale": (
            "each item appears once per assignment, so the three assignment estimates use "
            "disjoint rows but the same items. Pooling into one sum would count each item three "
            "times and understate the standard error."
        ),
        "is_primary": True,
    },
    "E1b_falsification": {
        "statistic": "Delta_perm = C_joint - C_fakebatch",
        "construction": (
            "exact-context labels, which were produced in independent calls, are regrouped under "
            "the frozen assignment generator"
        ),
        "expectation_under_the_null_of_no_call_effect": "approximately zero",
        "role": "prespecified falsification endpoint, not a second primary test",
        "monte_carlo_regroupings": 2000,
    },
    "E4_accuracy": {
        "statistic": "paired per-item change in error probability between arms",
        "pairs": "the same item in joint and in exact-context, matched within assignment",
        "reported_separately_from": "E1 and E1b",
        "rule": (
            "marginal accuracy and covariance are separate endpoints. A joint accuracy gain, "
            "which the pilot observed, neither supports nor refutes a covariance claim."
        ),
        "interval": "document-level bootstrap, 2000 resamples, alpha 0.05",
    },
    "E5_inferential_consequence": {
        "target": "mean reference-relative disagreement over the frozen item frame",
        "target_definition": (
            "expected mean disagreement over independent uniform 512-item samples drawn without "
            "replacement within a replicate from the frozen 2,048-item frame, under random "
            "size-16 partitions produced by the exact operational batching policy"
        ),
        "is_not": "gold-label population prevalence",
        "cells": ["qwen_civil", "mistral_secondtask"],
        "cells_frozen_before_outputs": True,
        "cell_selection_forbidden_input": "any observed covariance in any cell",
    },
    "out_of_scope_for_this_paper": [
        "E2 output-label covariance as a co-primary endpoint",
        "E3 conditional covariance from a crossed hierarchical model",
        "E6 peer-composition effects",
        "E7 output-history effects",
        "Type-I error under a generated null",
        "regression coefficients, model rankings, population prevalence",
        "the balanced re-batching mechanism design of plan.md section 7.3",
        "any cost-optimality claim",
    ],
}

TESTS: dict[str, Any] = {
    "primary_contrast": {
        "name": "exact-context primary contrast",
        "statistic": "Delta_exact per cell",
        "test": "design-respecting paired randomization test",
        "null": (
            "the sharp null that within each arm and execution block every item's output would "
            "be unchanged by any allowable reassignment of calls, peers, or positions. Different "
            "fixed outputs between arms, including a marginal accuracy shift, are permitted."
        ),
        "replay": (
            "the identical frozen assignment generator is replayed in both arms with the same "
            "draw, keeping length strata, one-item-per-stratum call composition, slot constraints, "
            "batch sizes, repeated-item structure, and execution blocks"
        ),
        "forbidden": "unrestricted shuffling of labels across items or arms",
        "monte_carlo_draws": 2000,
        "monte_carlo_draws_rationale": (
            "the count study reports Type-I rates at [499, 999, 1999] draws without using those "
            "rates to tune the production count. At 2000 draws the smallest reachable p is "
            "0.000500, far below the three-cell Holm threshold of 0.0167."
        ),
        "tail": "one-sided upper",
        "p_value_formula": "(1 + #{draws with T_draw >= T_obs}) / (1 + n_draws)",
        "alpha": 0.05,
        "effect_interval": (
            "design-respecting interval from the replay distribution: T_obs minus the alpha/2 and "
            "1-alpha/2 quantiles of the mean-centred replay distribution, reported as a shift "
            "interval, not as a bootstrap interval"
        ),
    },
    "fake_batch_falsification": {
        "statistic": "Delta_perm per cell",
        "pass_rule": (
            "the fake-batch contrast must be positive and its one-sided p must be below 0.05, "
            "while the fake-batch covariance itself stays inside the bounded null band"
        ),
        "bounded_null_band_high": 0.001,
        "failure_meaning": (
            "if regrouped independent labels reproduce the joint covariance, the statistic is "
            "detecting the grouping rule and not the shared call. Then no dependence claim is made."
        ),
    },
    "marginal_accuracy_separation": {
        "rule": (
            "the covariance endpoint and the accuracy endpoint are reported in different tables "
            "and never combined into one verdict"
        ),
        "check": (
            "the centred statistic is recomputed after forcing both arms to a common marginal "
            "rate by centring each arm on its own mean. A contrast that vanishes under that "
            "recomputation was a marginal-rate artefact."
        ),
    },
    "practical_floors": {
        "covariance_floor_absolute": 0.002,
        "covariance_floor_direction": "WP0 simulation may raise it, never lower it",
        "covariance_floor_relative": "not used; the control covariance can be zero or change sign",
        "variance_ratio_floor": 1.25,
        "variance_ratio_meaning": (
            "at least 20 percent effective-sample-size loss for the same mean "
            "estimand"
        ),
        "independence_of_floors": (
            "meeting the covariance floor does not imply meeting the variance-ratio floor or the "
            "coverage gate. They are separate claims about separate estimands."
        ),
    },
    "multiplicity": {
        "family_1_cell_contrasts": {
            "members": ["qwen_secondtask", "mistral_civil", "mistral_secondtask"],
            "method": "Holm, family-wise alpha 0.05",
            "gatekeeping": (
                "a cell claim also needs its effect size above the floor and its controls to "
                "pass"
            ),
        },
        "family_2_recurrence": {
            "members": ["equal-cell-weight mean contrast over the three confirmation cells"],
            "method": "single test at alpha 0.05, tested before the per-cell family",
        },
        "family_3_repeated_sampling": {
            "members": [
                "qwen_civil variance ratio",
                "qwen_civil coverage",
                "mistral_secondtask variance ratio",
                "mistral_secondtask coverage",
            ],
            "method": "Holm, family-wise alpha 0.05",
        },
        "anchor_cell": {
            "members": ["qwen_civil"],
            "method": "reported separately as a replication, never pooled into family 1 or 2",
        },
        "excluded_from_every_family": [
            "the old Gate A pilot result",
            "descriptive ICC, design effect, effective sample size",
            "position and slot diagnostics",
            "exploratory interactions",
        ],
        "fdr": "not used in any confirmatory family",
    },
    "recurrence_gate": {
        "statistic": "equal-cell-weight mean of Delta_exact over the three confirmation cells",
        "weighting": "equal per cell, never weighted by item count or precision",
        "weighting_rationale": (
            "the three cells are purposively selected panel members, not a random sample of "
            "models and tasks. Precision weighting would let one cell speak for the panel."
        ),
        "pass_conditions_all_required": [
            "the equal-cell-weight mean contrast is positive",
            "the equal-cell-weight mean contrast is at least 0.002",
            "at least 2 of the 3 confirmation cells have a positive contrast of at least 0.002 "
            "with a Holm-adjusted p below 0.05",
            "the double-new cell mistral_secondtask is one of those passing cells",
            "every passing cell also passes its fake-batch falsification",
        ],
        "fail_meaning": (
            "a positive result only in the anchor is a bounded case study on one model-task cell, "
            "not a panel result, and the paper must say so"
        ),
    },
    "parse_failure_bounds": {
        "rule": "the primary contrast is recomputed under all four mandatory imputations",
        "claim_condition": (
            "a dependence claim requires the contrast to stay positive and above the floor under "
            "the errors-as-correct and errors-as-error bounds. The whole-call-failure bound is "
            "reported and, when parse failures cluster by call, becomes the headline bound."
        ),
    },
}

REPEATED_SAMPLING: dict[str, Any] = {
    "cells": ["qwen_civil", "mistral_secondtask"],
    "frozen_before_outputs": True,
    "target": ESTIMANDS["E5_inferential_consequence"]["target_definition"],
    "replicate_design": {
        "items_per_replicate": 512,
        "sampling_within_replicate": "uniform without replacement from the frozen 2,048-item frame",
        "calls_per_replicate": 32,
        "partition": "random size-16 partition under the exact operational batching policy",
        "item_recurrence_across_replicates": (
            "permitted; each replicate is a new draw from the fixed frame and every call is fresh"
        ),
        "output_reuse_across_replicates": "forbidden",
    },
    "reference_stream": {
        "replicates": 250,
        "purpose": "estimate the target mean without using any evaluation replicate",
        "disjoint_random_stream": True,
        "seed_namespace": "reference",
        "precision_check": (
            "the target estimate's Monte Carlo standard error must be small relative to the "
            "average evaluation interval half-width. The frozen rule is: reference SE must be at "
            "most one quarter of the mean nominal i.i.d. half-width."
        ),
        "if_too_imprecise": (
            "no measured-coverage claim is made. Report the empirical variance of replicate means "
            "and its ratio to the average nominal i.i.d. variance instead."
        ),
    },
    "evaluation_stream": {
        "replicates_minimum": 400,
        "replicates_maximum": 1000,
        "maximum_fixed_by": (
            "the Step 1 A100 timing benchmark only, before any evaluation outcome is "
            "opened"
        ),
        "disjoint_random_stream": True,
        "seed_namespace": "evaluation",
        "monte_carlo_se_at_minimum": "about 1.1 percentage points near 95 percent coverage",
    },
    "interval_methods": {
        "iid": {
            "name": "item-level 95 percent Wald interval",
            "formula": "phat +- 1.959964 * sqrt(phat(1-phat)/n) with n = 512 item appearances",
            "finite_population_correction": False,
        },
        "call_aware": {
            "name": "95 percent t interval over the 32 call means",
            "degrees_of_freedom": 31,
            "critical_value": "Student t 0.975 quantile at 31 df",
            "finite_population_correction": False,
        },
        "finite_population_correction_decision": (
            "frozen off for both methods. The 512-of-2,048 sample would give a factor of "
            "sqrt(1-0.25) = 0.866, which narrows the i.i.d. interval and therefore makes an "
            "undercoverage claim easier. Leaving it off is the conservative choice. The corrected "
            "version is reported as a labelled sensitivity analysis only."
        ),
        "sensitivity": {
            "whole_call_bootstrap": {
                "resamples": 2000,
                "unit": "the whole call mean, resampled with replacement",
                "role": "sensitivity for the call-aware interval, never the primary method",
            }
        },
    },
    "endpoints": {
        "variance_ratio": {
            "statistic": (
                "empirical variance of replicate means divided by mean nominal i.i.d. "
                "variance"
            ),
            "floor": 1.25,
            "interval": "bootstrap over replicates, 2000 resamples",
        },
        "coverage": {
            "statistic": "fraction of replicates whose interval contains the reference target",
            "interval": "Clopper-Pearson exact 95 percent interval on the coverage proportion",
            "undercoverage_claim_condition": (
                "the upper Clopper-Pearson bound on measured i.i.d. coverage must be below 0.95 "
                "after the reference-target uncertainty sensitivity, which recomputes coverage at "
                "the target plus and minus 1.959964 reference standard errors and takes the "
                "least favourable result"
            ),
            "forbidden": (
                "calling the call-aware method calibrated from a favourable point "
                "estimate"
            ),
        },
    },
    "call_budget": {
        "calls_per_replicate": 32,
        "replicates_per_cell_at_maximum": 1250,
        "calls_per_cell_at_maximum": 40000,
        "calls_per_cell_at_minimum": 20800,
        "note": "650 replicates of 32 calls is the plan's stated per-cell minimum of 20,800 calls",
    },
}

# --------------------------------------------------------------------------------------
# Assignment generator
# --------------------------------------------------------------------------------------

ASSIGNMENT_GENERATOR: dict[str, Any] = {
    "policy": "operational",
    "policy_definition": (
        "items are partitioned at random subject only to context limits and the prespecified "
        "length strata. No composition, difficulty, or co-occurrence balancing is applied."
    ),
    "balanced_mechanism_policy": (
        "out of scope for this bounded paper. plan.md section 7.3 defines a separate balanced "
        "estimand with its own generator and target; mixing the two would pool two different "
        "objects."
    ),
    "length_strata": {
        "count": 16,
        "construction": (
            "sort the 2,048 frozen items by character count, then cut into 16 blocks of "
            "128"
        ),
        "call_rule": "every call takes exactly one item from each of the 16 strata",
        "effect": "total prompt length is balanced across calls by construction",
    },
    "partition": {
        "items": 2048,
        "calls_per_assignment": 128,
        "batch_size": 16,
        "item_appearances_per_assignment": 1,
        "method": (
            "for each assignment and each stratum, draw an independent permutation of that "
            "stratum's 128 items with a seed derived from sha256 of the key tuple; call b takes "
            "position b of every stratum permutation"
        ),
    },
    "slot_assignment": {
        "method": (
            "per call, a deterministic augmenting-path bipartite matching between the call's 16 "
            "items and slots 0-15, where an item may not take a slot it already used in an "
            "earlier assignment. Candidate slot order inside the matching is seeded."
        ),
        "guarantee": (
            "each item occupies 3 distinct slots across the 3 assignments, and each slot is used "
            "exactly 128 times per assignment"
        ),
        "feasibility": (
            "after two assignments an item has at least 14 of 16 slots free, so Hall's condition "
            "holds and the matching always succeeds"
        ),
    },
    "seeds": {
        "namespace": "step0-assign-v1",
        "partition_key": ["namespace", "cell_id", "assignment", "stratum"],
        "slot_key": ["namespace", "cell_id", "assignment", "call_index"],
        "replay_key": [
            "namespace",
            "cell_id",
            "replay",
            "assignment",
            "draw_index",
            "stratum",
            "execution_block",
        ],
        "derivation": "seed = int(sha256('|'.join(str(k) for k in key))[:16], 16)",
        "master_seed": 20260912,
    },
    "replay": {
        "purpose": "exact randomization-inference draws",
        "rule": (
            "a replay draw starts from the observed partition and independently permutes items "
            "within each assignment, length stratum, and observed execution block. It changes "
            "peers without moving an item between blocks. The observed assignment is excluded "
            "from the null draws."
        ),
        "determinism_requirement": (
            "the same draw index must reproduce the same partition byte for "
            "byte"
        ),
    },
    "emitted_artifacts": [
        "assignment_matrix.jsonl: one row per call with items in slot order",
        "job_skeletons.jsonl: one row per planned call with a globally unique call_id",
        "balance.json: realized strata, slot, class, and length diagnostics",
        "meta.json: seeds, counts, and SHA-256 of every emitted file",
    ],
}

EXECUTION_LEDGER: dict[str, Any] = {
    "a100_ceilings_hours": {
        "step_0_freeze": 0,
        "step_1_screen_and_benchmark": 8,
        "step_2_anchor_replication": 14,
        "step_3_three_fresh_cells": 32,
        "step_4_repeated_sampling": 46,
        "step_5_optional_sensitivity": 5,
        "step_6_reserve": 15,
        "total": 120,
    },
    "ceilings_are_not_predictions": True,
    "planned_call_counts": {
        "per_cell_joint": 384,
        "per_cell_exact_context": 6144,
        "per_cell_solo": 2048,
        "per_cell_base_total": 8576,
        "four_cells_base_total": 34304,
        "anchor_filler": 6144,
        "double_new_filler_subset": 512,
        "double_new_serial_subset": 544,
        "steps_2_and_3_total": 41504,
        "step_4_minimum_two_cells": 41600,
        "step_4_maximum_two_cells": 80000,
    },
    "throughput_anchor": {
        "source": "the pilot ran 8,375 prompts with 3,141 seconds of recorded generation time",
        "implied_rate_prompts_per_second": 2.666,
        "status": (
            "a workload anchor only. The pilot did not record comparable single-A100 throughput, "
            "so Step 1 must measure calls per second on the real prompt-length distribution and "
            "project the full remaining cost including loading and failure overhead."
        ),
    },
    "trim_order_if_the_budget_is_exceeded": [
        "delete step 5",
        (
            "reduce the evaluation-replicate maximum toward its frozen 400 minimum using "
            "timing data only"
        ),
        "stop before confirmatory calls and publish a revised, newly frozen smaller design",
    ],
    "never_trim": [
        "the exact-context control",
        "a primary cell, based on an observed effect",
        "the second model or task, after joint outputs are seen",
        "the reserve, before the two core result streams complete",
    ],
    "metering": (
        "every charged A100 minute is metered, including startup, loading, failures, and "
        "debugging"
    ),
}

CLAIM_OUTCOMES: dict[str, Any] = {
    "strong_bounded": (
        "the anchor replicates, the recurrence gate passes, and at least one frozen repeated-"
        "sampling cell shows a variance ratio of at least 1.25 with statistically supported "
        "i.i.d. undercoverage. Claim that preserving call lineage matters for the tested policies."
    ),
    "phenomenon_without_demonstrated_damage": (
        "excess covariance recurs but the variance or coverage endpoint is small or imprecise. "
        "Publish the dependence measurement and a bound on practical consequences. Do not say "
        "confidence intervals fail."
    ),
    "narrow_or_negative": (
        "only the anchor works, the exact-context control explains it, or neither fresh axis "
        "replicates. Report the boundary and stop the broad claim."
    ),
    "forbidden_claims": [
        "a new cluster-robust estimator",
        "a general statement about all large language models",
        "a mechanism for the covariance",
        "batching always harms accuracy",
        "measured coverage, when the reference stream is too imprecise",
        "a pooled p-value that mixes the old pilot with new data",
    ],
}

DISCOVERY_CONFIRMATION: dict[str, Any] = {
    "discovery_tier": [
        "the completed Gate A pilot on Qwen3-8B and Civil Comments, 2,000 records",
        "its reported joint ICC 0.0754 with decision interval [0.0182, 0.1314]",
        "its filler placebo ICC 0.0048 and fake-batch p 0.0010",
        "all analysis-code development and debugging",
    ],
    "disclosed_discovery_inputs_used_to_set_this_design": [
        "batch size 16",
        "three assignments",
        "the joint marginal error rate near 0.113, used to choose simulation scenarios",
        "the observed ICC magnitude near 0.075, used to choose the power scenario grid",
        "the 384-token output cap lesson",
        "the 0.02 parse-failure ceiling",
    ],
    "not_used_to_set_this_design": [
        "any per-assignment pilot value, to set a threshold",
        "any pilot outcome, to choose a cell, model, or task",
        "the pilot accuracy ordering, to set a direction",
    ],
    "confirmation_tier": [
        "qwen_secondtask",
        "mistral_civil",
        "mistral_secondtask",
        "both repeated-sampling streams",
    ],
    "anchor_tier": ["qwen_civil on fresh records, reported as a replication"],
    "pooling_rule": "the pilot is never pooled into a confirmatory p-value or interval",
    "post_hoc_rule": (
        "any change made after a confirmatory output is opened creates a new exploratory analysis "
        "and is labelled exploratory in the paper"
    ),
}

VENUE: dict[str, Any] = {
    "frozen_route": "IEEE BigData 2026 Undergraduate and REU Consortium",
    "frozen_route_url": "https://bigdataieee.org/BigData2026/undergraduate-reu-consortium/",
    "page_limit": 6,
    "deadline": "September 20, 2026, 11:59 PM AoE",
    "ineligible_route": "IEEE BigData 2026 High School Symposium",
    "ineligible_route_url": "https://bigdataieee.org/BigData2026/high-school-symposium/",
    "ineligible_reason": (
        "the High School Symposium requires the first author to be a high school student at the "
        "time of submission. The workspace describes Daniel as an undergraduate, so that track is "
        "ineligible."
    ),
    "conditional_on": (
        "Daniel being an undergraduate student pursuing an academic degree at the time of "
        "submission. This is a personal fact that the record cannot verify."
    ),
    "authorship_rule": "never assign first authorship merely to satisfy a venue eligibility rule",
    "retrieval_date_utc": "2026-09-11",
    "human_confirmation_required": True,
}

DECISIONS: list[dict[str, str]] = [
    {
        "id": "D1",
        "topic": "second task",
        "underspecified": "the plan names AG News first but does not pin a split or frame size",
        "decision": "AG News train split, revision eb185aa, 512 records per class, 2,048 total",
        "why_conservative": (
            "the train split is exactly class balanced at 30,000 per class, so 512 per class "
            "reproduces the natural frame composition instead of engineering it"
        ),
    },
    {
        "id": "D2",
        "topic": "AG News license",
        "underspecified": "the plan requires a license audit but states no outcome",
        "decision": (
            "AG News is declared license:unknown on HuggingFace. Research use proceeds; text is "
            "never redistributed. The release publishes ids, indices, character counts, and hashes"
        ),
        "why_conservative": "assumes no redistribution right rather than inferring one",
    },
    {
        "id": "D3",
        "topic": "pilot exclusion set",
        "underspecified": "the plan excludes the 2,000 pilot targets only",
        "decision": "exclude the 2,000 pilot targets and the 4,000 pilot filler records, 6,000 ids",
        "why_conservative": (
            "the filler records were inside pilot prompts served by the same model revision, so "
            "they are not fresh records"
        ),
    },
    {
        "id": "D4",
        "topic": "Monte Carlo count",
        "underspecified": "the plan requires a frozen count but names none",
        "decision": "2,000 replay draws per randomization test, and 2,000 fake-batch regroupings",
        "why_conservative": (
            "the smallest reachable p is 1/2001 = 0.0005, well under the Holm threshold of "
            "0.0167 for three cells. The count is fixed by resolution; simulated Type-I rates "
            "are reported without using them to tune the count"
        ),
    },
    {
        "id": "D5",
        "topic": "error-information threshold",
        "underspecified": "the plan defers the threshold to CPU simulation",
        "decision": "set from the simulation power curve; see simulations/results.json",
        "why_conservative": (
            "the threshold is the smallest prespecified grid rate at which the fixed design "
            "reaches the frozen power target for the frozen 0.002 floor. The grid includes "
            "unreachable rates and reachable rates immediately above the arithmetic boundary"
        ),
    },
    {
        "id": "D6",
        "topic": "covariance floor",
        "underspecified": "the plan sets 0.002 and allows WP0 to raise it",
        "decision": "keep 0.002 and report explicitly whether the fixed design can support it",
        "why_conservative": (
            "the floor is never lowered, and a design that cannot reach it is "
            "disclosed"
        ),
    },
    {
        "id": "D7",
        "topic": "finite population correction",
        "underspecified": "the plan requires the choice to be frozen but does not state it",
        "decision": (
            "off for both interval methods; the corrected version is a labelled "
            "sensitivity"
        ),
        "why_conservative": (
            "the correction narrows the i.i.d. interval and would make an undercoverage claim "
            "easier to reach"
        ),
    },
    {
        "id": "D8",
        "topic": "anchor filler condition",
        "underspecified": "the plan says the anchor repeats the pilot filler condition",
        "decision": "all three assignments, 6,144 calls, exactly repeating the pilot condition",
        "why_conservative": "a reduced control would weaken the only context-length placebo",
    },
    {
        "id": "D9",
        "topic": "double-new filler and serial subsets",
        "underspecified": "the plan says 'small, prespecified' without numbers",
        "decision": (
            "filler 512 target calls, serial 32 joint calls plus their 512 exact-context calls, "
            "all from the first 32 calls of assignment 1 in frozen call order"
        ),
        "why_conservative": (
            "the subset is named by frozen call order, so it cannot be chosen by "
            "outcome"
        ),
    },
    {
        "id": "D10",
        "topic": "execution blocks",
        "underspecified": "the plan requires randomized interleaved blocks but no count",
        "decision": (
            "8 blocks per cell, execution_block = call_index modulo 8, fresh session per "
            "block"
        ),
        "why_conservative": (
            "8 blocks give every arm a presence in every block while keeping enough calls per "
            "block to estimate a block effect"
        ),
    },
    {
        "id": "D11",
        "topic": "retry policy",
        "underspecified": "the plan forbids replacement but sets no retry limit",
        "decision": (
            "at most 1 retry, only for an infrastructure error or truncation, as a new linked "
            "call. Confirmatory endpoints use first responses only"
        ),
        "why_conservative": (
            "a first-response-only endpoint cannot be improved by selective "
            "retrying"
        ),
    },
    {
        "id": "D12",
        "topic": "capability floors",
        "underspecified": "the plan requires a floor but names no value",
        "decision": "0.70 solo accuracy for Civil Comments and AG News, 0.55 for MultiNLI",
        "why_conservative": (
            "the floors sit far below the pilot's 0.832 solo accuracy, so they cannot be "
            "reverse-engineered from a favourable result, and they still reject a model that "
            "cannot do the task"
        ),
    },
    {
        "id": "D13",
        "topic": "slot balance",
        "underspecified": "the plan requires position balance without an algorithm",
        "decision": (
            "exact bipartite matching so each item takes 3 distinct slots across "
            "assignments"
        ),
        "why_conservative": (
            "an exact constraint removes the chance that position mimics a call "
            "effect"
        ),
    },
    {
        "id": "D14",
        "topic": "second model replacement",
        "underspecified": "the plan requires a policy frozen before joint output",
        "decision": (
            "one replacement at most, from the pinned ordered list OLMo-2-1124-7B-Instruct "
            "then Phi-3.5-mini-instruct, triggered only by technical or capability failures "
            "in the solo-only screen; information-poor cells are not replaced"
        ),
        "why_conservative": (
            "both candidates were public and ungated at freeze time, and the trigger set "
            "contains no joint or covariance quantity"
        ),
    },
    {
        "id": "D15",
        "topic": "fake-batch bounded null band",
        "underspecified": "the plan names no band for the falsification endpoint",
        "decision": "the fake-batch covariance must stay below 0.001 in absolute value",
        "why_conservative": (
            "half the primary floor, so a fake-batch artefact large enough to matter fails the "
            "endpoint before it can be read as support"
        ),
    },
    {
        "id": "D16",
        "topic": "scipy dependency",
        "underspecified": "not addressed by the plan",
        "decision": (
            "use scipy for the Student t and Beta quantiles instead of hand-rolled "
            "versions"
        ),
        "why_conservative": "a tested library quantile is more reliable than a local approximation",
    },
    {
        "id": "D17",
        "topic": "serving engine version",
        "underspecified": (
            "the plan says one pinned serving engine but names no engine version, and an earlier "
            "draft of this freeze deferred the version to Step 1"
        ),
        "decision": (
            "pin vllm==0.28.0 with its manylinux x86_64 wheel and PyPI evidence now. Step 1 "
            "records only host facts: CUDA, driver, GPU, torch, and the installed version string"
        ),
        "why_conservative": (
            "a version chosen after the host is seen is a design choice made after the study "
            "starts. Pinning it now makes any later change a visible freeze violation"
        ),
    },
    {
        "id": "D18",
        "topic": "MultiNLI fallback frame",
        "underspecified": (
            "the plan names MultiNLI as a conditional fallback but does not say when its frame "
            "is built, and it does not divide 2,048 items into three classes"
        ),
        "decision": (
            "materialise and freeze the MultiNLI frame at Step 0 under natural prevalence, with "
            "premise and hypothesis joined by a frozen template"
        ),
        "why_conservative": (
            "the fallback frame cannot then be designed after an AG News screen result is known, "
            "and natural prevalence avoids engineering the class margins"
        ),
    },
    {
        "id": "D19",
        "topic": "information threshold criterion",
        "underspecified": "the plan defers the threshold to simulation but names no criterion",
        "decision": (
            "set the threshold from detection power at the 0.002 floor, and report the joint "
            "claim gate, which also requires the estimate to clear the floor, as a separate number"
        ),
        "why_conservative": (
            "an unbiased estimator clears a boundary equal to the truth about half the time, so a "
            "joint gate at the boundary can never reach a 0.90 target. Reporting both numbers "
            "avoids redefining the effect to make a power target pass"
        ),
    },
]

SIMULATION_SPEC: dict[str, Any] = {
    "purpose": [
        "justify and freeze the Monte Carlo draw count",
        "justify and freeze the solo-screen error-information threshold",
        "report whether the fixed design supports the 0.002 covariance floor",
        "name the power and Type-I diagnostics",
    ],
    "design_is_exact": (
        "every simulation uses the real 2,048-item frame, k=16, 128 calls per assignment, 3 "
        "assignments, the real length strata, and the real assignment generator"
    ),
    "error_model": {
        "joint_arm": (
            "beta-binomial exchangeable within call: call-level rate drawn from Beta with mean p "
            "and intraclass correlation rho, then item errors are Bernoulli given the call rate"
        ),
        "rho_from_delta": "rho = delta / (p * (1 - p)), because E[C] = rho * p * (1 - p)",
        "exact_arm": "independent Bernoulli, so its expected pair-weighted covariance is 0",
        "marginal_shift_variant": (
            "the exact arm runs at p minus 0.03 in a separate scenario, to confirm the centred "
            "statistic is not driven by a marginal accuracy difference"
        ),
    },
    "scenario_grid": {
        "marginal_error_rate_p": [0.05, 0.08, 0.113, 0.20],
        "true_excess_covariance_delta": [0.0, 0.002, 0.004, 0.0075],
        "information_threshold_sweep_p": [
            0.001,
            0.002,
            0.0025,
            0.005,
            0.01,
            0.015,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.08,
            0.10,
            0.113,
            0.15,
            0.20,
            0.30,
        ],
        "monte_carlo_count_study": [499, 999, 1999],
    },
    "power_target": 0.90,
    "alpha": 0.05,
    "holm_adjusted_alpha_for_three_cells": 0.0167,
    "diagnostics": [
        "typeI_rate_at_delta_zero",
        "typeI_rate_with_marginal_shift",
        "power_at_floor_0_002",
        "power_at_pilot_like_0_0075",
        "information_threshold_min_error_rate",
        "monte_carlo_resolution",
        "estimator_bias",
        "effect_interval_coverage",
    ],
    "tuning_boundary": (
        "the scenario grid is chosen from the disclosed discovery inputs only. No pilot "
        "per-assignment value, no pilot interval bound, and no pilot verdict is used to set a "
        "threshold."
    ),
}

REPRODUCIBILITY: dict[str, Any] = {
    "immutability": [
        "the Gate A artifact directories are historical evidence and are never modified",
        "this freeze directory is sealed with a SHA-256 manifest and read-only permissions",
        "a new decision requires a new freeze directory, never an edit",
    ],
    "recorded_per_run": list(CALL_LINEAGE_SCHEMA["fields"].keys()),
    "credentials": "never written into a config, command, log, prompt, trace, or report",
    "integrity_command": "step0 verify",
    "gate_a_reproduction": {
        "purpose": "confirm the historical analysis is reproducible from its saved raw rows",
        "input": "gate-experiments/a/artifacts/full/raw/predictions.jsonl",
        "method": (
            "run the unmodified Gate A analyze stage with a new absolute output directory, then "
            "compare declared key metrics against the preserved analysis.json and report.md"
        ),
        "historical_artifact_write_check": (
            "the SHA-256 of every preserved Gate A file is recorded before and after the rerun "
            "and must be identical"
        ),
    },
}

STEP0_EXIT_EVIDENCE: dict[str, str] = {
    "timestamped_protocol": (
        "PROTOCOL.md, protocol.lock.toml, protocol.lock.json in the freeze "
        "directory"
    ),
    "hashes": "MANIFEST.sha256 over every frozen file, validated by step0 verify",
    "null_power_simulation": "simulations/results.json and simulations/REPORT.md",
    "sample_frames": (
        "data/<task>/items.jsonl, filler_pool.jsonl, provenance.json for civil_comments, "
        "ag_news, and the multi_nli fallback"
    ),
    "model_and_engine_pins": "pins/model_pins.json and pins/engine_pin.json from live metadata",
    "assignment_generator": "assignments/<cell>/ matrices, job skeletons, balance diagnostics",
    "analysis_tests": "pytest over tests/, plus ruff and mypy",
    "old_analysis_reproduced": "repro/REPRO_REPORT.md with a PASS or FAIL verdict",
    "venue_route": "venue/VENUE.md and venue.json from live pages",
    "no_new_joint_outputs": NO_NEW_LABELS,
    "machine_readable_status": "STATUS.json, written by `step0 status`, with a COMPLETE or "
    "INCOMPLETE verdict per exit-gate item",
}


def protocol() -> dict[str, Any]:
    """The whole frozen protocol as one plain dictionary."""
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "no_new_experimental_model_labels": NO_NEW_LABELS,
        "status": (
            "frozen Step 0 design. No new joint, exact-context, solo, fake-batch, filler, or "
            "repeated-sampling output exists. Every threshold here is a planned decision rule, "
            "not a finding."
        ),
        "panel": PANEL,
        "models": MODELS,
        "second_model_replacement_policy": SECOND_MODEL_REPLACEMENT_POLICY,
        "tasks": TASKS,
        "solo_screen": SOLO_SCREEN,
        "prompts": PROMPTS,
        "conditions": CONDITIONS,
        "call_lineage_schema": CALL_LINEAGE_SCHEMA,
        "failure_policy": FAILURE_POLICY,
        "estimands": ESTIMANDS,
        "tests": TESTS,
        "repeated_sampling": REPEATED_SAMPLING,
        "assignment_generator": ASSIGNMENT_GENERATOR,
        "execution_ledger": EXECUTION_LEDGER,
        "claim_outcomes": CLAIM_OUTCOMES,
        "discovery_confirmation": DISCOVERY_CONFIRMATION,
        "venue": VENUE,
        "simulation_spec": SIMULATION_SPEC,
        "reproducibility": REPRODUCIBILITY,
        "step0_exit_evidence": STEP0_EXIT_EVIDENCE,
        "decisions": DECISIONS,
    }
