"""The frozen protocol must be complete, self-consistent, and round-trip through TOML."""

from __future__ import annotations

import json
import tomllib

import tomli_w

from step0 import freeze, lock

REQUIRED_TOP_LEVEL = (
    "panel",
    "models",
    "second_model_replacement_policy",
    "tasks",
    "solo_screen",
    "prompts",
    "conditions",
    "call_lineage_schema",
    "failure_policy",
    "estimands",
    "tests",
    "repeated_sampling",
    "assignment_generator",
    "execution_ledger",
    "claim_outcomes",
    "discovery_confirmation",
    "venue",
    "simulation_spec",
    "reproducibility",
    "step0_exit_evidence",
    "decisions",
)


def test_every_required_section_is_present() -> None:
    payload = lock.protocol()
    for key in REQUIRED_TOP_LEVEL:
        assert key in payload, key


def test_panel_counts_match_the_plan() -> None:
    panel = lock.PANEL
    assert panel["batch_size_k"] == 16
    assert panel["records_per_task"] == 2048
    assert panel["assignments_per_cell"] == 3
    assert len(panel["cells"]) == 4
    confirmation = [c for c in panel["cells"] if c["in_confirmation_set"]]
    assert len(confirmation) == 3
    assert {c["cell_id"] for c in panel["cells"] if c["repeated_sampling"]} == {
        "qwen_civil",
        "mistral_secondtask",
    }


def test_call_counts_are_internally_consistent() -> None:
    conditions = lock.CONDITIONS
    assert conditions["joint"]["calls_per_assignment"] * 3 == conditions["joint"]["calls_per_cell"]
    assert conditions["exact_context"]["calls_per_cell"] == 384 * 16
    counts = lock.EXECUTION_LEDGER["planned_call_counts"]
    assert counts["per_cell_base_total"] == 384 + 6144 + 2048
    assert counts["four_cells_base_total"] == 4 * counts["per_cell_base_total"]
    assert counts["steps_2_and_3_total"] == (
        counts["four_cells_base_total"]
        + counts["anchor_filler"]
        + counts["double_new_filler_subset"]
        + counts["double_new_serial_subset"]
    )


def test_a100_ceilings_sum_to_120() -> None:
    hours = dict(lock.EXECUTION_LEDGER["a100_ceilings_hours"])
    total = hours.pop("total")
    assert sum(hours.values()) == total == 120


def test_ag_news_is_the_first_candidate_and_multinli_is_the_fallback() -> None:
    assert lock.TASKS["second_task_first_candidate"]["task_id"] == "ag_news"
    assert lock.TASKS["second_task_conditional_fallback"]["task_id"] == "multi_nli"
    assert lock.SOLO_SCREEN["task_fallback_rule"]["order"] == ["ag_news", "multi_nli"]
    assert "joint" in lock.SOLO_SCREEN["task_fallback_rule"]["forbidden"]


def test_solo_screen_forbids_joint_inputs() -> None:
    forbidden = " ".join(lock.SOLO_SCREEN["inputs_forbidden"]).lower()
    for banned in ("joint", "exact-context", "fake-batch", "covariance", "coverage"):
        assert banned in forbidden


def test_target_only_conditions_name_exactly_one_document() -> None:
    shared = lock.PROMPTS["shared"]
    assert "{target_doc_id}" in shared["single_target_instruction"]
    assert set(shared["single_target_applies_to"]) == {
        "exact_context",
        "solo",
        "filler",
        "serial_exact",
    }
    assert "Return no other id" in shared["single_target_instruction"]


def test_second_model_replacement_triggers_are_solo_only() -> None:
    policy = lock.SECOND_MODEL_REPLACEMENT_POLICY
    joined = " ".join(policy["trigger_conditions_all_solo_only"]).lower()
    assert "joint" not in joined
    assert "covariance" not in joined
    assert policy["replacement_limit"] == 1
    assert policy["ordered_replacement_list"] == [
        "allenai/OLMo-2-1124-7B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
    ]
    assert all(model in lock.MODELS for model in policy["ordered_replacement_list"])


def test_floors_and_multiplicity_are_frozen() -> None:
    floors = lock.TESTS["practical_floors"]
    assert floors["covariance_floor_absolute"] == 0.002
    assert floors["variance_ratio_floor"] == 1.25
    assert lock.TESTS["primary_contrast"]["monte_carlo_draws"] == 2000
    families = lock.TESTS["multiplicity"]
    assert families["family_1_cell_contrasts"]["method"].startswith("Holm")
    assert len(families["family_1_cell_contrasts"]["members"]) == 3


def test_recurrence_gate_requires_the_double_new_cell() -> None:
    conditions = " ".join(lock.TESTS["recurrence_gate"]["pass_conditions_all_required"])
    assert "mistral_secondtask" in conditions
    assert "0.002" in conditions


def test_repeated_sampling_streams_and_intervals_are_frozen() -> None:
    rs = lock.REPEATED_SAMPLING
    assert rs["reference_stream"]["replicates"] == 250
    assert rs["evaluation_stream"]["replicates_minimum"] == 400
    assert rs["evaluation_stream"]["replicates_maximum"] == 1000
    assert rs["replicate_design"]["items_per_replicate"] == 512
    assert rs["replicate_design"]["calls_per_replicate"] == 32
    assert rs["interval_methods"]["call_aware"]["degrees_of_freedom"] == 31
    assert rs["interval_methods"]["iid"]["finite_population_correction"] is False
    assert rs["interval_methods"]["call_aware"]["finite_population_correction"] is False


def test_venue_route_is_the_undergraduate_consortium() -> None:
    assert lock.VENUE["page_limit"] == 6
    assert "Undergraduate" in lock.VENUE["frozen_route"]
    assert "High School" in lock.VENUE["ineligible_route"]
    assert lock.VENUE["human_confirmation_required"] is True


def test_decision_table_is_well_formed() -> None:
    ids = [d["id"] for d in lock.DECISIONS]
    assert ids == sorted(ids, key=lambda x: int(x[1:]))
    assert len(set(ids)) == len(ids)
    for row in lock.DECISIONS:
        assert set(row) == {"id", "topic", "underspecified", "decision", "why_conservative"}
        assert all(row[k].strip() for k in row)


def test_the_lock_round_trips_through_toml() -> None:
    payload = lock.protocol()
    payload["freeze"] = {"stamp_utc": "test"}
    payload["content_hash"] = "test"
    restored = tomllib.loads(tomli_w.dumps(freeze._toml_safe(payload)))
    assert restored["panel"]["batch_size_k"] == 16
    assert restored["tests"]["practical_floors"]["covariance_floor_absolute"] == 0.002
    assert len(restored["decisions"]) == len(lock.DECISIONS)


def test_protocol_md_names_every_required_element() -> None:
    payload = lock.protocol()
    payload["freeze"] = {"stamp_utc": "test"}
    payload["content_hash"] = "test"
    text = freeze.render_protocol_md(payload)
    for phrase in (
        "Batch size k = 16",
        "2048 distinct records",
        "exact_context",
        "fake_batch",
        "serial_serving",
        "interleaving",
        "Call lineage schema",
        "Holm",
        "recurrence_gate",
        "0.002",
        "1.25",
        "Clopper-Pearson",
        "Undergraduate and REU Consortium",
        "ag_news",
        "multi_nli",
    ):
        assert phrase in text, phrase


def test_the_protocol_never_claims_a_result() -> None:
    payload = lock.protocol()
    assert "No new joint" in payload["status"]
    forbidden = lock.CLAIM_OUTCOMES["forbidden_claims"]
    assert any("all large language models" in f for f in forbidden)


def test_no_design_field_is_left_for_a_later_stage() -> None:
    text = json.dumps(lock.protocol())
    assert "SET_BY_SIMULATION" not in text
    assert "RECORD_AT_STEP_1" not in text


def test_the_serving_engine_and_version_are_predeclared() -> None:
    serving = lock.PANEL["serving"]
    assert serving["engine"] == "vllm"
    assert serving["engine_version_pin"].count("==") == 1
    assert serving["engine_wheel_pin"].endswith(".whl")
    host_only = serving["recorded_from_the_step_1_host"]
    assert "cuda_version" in host_only
    assert "driver_version" in host_only
    assert not any("engine_version_pin" == field for field in host_only)


def test_an_information_poor_cell_never_triggers_model_replacement() -> None:
    policy = lock.SECOND_MODEL_REPLACEMENT_POLICY
    assert all("error" not in trigger for trigger in policy["trigger_conditions_all_solo_only"])
    assert "never activates model replacement" in policy[
        "information_poor_is_not_a_replacement_trigger"
    ]
    assert "not replaced" in lock.SOLO_SCREEN["information_poor_handling"]


def test_no_repository_claim_is_made_without_stored_evidence() -> None:
    policy = lock.SECOND_MODEL_REPLACEMENT_POLICY["gated_repo_check"]
    assert "pins/model_pins.json" in policy
    for spec in lock.MODELS.values():
        assert "pins/model_pins.json" in spec["access_evidence"]
    for key in ("second_task_first_candidate", "second_task_conditional_fallback"):
        assert lock.TASKS[key]["redistribution_permitted"] is False
        assert "never text" in lock.TASKS[key]["license_finding"]


def test_the_multinli_fallback_frame_can_be_materialised() -> None:
    spec = lock.TASKS["second_task_conditional_fallback"]
    for field in (
        "text_fields",
        "text_template",
        "label_field",
        "min_chars",
        "max_chars",
        "scan_limit",
        "sampling",
        "reference_label_rule",
        "n_items",
        "filler_pool_size",
    ):
        assert field in spec, field
    assert spec["sampling"].startswith("natural")
    assert "not materialised" not in spec["status"]


def test_every_output_states_that_no_new_labels_exist() -> None:
    assert "No new experimental model labels exist" in lock.NO_NEW_LABELS
    assert lock.protocol()["no_new_experimental_model_labels"] == lock.NO_NEW_LABELS


def test_the_information_threshold_criterion_is_detection_power() -> None:
    criterion = lock.SOLO_SCREEN["criteria"]["error_information_threshold"]["criterion"]
    assert "detection power" in criterion
    assert "half the time" in criterion
    assert lock.TESTS["practical_floors"]["covariance_floor_absolute"] == 0.002
