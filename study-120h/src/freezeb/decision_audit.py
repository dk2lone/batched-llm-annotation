"""CPU-only, post-hoc checks of the completed B3/B4 evidence."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np

from step0 import stats
from step0.hashing import derived_seed, sha256_file, sha256_text

from . import assign, b4_revised
from . import extension as ext
from . import extension_b3_analysis as b3
from . import extension_b4 as b4


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def repair_mapping(base: Path, recovery: Path, inventory: Path) -> dict[str, dict[str, Any]]:
    """Use the authenticated inventory, never event status, to identify repaired B4 calls."""
    records, _, _ = b4_revised.recovery_evidence(base, recovery, inventory)
    return records


def clean_replicates(planned: list[dict[str, Any]], repairs: dict[str, Any]) -> np.ndarray:
    """A replicate is retained only when every planned call is original."""
    counts = Counter(int(job["replicate"]) for job in planned)
    if any(value != 32 for value in counts.values()):
        raise ValueError("a B4 replicate must have exactly 32 calls")
    planned_ids = {str(job["call_id"]) for job in planned}
    if not set(repairs) <= planned_ids:
        raise ValueError("repair mapping contains a foreign call")
    contaminated = {int(job["replicate"]) for job in planned if job["call_id"] in repairs}
    return np.asarray([index for index in sorted(counts) if index not in contaminated], dtype=int)


def _intervals(blocks: np.ndarray) -> dict[str, np.ndarray]:
    iid, aware, nominal, means = [], [], [], []
    for block in blocks:
        flat = block.ravel()
        p = float(flat.mean())
        iid.append(stats.wald_interval(flat))
        aware.append(stats.call_t_interval(block.mean(axis=1)))
        nominal.append(p * (1 - p) / flat.size)
        means.append(p)
    return {
        "iid_low": np.asarray([item.low for item in iid]),
        "iid_high": np.asarray([item.high for item in iid]),
        "aware_low": np.asarray([item.low for item in aware]),
        "aware_high": np.asarray([item.high for item in aware]),
        "nominal": np.asarray(nominal),
        "means": np.asarray(means),
    }


def coverage_variant(
    evaluation: np.ndarray, target: float, target_se: float, cell: str
) -> dict[str, Any]:
    values = _intervals(evaluation)
    iid = stats.coverage(values["iid_low"], values["iid_high"], target)
    aware = stats.coverage(values["aware_low"], values["aware_high"], target)
    reference_uncertainty = stats.coverage_with_reference_uncertainty(
        values["iid_low"], values["iid_high"], target, target_se
    )
    rng = np.random.default_rng(
        int(sha256_text(f"decision-audit|{cell}|{len(evaluation)}")[:16], 16)
    )
    ratio = stats.variance_ratio(values["means"], values["nominal"], 2000, rng)
    return {
        "n_evaluation": len(evaluation),
        "iid_coverage": iid,
        "call_aware_coverage": aware,
        "mean_width_iid": float(np.mean(values["iid_high"] - values["iid_low"])),
        "mean_width_call_aware": float(np.mean(values["aware_high"] - values["aware_low"])),
        "variance_ratio": ratio,
        "worst_reference_upper_coverage": reference_uncertainty["least_favourable_upper_bound"],
        "reference_uncertainty": reference_uncertainty,
    }


def _call_metrics(predictions: dict[str, Any], target_ids: list[str]) -> dict[str, float]:
    labels = [predictions[doc_id] for doc_id in target_ids]
    counts = np.asarray(list(Counter(labels).values()), dtype=float)
    n = len(labels)
    fractions = counts / n
    return {
        "maximum_label_share": float(fractions.max()),
        "distinct_labels": float(len(counts)),
        "label_entropy": float(-(fractions * np.log(fractions)).sum()),
        "pairwise_label_agreement": float((counts * (counts - 1)).sum() / (n * (n - 1))),
    }


def _homogeneity(
    original: list[dict[str, float]], repaired: list[dict[str, float]], seed: str
) -> dict[str, Any]:
    if not original or not repaired:
        raise ValueError("homogeneity contrast needs both call groups")
    rng = np.random.default_rng(int(sha256_text(seed)[:16], 16))
    out: dict[str, Any] = {"original_calls": len(original), "repaired_calls": len(repaired)}
    for name in original[0]:
        a = np.asarray([row[name] for row in original])
        b = np.asarray([row[name] for row in repaired])
        # Each draw resamples whole calls within the observed cell and stream.
        a_idx = rng.integers(0, len(a), (999, len(a)))
        b_idx = rng.integers(0, len(b), (999, len(b)))
        draws = b[b_idx].mean(axis=1) - a[a_idx].mean(axis=1)
        out[name] = {
            "original_mean": float(a.mean()),
            "repaired_mean": float(b.mean()),
            "repaired_minus_original": float(b.mean() - a.mean()),
            "bootstrap_95_interval": [float(v) for v in np.percentile(draws, [2.5, 97.5])],
        }
    return out


def _call_error_covariance(errors: np.ndarray) -> float:
    calls = np.asarray(errors, dtype=float).reshape(-1, 16)
    return stats.covariance_from_stratified(calls)


def b4_audit(
    study: Path, base: Path, recovery: Path, inventory: Path, revised_path: Path
) -> dict[str, Any]:
    stored = _load(revised_path)
    records, repaired, hashes = b4_revised.recovery_evidence(base, recovery, inventory)
    if stored["repair_calls_used"] != len(records) or len(records) != len(repaired):
        raise ValueError("revised B4 repair count differs from authenticated inventory")
    cells: list[dict[str, Any]] = []
    used: set[str] = set()
    for cell in b4.CELLS:
        task = cell.rsplit("::", 1)[1]
        rows = b4.frame(study, task)
        arrays: dict[str, np.ndarray] = {}
        clean: dict[str, np.ndarray] = {}
        repairs_by_stream: dict[str, dict[str, Any]] = {}
        homogeneity: dict[str, Any] = {}
        streams: dict[str, Any] = {}
        for stream, count in (("reference", 250), ("evaluation", 400)):
            planned = list(b4.jobs(cell, rows, stream, count, render=False))
            ids = {str(job["call_id"]) for job in planned}
            stream_records = {key: value for key, value in records.items() if key in ids}
            stream_repairs = {key: value for key, value in repaired.items() if key in ids}
            repairs_by_stream[stream] = stream_records
            used.update(stream_records)
            clean[stream] = clean_replicates(planned, stream_records)
            original_events = b4.events(b4.stream_path(base, cell, stream))
            arrays[stream], report = b4_revised.revised_stream_errors(
                planned, original_events, rows, stream_records, stream_repairs
            )
            if report["repaired_calls"] != len(stream_records):
                raise ValueError("B4 repaired call count differs")
            final_original = b4.validate_events(planned, original_events)
            original_metrics: list[dict[str, float]] = []
            repaired_metrics: list[dict[str, float]] = []
            for job in planned:
                call_id = str(job["call_id"])
                event = final_original.get(
                    f"{call_id}::attempt::2", final_original.get(str(job["attempt_id"]))
                )
                if call_id in stream_repairs:
                    event = stream_repairs[call_id]
                    group = repaired_metrics
                else:
                    group = original_metrics
                if event is None or not isinstance(event.get("predictions"), dict):
                    raise ValueError("B4 call is unresolved after verified substitution")
                group.append(_call_metrics(event["predictions"], job["target_ids"]))
            homogeneity[stream] = _homogeneity(
                original_metrics, repaired_metrics, f"homogeneity|{cell}|{stream}"
            )
            call_errors = arrays[stream].reshape(-1, 16)
            repair_mask = np.asarray(
                [str(job["call_id"]) in stream_records for job in planned], dtype=bool
            )
            streams[stream] = {
                "original_calls": report["original_resolved_calls"],
                "repaired_calls": len(stream_records),
                "retained_replicates": len(clean[stream]),
                "excluded_replicates": count - len(clean[stream]),
                "all_call_error_covariance": _call_error_covariance(arrays[stream]),
                "original_call_error_covariance": _call_error_covariance(call_errors[~repair_mask]),
                "repaired_call_error_covariance": _call_error_covariance(call_errors[repair_mask]),
                "repair_free_replicate_error_covariance": _call_error_covariance(
                    arrays[stream][clean[stream]]
                ),
            }
            streams[stream]["all_minus_original_call_covariance"] = (
                streams[stream]["all_call_error_covariance"]
                - streams[stream]["original_call_error_covariance"]
            )
        existing = next(item["analysis"] for item in stored["cells"] if item["cell_key"] == cell)
        full_ref = arrays["reference"].mean(axis=(1, 2))
        clean_ref = arrays["reference"][clean["reference"]].mean(axis=(1, 2))
        full_target = float(full_ref.mean())
        original_target = float(clean_ref.mean())
        full_se = float(full_ref.std(ddof=1) / math.sqrt(len(full_ref)))
        original_se = float(clean_ref.std(ddof=1) / math.sqrt(len(clean_ref)))
        if not math.isclose(existing["reference"]["target"], full_target, abs_tol=1e-15):
            raise ValueError("B4 reference target differs from revised analysis")
        variants = {
            f"{reference}_{evaluation}": coverage_variant(
                arrays["evaluation"]
                if evaluation == "all"
                else arrays["evaluation"][clean["evaluation"]],
                full_target if reference == "existing" else original_target,
                full_se if reference == "existing" else original_se,
                cell,
            )
            for reference in ("existing", "original_only")
            for evaluation in ("all", "repair_free")
        }
        if variants["existing_all"]["iid_coverage"]["n_covered"] != existing[
            "n_covered"
        ] or not math.isclose(
            variants["existing_all"]["call_aware_coverage"]["point"],
            existing["intervals"]["call_aware_t_coverage"]["point"],
            abs_tol=1e-15,
        ):
            raise ValueError("B4 interval totals differ from stored revised report")
        mean_iid_half_width = variants["existing_all"]["mean_width_iid"] / 2
        original_precision = stats.reference_precision_check(clean_ref, mean_iid_half_width)
        evaluation_call_means = arrays["evaluation"].mean(axis=2)
        evaluation_marginal_error = float(arrays["evaluation"].mean())
        call_aware_variance = float(
            np.mean(evaluation_call_means.var(axis=1, ddof=1) / 32)
        )
        cells.append(
            {
                "cell_key": cell,
                "streams": streams,
                "a1_a2_coverage_2x2": variants,
                "reference_targets": {
                    "existing": full_target,
                    "existing_se": full_se,
                    "existing_precise": existing["reference_precise"],
                    "original_only_complete_replicates": original_target,
                    "original_only_se": original_se,
                    "difference_original_minus_existing": original_target - full_target,
                    "inside_existing_95_uncertainty_band": abs(original_target - full_target)
                    <= stats.Z975 * full_se,
                    "original_only_precision": original_precision,
                },
                "a3_homogeneity": homogeneity,
                "a3_variance_contribution": {
                    "all_minus_repair_free_variance_ratio": (
                        existing["variance_ratio"]["point"]
                        - variants["existing_repair_free"]["variance_ratio"]["point"]
                    ),
                    "comparison_note": (
                        "Descriptive full-versus-clean-replicate contrast; it also changes "
                        "the replicate population and is not a causal repair effect."
                    ),
                },
                "a5_frozen_variance": {
                    "variance_ratio": existing["variance_ratio"],
                    "call_aware_variance": call_aware_variance,
                    "iid_variance": existing["variance_ratio"]["mean_nominal_variance"],
                    "empirical_replicate_mean_variance": existing["variance_ratio"][
                        "empirical_variance"
                    ],
                    "call_aware_coverage": existing["intervals"]["call_aware_t_coverage"],
                    "iid_coverage": existing["intervals"]["iid_wald_coverage"],
                    "mean_width_call_aware": existing["intervals"]["mean_width_call_aware_t"],
                    "mean_width_iid": existing["intervals"]["mean_width_iid_wald"],
                    "effective_n_approx": 512 / existing["variance_ratio"]["point"],
                    "evaluation_marginal_error": evaluation_marginal_error,
                    "floor_0_002_inflation_approx": 1
                    + 15 * 0.002
                    / (evaluation_marginal_error * (1 - evaluation_marginal_error)),
                },
            }
        )
    if used != set(records):
        raise ValueError("some authenticated B4 repairs were not used")
    sensitivity_gates = {}
    for scenario in (
        "existing_all",
        "original_only_all",
        "existing_repair_free",
        "original_only_repair_free",
    ):
        gate_rows = []
        for item in cells:
            variant = item["a1_a2_coverage_2x2"][scenario]
            reference = item["reference_targets"]
            precise = (
                reference["existing_precise"]
                if scenario.startswith("existing_")
                else reference["original_only_precision"]["precise_enough_for_a_coverage_claim"]
            )
            gate_rows.append(
                {
                    "cell_key": item["cell_key"],
                    "n_evaluation": variant["n_evaluation"],
                    "n_covered": variant["iid_coverage"]["n_covered"],
                    "coverage": variant["iid_coverage"]["point"],
                    "worst_reference_upper_coverage": variant[
                        "worst_reference_upper_coverage"
                    ],
                    "reference_precise": precise,
                }
            )
        sensitivity_gates[scenario] = ext.c2_coverage_gate(gate_rows, list(b4.CELLS))
    clean_gate = sensitivity_gates["existing_repair_free"]
    prior_supported = set(stored["coverage_gate"]["supported_cells"])
    for item in cells:
        key = item["cell_key"]
        classification: str | None
        if key not in prior_supported:
            classification = None
        elif item["streams"]["evaluation"]["retained_replicates"] < ext.E5_MIN_EVALUATION:
            classification = "too_imprecise"
        elif key in clean_gate["supported_cells"]:
            classification = "still_supported"
        else:
            classification = "unsupported"
        item["a1_prior_supported_cell_classification"] = classification
        item["a1_classification_reason"] = (
            "Repair-free evaluation count is below the frozen 400-replicate C2 minimum"
            if classification == "too_imprecise"
            else None
        )
    return {
        "provenance": {
            "base_run": str(base),
            "revised_analysis_sha256": sha256_file(revised_path),
            **hashes,
            "b4_repair_calls": len(records),
        },
        "cells": cells,
        "frozen_c2": stored["coverage_gate"],
        "repair_free_frozen_c2_check": clean_gate,
        "a2_sensitivity_c2_checks": sensitivity_gates,
        "post_hoc": True,
    }


def _dependence(vector: np.ndarray, matrix: np.ndarray) -> dict[str, Any]:
    values = np.nan_to_num(vector, nan=0.0)[matrix].T
    p = float(values.mean())
    covariance = stats.covariance_from_stratified(values)
    denominator = p * (1 - p)
    call_shares = values.mean(axis=1)
    return {
        "marginal_error": p,
        "accuracy": 1 - p,
        "covariance": covariance,
        "standardized": covariance / denominator if denominator > 0 else None,
        "within_group_error_homogeneity": float(np.maximum(call_shares, 1 - call_shares).mean()),
    }


def _shift_interval(point: float, null: np.ndarray) -> list[float]:
    centered = null - null.mean()
    high, low = np.percentile(centered, [97.5, 2.5])
    return [float(point - high), float(point - low)]


def _point_for_interval(result: dict[str, Any], name: str) -> float:
    if name in ("joint", "exact_context", "solo_fake"):
        return float(result[name]["covariance"])
    if name.endswith("_standardized"):
        return float(result[name.removesuffix("_standardized")]["standardized"])
    if name.endswith("_homogeneity"):
        return float(
            result[name.removesuffix("_homogeneity")]["within_group_error_homogeneity"]
        )
    return float(result[name])


def replay_from_observed(
    observed: list[np.ndarray], cell_key: str, draw: int
) -> list[np.ndarray]:
    """Replay sealed within-stratum/block shuffles without rebuilding the fixed partition."""
    result = []
    for assignment, matrix in enumerate(observed, start=1):
        replay = matrix.copy()
        for stratum in range(16):
            for block in range(assign.N_BLOCKS):
                call_indices = np.arange(block, matrix.shape[1], assign.N_BLOCKS)
                rng = np.random.default_rng(
                    derived_seed(
                        ext.ASSIGN_NAMESPACE, cell_key, 16, assignment,
                        "replay", draw, stratum, block,
                    )
                )
                replay[stratum, call_indices] = rng.permutation(matrix[stratum, call_indices])
        result.append(replay)
    return result


def b3_audit(study: Path, draws: int = 319) -> dict[str, Any]:
    """Use sealed B3 assignments and conditional design replays for exploratory intervals."""
    lock_path = study / "full-grid-lock/20260914T032937Z-X1R4-A8/FULL_GRID_LOCK.json"
    lock = _load(lock_path)
    if lock["design"]["assignments"] != 3 or lock["design"]["batch_size"] != 16:
        raise ValueError("B3 assignment design differs")
    source = _load(study / "extension-b3/analysis/analysis.json")
    source_cells = {row["cell_key"]: row for row in source["cells"]}
    rows = []
    for model in lock["models"]:
        for task in lock["claim_dimensions"]["tasks"]:
            prepared = b3.prepare_cell(model, task, study, lock)
            key = prepared.cell.key
            stored = source_cells[key]
            observed: list[dict[str, Any]] = []
            null: dict[str, list[list[float]]] = {
                name: [[] for _ in range(3)]
                for name in (
                    "joint",
                    "exact_context",
                    "solo_fake",
                    "joint_standardized",
                    "exact_context_standardized",
                    "solo_fake_standardized",
                    "joint_homogeneity",
                    "exact_context_homogeneity",
                    "solo_fake_homogeneity",
                    "joint_minus_exact",
                    "standardized_joint_minus_exact",
                    "standardized_joint_minus_solo",
                )
            }
            for index, matrix in enumerate(prepared.matrices):
                joint = _dependence(prepared.joint[index], matrix)
                exact = _dependence(prepared.exact[index], matrix)
                solo = _dependence(prepared.solo, matrix)
                if not math.isclose(
                    float(joint["covariance"]),
                    stored["covariance_unresolved_as_correct"]["joint_per_assignment"][index],
                    abs_tol=1e-15,
                ) or not math.isclose(
                    float(exact["covariance"]),
                    stored["covariance_unresolved_as_correct"]["exact_context_per_assignment"][
                        index
                    ],
                    abs_tol=1e-15,
                ):
                    raise ValueError(f"B3 covariance differs from sealed A9: {key}")
                observed.append(
                    {
                        "assignment": index + 1,
                        "joint": joint,
                        "exact_context": exact,
                        "solo_fake": solo,
                        "joint_minus_exact": float(joint["covariance"])
                        - float(exact["covariance"]),
                        "standardized_joint_minus_exact": float(joint["standardized"])
                        - float(exact["standardized"]),
                        "standardized_joint_minus_solo": float(joint["standardized"])
                        - float(solo["standardized"]),
                    }
                )
            for draw in range(draws):
                matrices = replay_from_observed(prepared.matrices, key, draw)
                for index, matrix in enumerate(matrices):
                    joint = _dependence(prepared.joint[index], matrix)
                    exact = _dependence(prepared.exact[index], matrix)
                    solo = _dependence(prepared.solo, matrix)
                    null["joint"][index].append(float(joint["covariance"]))
                    null["exact_context"][index].append(float(exact["covariance"]))
                    null["solo_fake"][index].append(float(solo["covariance"]))
                    for name, value in (
                        ("joint", joint),
                        ("exact_context", exact),
                        ("solo_fake", solo),
                    ):
                        null[f"{name}_standardized"][index].append(
                            float(value["standardized"])
                        )
                        null[f"{name}_homogeneity"][index].append(
                            float(value["within_group_error_homogeneity"])
                        )
                    null["joint_minus_exact"][index].append(
                        float(joint["covariance"]) - float(exact["covariance"])
                    )
                    null["standardized_joint_minus_exact"][index].append(
                        float(joint["standardized"]) - float(exact["standardized"])
                    )
                    null["standardized_joint_minus_solo"][index].append(
                        float(joint["standardized"]) - float(solo["standardized"])
                    )
            for index, result in enumerate(observed):
                result["replay_shift_95_intervals"] = {
                    name: _shift_interval(
                        _point_for_interval(result, name), np.asarray(null[name][index])
                    )
                    for name in null
                }
            mean: dict[str, Any] = {
                name: float(np.mean([item[name] for item in observed]))
                for name in (
                    "joint_minus_exact",
                    "standardized_joint_minus_exact",
                    "standardized_joint_minus_solo",
                )
            }
            mean["standardized_joint_minus_exact_interval"] = _shift_interval(
                mean["standardized_joint_minus_exact"],
                np.mean(np.asarray(null["standardized_joint_minus_exact"]), axis=0),
            )
            mean["standardized_joint_minus_solo_interval"] = _shift_interval(
                mean["standardized_joint_minus_solo"],
                np.mean(np.asarray(null["standardized_joint_minus_solo"]), axis=0),
            )
            for condition in ("joint", "exact_context", "solo_fake"):
                for metric, null_key in (
                    ("covariance", condition),
                    ("standardized", f"{condition}_standardized"),
                    ("homogeneity", f"{condition}_homogeneity"),
                ):
                    observed_mean = float(
                        np.mean(
                            [
                                item[condition][
                                    "within_group_error_homogeneity"
                                    if metric == "homogeneity"
                                    else metric
                                ]
                                for item in observed
                            ]
                        )
                    )
                    mean[f"{condition}_{metric}_interval"] = _shift_interval(
                        observed_mean, np.mean(np.asarray(null[null_key]), axis=0)
                    )
            rows.append(
                {
                    "cell_key": key,
                    "family": prepared.cell.family,
                    "task": task,
                    "assignments": observed,
                    "cell_mean": mean,
                    "assignment_excess_range": float(
                        np.ptp([item["joint_minus_exact"] for item in observed])
                    ),
                }
            )
    if len(rows) != 16:
        raise ValueError("B3 audit requires all 16 cells")
    assignment_global = [
        float(np.mean([row["assignments"][index]["joint_minus_exact"] for row in rows]))
        for index in range(3)
    ]
    return {
        "provenance": {
            "a9_analysis_sha256": sha256_file(study / "extension-b3/analysis/analysis.json"),
            "assignment_lock_sha256": sha256_file(lock_path),
            "malformed_treatment": "unresolved_as_correct",
        },
        "assignment_interpretation": (
            "Three seeded length-stratified B3 item regroupings under the sealed assignment "
            "namespace. They are repeated assignments, not homogeneous/mixed composition arms. "
            "Exact-context and solo covariances use fake groups; only joint groups "
            "are shared calls."
        ),
        "interval_method": (
            f"{draws} conditional assignment replays; mean-centred shift interval, "
            "fixed labels"
        ),
        "cells": rows,
        "assignment_global_excess": assignment_global,
        "assignment_global_range": float(np.ptp(assignment_global)),
        "standardized_positive_cells": sum(
            row["cell_mean"]["standardized_joint_minus_exact"] > 0 for row in rows
        ),
        "standardized_positive_families": sorted(
            {
                row["family"]
                for row in rows
                if row["cell_mean"]["standardized_joint_minus_exact"] > 0
            }
        ),
        "post_hoc": True,
    }
