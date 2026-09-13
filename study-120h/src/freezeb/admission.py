"""Solo-only capability rules for the new model-task panel."""

from __future__ import annotations

from collections import Counter
from typing import Any

MIN_PARSE_RATE = 0.98
MIN_ERROR_RATE = 0.05
MAX_ERROR_RATE = 0.80
SINGLE_LABEL_FLOORS = {
    "civil_comments": 0.70,
    "ag_news": 0.70,
    "multi_nli": 0.55,
    "imdb": 0.70,
}


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def _set_micro_f1(predicted: list[Any], references: list[Any]) -> float:
    tp = fp = fn = 0
    for pred, ref in zip(predicted, references, strict=True):
        a, b = set(pred), set(ref)
        tp += len(a & b)
        fp += len(a - b)
        fn += len(b - a)
    return _f1(tp, fp, fn)


def _span_micro_f1(predicted: list[Any], references: list[Any]) -> float:
    return _set_micro_f1(predicted, references)


def evaluate(
    task: str, predicted: list[Any | None], references: list[Any]
) -> dict[str, Any]:
    """No joint output or observed dependence may enter this decision.

    `None` denotes an unresolved first response. Capability statistics use only
    parsed rows, while unresolved rows still reduce the parse-rate denominator.
    """
    if not predicted or len(predicted) != len(references):
        raise ValueError("solo predictions and references must align")
    paired = [(p, r) for p, r in zip(predicted, references, strict=True) if p is not None]
    parse_rate = len(paired) / len(predicted)
    if not paired:
        return {"admitted": False, "reasons": ["no_parsed_solo_labels"],
                "parse_rate": parse_rate}
    preds, refs = map(list, zip(*paired, strict=True))
    accuracy = sum(p == r for p, r in paired) / len(paired)
    error_rate = 1 - accuracy
    reasons: list[str] = []
    if parse_rate < MIN_PARSE_RATE:
        reasons.append("parse_rate_below_0.98")
    if not MIN_ERROR_RATE <= error_rate <= MAX_ERROR_RATE:
        reasons.append("error_information_outside_0.05_to_0.80")
    metrics: dict[str, float] = {"accuracy": accuracy, "error_rate": error_rate}
    if task in SINGLE_LABEL_FLOORS:
        if accuracy < SINGLE_LABEL_FLOORS[task]:
            reasons.append("accuracy_below_frozen_floor")
        if task == "civil_comments":
            counts = Counter(refs)
            if set(counts) != {0, 1}:
                reasons.append("civil_screen_missing_class")
            else:
                recall0 = sum(p == 0 and r == 0 for p, r in paired) / counts[0]
                recall1 = sum(p == 1 and r == 1 for p, r in paired) / counts[1]
                metrics.update(balanced_accuracy=(recall0 + recall1) / 2,
                               toxic_recall=recall1)
                if metrics["balanced_accuracy"] < 0.60:
                    reasons.append("balanced_accuracy_below_0.60")
                if recall1 < 0.25:
                    reasons.append("toxic_recall_below_0.25")
    elif task == "go_emotions":
        baseline = max(Counter(map(tuple, references)).values()) / len(references)
        metrics.update(majority_exact_baseline=baseline,
                       label_micro_f1=_set_micro_f1(preds, refs))
        if accuracy < baseline + 0.05:
            reasons.append("exact_set_accuracy_below_baseline_plus_0.05")
        if metrics["label_micro_f1"] < 0.35:
            reasons.append("label_micro_f1_below_0.35")
    elif task == "multinerd":
        baseline = max(Counter(map(str, references)).values()) / len(references)
        metrics.update(majority_exact_baseline=baseline,
                       span_micro_f1=_span_micro_f1(preds, refs))
        if accuracy < baseline + 0.05:
            reasons.append("exact_span_accuracy_below_baseline_plus_0.05")
        if metrics["span_micro_f1"] < 0.30:
            reasons.append("span_micro_f1_below_0.30")
    else:
        raise ValueError(f"unknown task {task}")
    return {"admitted": not reasons, "reasons": reasons,
            "parse_rate": parse_rate, **metrics}
