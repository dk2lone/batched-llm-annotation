"""Freeze B task prompts; rendering is shared by solo, joint, and exact controls."""

from __future__ import annotations

from step0.lock import PROMPTS as ANCHOR_PROMPTS

from .labels import NER_TYPES

TASK_PROMPTS: dict[str, dict[str, str]] = {
    key: {field: str(ANCHOR_PROMPTS[key][field]) for field in (
        "system", "definition", "joint_instruction", "single_instruction"
    )}
    for key in ("civil_comments", "ag_news", "multi_nli")
}
TASK_PROMPTS.update({
    "imdb": {
        "system": (
            "You are a careful movie-review sentiment annotator. "
            "Answer with one JSON object and nothing else."
        ),
        "definition": (
            "Classify the overall sentiment of each movie review. "
            "Use 0 for negative and 1 for positive."
        ),
        "joint_instruction": "Label each of the {n} reviews below for overall sentiment.",
        "single_instruction": "Label one review for overall sentiment.",
    },
    "go_emotions": {
        "system": (
            "You are a careful emotion annotator. Answer with one JSON object and nothing else."
        ),
        "definition": (
            "Assign every emotion that applies. Return a nonempty list of integer IDs. "
            "0 admiration, 1 amusement, 2 anger, 3 annoyance, 4 approval, 5 caring, "
            "6 confusion, 7 curiosity, 8 desire, 9 disappointment, 10 disapproval, "
            "11 disgust, 12 embarrassment, 13 excitement, 14 fear, 15 gratitude, "
            "16 grief, 17 joy, 18 love, 19 nervousness, 20 optimism, 21 pride, "
            "22 realization, 23 relief, 24 remorse, 25 sadness, 26 surprise, 27 neutral. "
            "Labels must be unique. Neutral may co-occur with another label."
        ),
        "joint_instruction": "Label each of the {n} comments below for emotions.",
        "single_instruction": "Label one comment for emotions.",
    },
    "multinerd": {
        "system": (
            "You are a careful named-entity annotator. "
            "Answer with one JSON object and nothing else."
        ),
        "definition": (
            "For each document, return one integer BIO tag per whitespace-separated token, "
            "in token order. 0 means outside an entity. For each type, odd ID means B "
            "(start) and the next even ID means I (continuation): "
            + ", ".join(f"{2 * i + 1}/{2 * i + 2} {name}" for i, name in enumerate(NER_TYPES))
            + ". Use only IDs 0 through 30."
        ),
        "joint_instruction": "Tag every token in each of the {n} documents below.",
        "single_instruction": "Tag every token in one document.",
    },
})


def _example(task: str, docs: list[tuple[str, str]], ids: list[str]) -> str:
    text_by_id = dict(docs)
    labels: list[str] = []
    for doc_id in ids:
        if task == "go_emotions":
            label = "[0]"
        elif task == "multinerd":
            label = "[" + ",".join("0" for _ in text_by_id[doc_id].split()) + "]"
        else:
            label = "0"
        labels.append(f'"{doc_id}": {label}')
    return '{"labels": {' + ", ".join(labels) + "}}"


def build_messages(
    task: str,
    docs: list[tuple[str, str]],
    target_ids: list[str],
    *,
    variant: str = "baseline",
) -> list[dict[str, str]]:
    """Exact controls show identical docs and order but request one target label."""
    if task not in TASK_PROMPTS:
        raise ValueError(f"unknown task {task}")
    ids = [doc_id for doc_id, _ in docs]
    if (
        not ids or len(ids) != len(set(ids)) or not target_ids
        or len(target_ids) != len(set(target_ids))
    ):
        raise ValueError("nonempty unique document and target IDs are required")
    if not set(target_ids) <= set(ids):
        raise ValueError("every target ID must be shown")
    if len(target_ids) not in (1, len(ids)):
        raise ValueError("request either one target or all shown documents")
    if variant not in {"baseline", "definition_last", "brief_directive"}:
        raise ValueError(f"unknown variant {variant}")
    spec = TASK_PROMPTS[task]
    instruction = (
        spec["joint_instruction"].format(n=len(docs))
        if len(target_ids) == len(ids) else spec["single_instruction"]
    )
    rendered = "\n".join(f'<doc id="{doc_id}">\n{text}\n</doc>' for doc_id, text in docs)
    target_line = (
        "Return one label for every document ID above, and no other ID."
        if len(target_ids) == len(ids) else
        f'Return only the label for document ID "{target_ids[0]}". Ignore all other documents.'
    )
    if variant == "baseline":
        user = f"{instruction}\n\n{spec['definition']}\n\nDocuments:\n{rendered}\n\n{target_line}"
    elif variant == "definition_last":
        user = f"{instruction}\n\nDocuments:\n{rendered}\n\n{spec['definition']}\n\n{target_line}"
    else:
        user = f"{spec['definition']}\n\n{instruction}\nDocuments:\n{rendered}\n{target_line}"
    user += "\nAnswer with this JSON object and nothing else:\n" + _example(task, docs, target_ids)
    return [{"role": "system", "content": spec["system"]}, {"role": "user", "content": user}]


def estimated_response_tokens(task: str, n_requested: int, token_counts: list[int]) -> int:
    """Conservative fixed cap; the benchmark will measure actual generated tokens."""
    if task == "multinerd":
        return 64 + 5 * sum(token_counts)
    if task == "go_emotions":
        return 64 + 64 * n_requested
    return 64 + 20 * n_requested
