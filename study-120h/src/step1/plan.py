"""Turn frozen job skeletons into concrete calls, and rebuild the plan on a fallback frame.

Step 1 executes the `solo` condition only. The other conditions appear here for one reason:
the frozen context-fit criterion covers "every frozen joint, exact-context, solo, and filler
prompt", so the preflight must build and token-count all of them without generating anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from step0 import assign
from step0.hashing import sha256_json
from step0.lock import ASSIGNMENT_GENERATOR, CONDITIONS

from .frozen import FreezeError, Frozen, Item
from .prompts import Message, build_messages, flat_text

NAMESPACE = str(ASSIGNMENT_GENERATOR["seeds"]["namespace"])
N_BLOCKS = int(CONDITIONS["interleaving"]["n_execution_blocks_per_cell"])

#: Conditions whose prompts the frozen context-fit criterion must cover.
PREFLIGHT_CONDITIONS = ("joint", "exact_context", "solo", "filler", "serial_joint", "serial_exact")


@dataclass(frozen=True)
class Call:
    """One concrete call: the frozen skeleton plus its rendered prompt and output cap."""

    job: dict[str, Any]
    messages: list[Message]
    max_tokens: int

    @property
    def call_id(self) -> str:
        return str(self.job["call_id"])

    @property
    def condition(self) -> str:
        return str(self.job["condition"])

    @property
    def target_ids(self) -> list[str]:
        return [str(t) for t in self.job["target_ids"]]

    @property
    def prompt_sha256(self) -> str:
        return sha256_json(flat_text(self.messages))


def call_id_for(
    cell_id: str,
    model_id: str,
    task_id: str,
    condition: str,
    assignment: int,
    call_index: int,
    doc_order: list[str],
    targets: list[str],
) -> str:
    """The frozen call-id construction of `step0.assign.job_skeletons`."""
    return sha256_json(
        {
            "cell_id": cell_id,
            "model_id": model_id,
            "task_id": task_id,
            "condition": condition,
            "assignment": assignment,
            "call_index": call_index,
            "doc_order": doc_order,
            "targets": targets,
        }
    )[:16]


def execution_order_key_for(cell_id: str, execution_block: int, call_id: str) -> str:
    return sha256_json(
        {
            "namespace": NAMESPACE,
            "purpose": "within-block-execution-order",
            "cell_id": cell_id,
            "execution_block": execution_block,
            "call_id": call_id,
        }
    )


def retry_call_id(parent_call_id: str, retry_index: int) -> str:
    """A retry is a new call with a new id, linked to the call it retries."""
    if retry_index < 1:
        raise ValueError("a retry index starts at 1")
    return sha256_json({"retry_of": parent_call_id, "retry_index": retry_index})[:16]


def retry_job(job: dict[str, Any], retry_index: int) -> dict[str, Any]:
    """A new linked call. It never replaces the first response and never overwrites it."""
    parent = str(job["call_id"])
    new_id = retry_call_id(parent, retry_index)
    return {
        **job,
        "call_id": new_id,
        "parent_call_id": parent,
        "retry_index": retry_index,
        "execution_order_key": execution_order_key_for(
            str(job["cell_id"]), int(job["execution_block"]), new_id
        ),
    }


def _solo_rows(
    cell_id: str, model_id: str, task_id: str, doc_ids: list[str]
) -> list[dict[str, Any]]:
    """Rebuild the frozen solo rows of one cell for any frame.

    `step0.assign.job_skeletons` emits one solo call per distinct item of assignment 1, in
    sorted doc-id order, with `execution_block = index modulo 8`. Every item appears exactly
    once per assignment, so the sorted item frame is that same set. `test_step1_plan`
    asserts this reproduces the sealed rows byte for byte.
    """
    rows: list[dict[str, Any]] = []
    for index, doc_id in enumerate(sorted(doc_ids)):
        call_id = call_id_for(cell_id, model_id, task_id, "solo", 0, -1, [doc_id], [doc_id])
        block = index % N_BLOCKS
        rows.append(
            {
                "call_id": call_id,
                "parent_call_id": None,
                "retry_index": 0,
                "batch_id": call_id,
                "cell_id": cell_id,
                "model_id": model_id,
                "task_id": task_id,
                "condition": "solo",
                "assignment": 0,
                "call_index": -1,
                "execution_block": block,
                "execution_order_key": execution_order_key_for(cell_id, block, call_id),
                "doc_order": [doc_id],
                "target_ids": [doc_id],
                "target_slots": [0],
                "n_requested": 1,
                "serving_concurrency": 0,
            }
        )
    return rows


def _rebuild_full_plan(
    frozen: Frozen, cell: dict[str, Any], model_id: str, items: list[Item], filler: list[Item]
) -> list[dict[str, Any]]:
    """Replay the frozen generator on a frame it was not pre-materialised for.

    Used for the MultiNLI fallback and for a frozen-policy model replacement. The generator,
    its seeds, and its rules are the sealed ones; only the frame or the model id changes.
    """
    frame = assign.Frame(
        doc_ids=tuple(i.doc_id for i in items), strata=tuple(i.length_stratum for i in items)
    )
    calls = assign.build_cell(
        frame, str(cell["cell_id"]), int(frozen.protocol["panel"]["assignments_per_cell"])
    )
    return assign.job_skeletons(
        {**cell, "model_id": model_id},
        calls,
        {i.doc_id: i.n_chars for i in items},
        [(i.doc_id, i.n_chars) for i in filler],
    )


def plan_matches_freeze(frozen: Frozen, cell: dict[str, Any], model_id: str, task_id: str) -> bool:
    """True when the sealed skeletons already describe this cell, model, and frame."""
    if model_id != cell["model_id"]:
        return False
    if cell["task_id"] == "civil_comments":
        return task_id == "civil_comments"
    return task_id == frozen.planning_second_task


def job_rows(
    frozen: Frozen,
    cell_id: str,
    task_id: str,
    *,
    model_id: str | None = None,
    conditions: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Frozen job skeletons for one cell, restricted to `conditions`.

    When the cell, the model, and the frame are the sealed ones, the rows come straight from
    the sealed file. Otherwise the sealed generator is replayed on the requested frame.
    """
    cell = frozen.cell(cell_id)
    model = model_id or str(cell["model_id"])
    wanted = set(conditions or PREFLIGHT_CONDITIONS)

    if plan_matches_freeze(frozen, cell, model, task_id):
        rows = [r for r in frozen.job_skeletons(cell_id) if r["condition"] in wanted]
    elif wanted == {"solo"}:
        rows = _solo_rows(
            cell_id, model, str(cell["task_id"]), [i.doc_id for i in frozen.items(task_id)]
        )
    else:
        rebuilt = _rebuild_full_plan(
            frozen, cell, model, frozen.items(task_id), frozen.filler_pool(task_id)
        )
        rows = [r for r in rebuilt if r["condition"] in wanted]
    if not rows:
        raise FreezeError(f"{cell_id} has no planned call in conditions {sorted(wanted)}")
    return rows


def solo_jobs(
    frozen: Frozen, cell_id: str, task_id: str, *, model_id: str | None = None
) -> list[dict[str, Any]]:
    """The 2,048 solo calls of one cell, in frozen within-block execution order."""
    rows = job_rows(frozen, cell_id, task_id, model_id=model_id, conditions=("solo",))
    return sorted(rows, key=lambda r: (int(r["execution_block"]), str(r["execution_order_key"])))


def doc_texts(frozen: Frozen, task_id: str) -> dict[str, str]:
    """Item text and filler text of one frame. Filler text is needed for filler prompts."""
    texts = {i.doc_id: i.text for i in frozen.items(task_id)}
    texts.update({i.doc_id: i.text for i in frozen.filler_pool(task_id)})
    return texts


def build_call(frozen: Frozen, job: dict[str, Any], task_id: str, texts: dict[str, str]) -> Call:
    """Render one frozen skeleton into a concrete prompt and its frozen output cap."""
    spec = frozen.protocol["prompts"][task_id]
    docs = [(str(d), texts[str(d)]) for d in job["doc_order"]]
    targets = [str(t) for t in job["target_ids"]]
    messages = build_messages(spec, str(job["condition"]), docs, targets)
    return Call(job=job, messages=messages, max_tokens=frozen.max_tokens(len(targets)))
