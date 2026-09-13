"""Deterministic solo-only balanced admission order for model-task cells."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    model: str
    family: str
    task: str
    model_order: int
    task_order: int

    @property
    def key(self) -> str:
        return f"{self.model}::{self.task}"


def balanced_order(cells: list[Cell]) -> list[Cell]:
    """Maximal family/task matching per round; deterministic coverage tie breaks.

    The caller supplies only solo-admitted cells. Each round uses a family and task
    at most once. Among maximum-cardinality matchings, new families, tasks, and models
    take priority in that order. The declared model/task candidate order breaks ties.
    """
    if len({c.key for c in cells}) != len(cells):
        raise ValueError("duplicate model-task cell")
    remaining = sorted(cells, key=lambda c: (c.model_order, c.task_order, c.key))
    seen_families: set[str] = set()
    seen_tasks: set[str] = set()
    seen_models: set[str] = set()
    out: list[Cell] = []
    while remaining:
        families = sorted({c.family for c in remaining})
        best: list[Cell] = []
        best_score: tuple[object, ...] | None = None

        def visit(
            index: int,
            used_tasks: set[str],
            chosen: list[Cell],
            families_in: tuple[str, ...] = tuple(families),
            remaining_in: tuple[Cell, ...] = tuple(remaining),
        ) -> None:
            nonlocal best, best_score
            if index == len(families_in):
                rank = tuple((c.model_order, c.task_order) for c in chosen)
                score: tuple[object, ...] = (
                    len(chosen),
                    len({c.family for c in chosen} - seen_families),
                    len({c.task for c in chosen} - seen_tasks),
                    len({c.model for c in chosen} - seen_models),
                    tuple((-m, -t) for m, t in rank),
                )
                if best_score is None or score > best_score:
                    best_score, best = score, chosen.copy()
                return
            if len(chosen) + len(families_in) - index < len(best):
                return
            family = families_in[index]
            for cell in (
                c for c in remaining_in if c.family == family and c.task not in used_tasks
            ):
                visit(index + 1, used_tasks | {cell.task}, [*chosen, cell])
            visit(index + 1, used_tasks, chosen)

        visit(0, set(), [])
        if not best:
            raise RuntimeError("balanced selection could not advance")
        best.sort(key=lambda c: (c.model_order, c.task_order, c.key))
        out.extend(best)
        selected = {c.key for c in best}
        remaining = [c for c in remaining if c.key not in selected]
        seen_families.update(c.family for c in best)
        seen_tasks.update(c.task for c in best)
        seen_models.update(c.model for c in best)
    return out


def panels(cells: list[Cell]) -> dict[str, list[str]]:
    ordered = balanced_order(cells)
    return {
        "primary": [c.key for c in ordered[:16]],
        "held_out": [c.key for c in ordered[16:28]],
        "screened_admitted_but_unassigned": [c.key for c in ordered[28:]],
    }
