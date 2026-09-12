"""Typed configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .util import sha256_json


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "google/civil_comments"
    revision: str | None = None
    split: str = "validation"
    text_field: str = "text"
    label_field: str = "toxicity"
    toxicity_threshold: float = 0.5
    n_targets: int = 2000
    filler_pool_size: int = 4000
    sampling: str = "natural"  # natural | balanced
    seed: int = 20260911
    min_chars: int = 40
    max_chars: int = 1000
    scan_limit: int | None = None
    out_dir: str = "artifacts/data"


@dataclass(frozen=True)
class PlanConfig:
    k: int = 16
    assignments: int = 3
    seed: int = 7
    conditions: tuple[str, ...] = ("solo", "joint", "filler")
    filler_fraction: float = 1.0
    length_match_tolerance: float = 0.05
    filler_candidate_draw: int = 64
    out_dir: str = "artifacts/plan"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "Qwen/Qwen3-8B"
    revision: str | None = None
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 64
    max_num_batched_tokens: int | None = None
    enforce_eager: bool = False
    trust_remote_code: bool = False
    seed: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens_base: int = 64
    max_tokens_per_label: int = 12
    enable_thinking: bool = False


@dataclass(frozen=True)
class RunConfig:
    backend: str = "mock"  # mock | vllm
    out_path: str = "artifacts/raw/predictions.jsonl"
    chunk_size: int = 64
    token_counter: str = "estimate"  # estimate | hf
    chars_per_token: float = 3.4
    token_safety_margin: int = 64
    mock_seed: int = 11
    mock_base_error: float = 0.12
    mock_batch_effect: float = 0.25
    mock_parse_failure_rate: float = 0.02
    mock_missing_id_rate: float = 0.01


@dataclass(frozen=True)
class AnalyzeConfig:
    bootstrap: int = 2000
    permutations: int = 1000
    seed: int = 5
    alpha: float = 0.05
    proceed_rho: float = 0.02
    bounded_null_high: float = 0.01
    material_accuracy_gap: float = 0.02
    filler_explain_fraction: float = 0.8
    min_errors_per_assignment: int = 20
    max_parse_failure_rate: float = 0.02
    out_dir: str = "artifacts/analysis"


@dataclass(frozen=True)
class Config:
    name: str
    root: Path
    data: DataConfig
    plan: PlanConfig
    model: ModelConfig
    run: RunConfig
    analyze: AnalyzeConfig

    def path(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else self.root / candidate

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        for section in ("data", "plan", "model", "run", "analyze"):
            out[section] = asdict(getattr(self, section))
        return out

    def fingerprint(self) -> str:
        """Hash of every value that changes the experiment, excluding output paths."""
        payload = self.to_dict()
        for section in ("data", "plan", "analyze"):
            payload[section].pop("out_dir", None)
        payload["run"].pop("out_path", None)
        return sha256_json(payload)[:16]


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "plan": PlanConfig,
    "model": ModelConfig,
    "run": RunConfig,
    "analyze": AnalyzeConfig,
}


def _build(cls: type, values: dict[str, Any], section: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ConfigError(f"[{section}] has unknown keys: {', '.join(unknown)}")
    coerced = dict(values)
    for key, value in values.items():
        if known[key].type == "tuple[str, ...]" and isinstance(value, list):
            coerced[key] = tuple(value)
    return cls(**coerced)


def load_config(path: Path, root: Path | None = None) -> Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(raw) - set(_SECTIONS) - {"name"})
    if unknown:
        raise ConfigError(f"unknown top-level keys: {', '.join(unknown)}")
    sections = {key: _build(cls, raw.get(key, {}), key) for key, cls in _SECTIONS.items()}
    cfg = Config(
        name=str(raw.get("name", path.stem)),
        root=(root or path.parent.parent).resolve(),
        **sections,
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.data.sampling not in ("natural", "balanced"):
        raise ConfigError("data.sampling must be 'natural' or 'balanced'")
    if cfg.data.n_targets < cfg.plan.k:
        raise ConfigError("data.n_targets must be at least plan.k")
    if cfg.data.filler_pool_size < cfg.plan.k * 4:
        raise ConfigError("data.filler_pool_size is too small for length-matched filler draws")
    if cfg.plan.k < 2:
        raise ConfigError("plan.k must be 2 or more")
    if cfg.plan.assignments < 2:
        raise ConfigError("plan.assignments must be 2 or more")
    if not 0.0 < cfg.plan.filler_fraction <= 1.0:
        raise ConfigError("plan.filler_fraction must be in (0, 1]")
    for condition in cfg.plan.conditions:
        if condition not in ("solo", "joint", "filler"):
            raise ConfigError(f"unknown condition {condition!r}")
    if "joint" not in cfg.plan.conditions:
        raise ConfigError("the joint condition is the primary measurement and cannot be dropped")
    if cfg.run.backend not in ("mock", "vllm"):
        raise ConfigError("run.backend must be 'mock' or 'vllm'")
    if cfg.run.token_counter not in ("estimate", "hf"):
        raise ConfigError("run.token_counter must be 'estimate' or 'hf'")
    if cfg.model.temperature != 0.0:
        raise ConfigError("the gate fixes greedy decoding; model.temperature must be 0.0")
    if cfg.analyze.min_errors_per_assignment < 2:
        raise ConfigError("analyze.min_errors_per_assignment must be at least 2")
    if not 0.0 <= cfg.analyze.max_parse_failure_rate < 1.0:
        raise ConfigError("analyze.max_parse_failure_rate must be in [0, 1)")
