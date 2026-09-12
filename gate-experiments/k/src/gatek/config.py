"""Typed configuration loaded from TOML (stdlib ``tomllib``, no YAML dependency)."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SyntheticConfig:
    """Latent 2D generator plus the three matched arms."""

    dims: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_demo: int = 60
    n_query: int = 24
    cluster_offset: float = 2.0
    cluster_sigma: float = 0.45
    margin: float = 0.75
    # Scale of the independent label-irrelevant columns, in units of the latent std.
    nuisance_scale: float = 1.0
    # "unit_std" rescales each dense projected column to unit standard deviation.
    # "raw" reproduces the source paper, which applies no scaling.
    dense_column_scale: str = "unit_std"
    number_format: str = "{:+08.3f}"
    # Optional local tokenizer directory used only to record token lengths on CPU.
    tokenizer_path: str | None = None


@dataclass(frozen=True)
class PlanConfig:
    """Which inference jobs the gate builds."""

    ablation_dims: list[int] = field(default_factory=lambda: [16, 32, 64])
    n_ablated_rows: int = 3
    random_removal_excludes_nearest: bool = True
    order_seed: int = 12345
    ordering_dims: list[int] = field(default_factory=lambda: [64])
    include_attention_job: bool = False
    attention_dims: list[int] = field(default_factory=lambda: [64])


@dataclass(frozen=True)
class RunConfig:
    """Backend and inference settings."""

    backend: str = "mock"
    model_id: str = "Qwen/Qwen3-8B"
    revision: str | None = None
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str = "sdpa"
    cache_dir: str | None = None
    # Pre-specified pooling layers. 24 is about two thirds of Qwen3-8B's 36 blocks.
    hidden_layers: list[int] = field(default_factory=lambda: [24])
    # "cosine" or "euclidean"; both are reported, this one is primary.
    repr_distance: str = "cosine"
    max_prompt_tokens: int = 16000
    label_tokens: list[str] = field(default_factory=lambda: ["A", "B"])
    enable_thinking: bool = False
    store_full_prompts: bool = False
    mock_seed: int = 7
    mock_repr_dim: int = 32


@dataclass(frozen=True)
class GateConfig:
    """Pre-registered thresholds. Change them here, not after seeing results."""

    min_classical_accuracy: float = 0.85
    min_anchor_accuracy: float = 0.65
    max_anchor_parse_failure_rate: float = 0.02
    min_nuisance_raw_contrast_ratio: float = 1.5
    min_repr_contrast_ratio: float = 1.2
    min_sparse_minus_dense_accuracy: float = 0.05
    min_sparse_minus_dense_ablation_delta: float = 0.05
    high_dims: list[int] = field(default_factory=lambda: [32, 64])
    sensitivity_scale: float = 0.2


@dataclass(frozen=True)
class OpenMLConfig:
    """Optional external-dataset expansion. Downloads only when invoked."""

    dataset_ids: list[int] = field(default_factory=lambda: [61, 187, 1510, 40975, 1462])
    dims: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64])
    projection_seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_demo: int = 60
    n_query: int = 24
    standardize_before_projection: bool = False


@dataclass(frozen=True)
class Config:
    name: str = "unnamed"
    out_dir: str = "artifacts/unnamed"
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    plan: PlanConfig = field(default_factory=PlanConfig)
    run: RunConfig = field(default_factory=RunConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    openml: OpenMLConfig = field(default_factory=OpenMLConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type[T], data: dict[str, Any]) -> T:
    """Construct a dataclass and reject unknown keys, so config typos are not silent."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> Config:
    """Read a TOML config file into a frozen ``Config``."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    sections = {
        "synthetic": SyntheticConfig,
        "plan": PlanConfig,
        "run": RunConfig,
        "gate": GateConfig,
        "openml": OpenMLConfig,
    }
    built: dict[str, Any] = {k: v for k, v in raw.items() if k not in sections}
    for key, cls in sections.items():
        if key in raw:
            built[key] = _build(cls, raw[key])
    cfg = _build(Config, built)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if 2 not in cfg.synthetic.dims:
        raise ValueError("synthetic.dims must include the 2D capability anchor")
    if any(d < 2 for d in cfg.synthetic.dims):
        raise ValueError("all synthetic dimensions must be at least 2")
    if not set(cfg.gate.high_dims) <= set(cfg.synthetic.dims):
        raise ValueError("gate.high_dims must be present in synthetic.dims")
    if not set(cfg.plan.ordering_dims) <= set(cfg.synthetic.dims):
        raise ValueError("plan.ordering_dims must be present in synthetic.dims")
    if cfg.synthetic.dense_column_scale not in {"raw", "unit_std"}:
        raise ValueError("synthetic.dense_column_scale must be 'raw' or 'unit_std'")
    if cfg.run.repr_distance not in {"cosine", "euclidean"}:
        raise ValueError("run.repr_distance must be 'cosine' or 'euclidean'")
    if len(cfg.run.label_tokens) != 2:
        raise ValueError("run.label_tokens must contain exactly two labels")
