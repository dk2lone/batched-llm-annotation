"""A100-hour block admission using all-worker metered time."""

from __future__ import annotations

from dataclasses import dataclass

TARGET_HOURS = 120.0
STOP_HOURS = 150.0
RESERVE_HOURS = 10.0
FORECAST_MULTIPLIER = 1.6


@dataclass(frozen=True)
class BlockDecision:
    start: bool
    reason: str
    cumulative_hours: float
    forecast_base_hours: float
    forecast_guarded_hours: float
    reserve_hours: float
    projected_total_hours: float


def may_start(
    cumulative_hours: float,
    forecast_base_hours: float,
    *,
    reserve_hours: float = RESERVE_HOURS,
    target_hours: float = TARGET_HOURS,
    stop_hours: float = STOP_HOURS,
) -> BlockDecision:
    """Call only at a complete-block boundary; never inspect scientific outcomes."""
    if cumulative_hours < 0 or forecast_base_hours < 0 or reserve_hours < 0:
        raise ValueError("hours cannot be negative")
    guarded = FORECAST_MULTIPLIER * forecast_base_hours
    total = cumulative_hours + guarded + reserve_hours
    if cumulative_hours >= target_hours:
        start, reason = False, "target_reached_after_complete_block"
    elif total > stop_hours:
        start, reason = False, "guarded_complete_block_exceeds_stop"
    else:
        start, reason = True, "complete_block_fits"
    return BlockDecision(start, reason, cumulative_hours, forecast_base_hours,
                         guarded, reserve_hours, total)


def must_stop(cumulative_hours: float, *, stop_hours: float = STOP_HOURS) -> bool:
    """Check after every metered call, including failed attempts and model loads."""
    if cumulative_hours < 0:
        raise ValueError("hours cannot be negative")
    return cumulative_hours >= stop_hours
