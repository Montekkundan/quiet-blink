"""Scoring of detected commands against labelled commands, and rates with uncertainty."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from quietblink.detector import Event
from quietblink.sim import Command

POST_MS = 1500
FIELDS = ("false", "minutes", "cmd2", "miss2", "cmd3", "miss3", "lat2", "lat3")
FALSE, MINUTES, CMD2, MISS2, CMD3, MISS3, LAT2, LAT3 = range(len(FIELDS))


class Match(NamedTuple):
    latencies_ms: dict[int, list[int]]
    missed: dict[int, int]
    false_events: list[Event]


def match(events: list[Event], commands: tuple[Command, ...]) -> Match:
    """Pair each command with the earliest unused same-kind event within 1.5 s of its last blink."""
    used = [False] * len(events)
    latencies: dict[int, list[int]] = {2: [], 3: []}
    missed = {2: 0, 3: 0}
    for command in commands:
        last = command.onsets_ms[-1]
        hit = next(
            (
                i
                for i, e in enumerate(events)
                if not used[i] and e.kind == command.kind and last <= e.ms <= last + POST_MS
            ),
            None,
        )
        if hit is None:
            missed[command.kind] += 1
        else:
            used[hit] = True
            latencies[command.kind].append(events[hit].ms - last)
    return Match(latencies, missed, [e for e, u in zip(events, used, strict=True) if not u])


def counts(
    result: Match, commands: tuple[Command, ...], minutes: float, count_false: bool
) -> np.ndarray:
    row = np.zeros(len(FIELDS))
    if count_false:
        row[FALSE], row[MINUTES] = len(result.false_events), minutes
    for kind, (cmd, miss, lat) in {2: (CMD2, MISS2, LAT2), 3: (CMD3, MISS3, LAT3)}.items():
        row[cmd] = sum(c.kind == kind for c in commands)
        row[miss] = result.missed[kind]
        row[lat] = sum(result.latencies_ms[kind])
    return row


def false_per_hour(row: np.ndarray) -> float:
    return float(row[FALSE] / row[MINUTES] * 60) if row[MINUTES] else float("nan")


def missed_percent(row: np.ndarray) -> float:
    total = row[CMD2] + row[CMD3]
    return float((row[MISS2] + row[MISS3]) / total * 100) if total else float("nan")


def missed_percent_of_kind(row: np.ndarray, kind: int) -> float:
    cmd, miss = (CMD2, MISS2) if kind == 2 else (CMD3, MISS3)
    return float(row[miss] / row[cmd] * 100) if row[cmd] else float("nan")


def mean_latency_ms(row: np.ndarray, kind: int) -> float:
    cmd, miss, lat = (CMD2, MISS2, LAT2) if kind == 2 else (CMD3, MISS3, LAT3)
    hits = row[cmd] - row[miss]
    return float(row[lat] / hits) if hits else float("nan")


def rule_of_three_per_hour(minutes: float) -> float:
    return 3.0 * 60.0 / minutes


def bootstrap(
    groups: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    n_boot: int = 2000,
) -> tuple[float, float]:
    """95% percentile interval, resampling whole groups (subjects or seeds) with replacement."""
    picks = rng.integers(0, len(groups), (n_boot, len(groups)))
    values = np.array([statistic(groups[i].sum(axis=0)) for i in picks])
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)
