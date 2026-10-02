"""Confidence gates layered on the firmware blink rule.

Shape gate: a blink is one envelope excursion above the lower threshold, registered when the
excursion ends, and rejected if its duration or peak is out of range.
Regularity gate: a triple is rejected if its two inter-blink gaps differ by more than a ratio.
Cooldown: blinks are ignored for a while after a command fires.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

import numpy as np

from quietblink.detector import Event, Params, blink_fires, count_commands, debounce, to_ms


@dataclass(frozen=True)
class Gate:
    cooldown_ms: int = 0
    min_ms: float = 0.0
    max_ms: float = inf
    peak_cap: float = inf
    ibi_ratio: float = inf

    @property
    def shaped(self) -> bool:
        return self.min_ms > 0 or self.max_ms < inf or self.peak_cap < inf


@dataclass(frozen=True)
class Prepared:
    p: Params
    fire_ms: np.ndarray
    end_ms: np.ndarray
    duration_ms: np.ndarray
    peak: np.ndarray
    n_fires: np.ndarray


def excursions(env: np.ndarray, lower: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.diff(np.concatenate(([False], env > lower, [False])).astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    bounds = np.column_stack([starts, ends]).ravel()
    peaks = np.maximum.reduceat(np.append(env, 0.0), bounds)[::2] if bounds.size else bounds
    return starts, ends, peaks


def prepare(env: np.ndarray, fs: float, p: Params) -> Prepared:
    fires = blink_fires(env, fs, p)
    starts, ends, peaks = excursions(env, p.blink_lower)
    hit, n_fires = np.unique(np.searchsorted(starts, fires, side="right") - 1, return_counts=True)
    return Prepared(
        p,
        to_ms(fires, fs),
        to_ms(ends[hit], fs),
        (ends[hit] - starts[hit]) * 1000 / fs,
        peaks[hit],
        n_fires,
    )


def decide(prep: Prepared, gate: Gate) -> list[Event]:
    p = prep.p
    blinks = prep.fire_ms
    if gate.shaped:
        keep = (
            (prep.duration_ms >= gate.min_ms)
            & (prep.duration_ms <= gate.max_ms)
            & (prep.peak <= gate.peak_cap * p.blink_upper)
        )
        blinks = prep.end_ms[keep]
        blinks = blinks[debounce(blinks, p.blink_debounce_ms)]
    return count_commands(blinks.tolist(), p, gate.cooldown_ms, gate.ibi_ratio)


def detect(env: np.ndarray, fs: float, p: Params, gate: Gate = Gate()) -> list[Event]:
    return decide(prepare(env, fs, p), gate)
