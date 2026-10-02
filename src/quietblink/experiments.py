"""Sweeps over threshold scale and gate parameters, evaluated per independent group."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import product
from math import inf
from pathlib import Path

import numpy as np

from quietblink.detector import Params, eog_envelope
from quietblink.gates import Gate, decide, prepare
from quietblink.metrics import FIELDS, counts, match
from quietblink.realdata import CHANNELS, inject_commands, load_subject, signal_scale
from quietblink.sim import DELIBERATE_GAIN, Command, SimConfig, simulate

BASE = Params()
SCALES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 2.0)
COOLDOWNS_MS = (1000, 3000)
SHAPES = tuple(
    dict(min_ms=lo, max_ms=hi, peak_cap=cap, ibi_ratio=ratio)
    for lo, hi, cap, ratio in product((0.0, 80.0), (300.0, 500.0), (1.2, 1.8, 3.0), (inf, 1.6))
)
VARIANTS = ("firmware", "cooldown", "gate", "both")
INJECTIONS_PER_RUN, INJECTION_REPEATS = 4, 2


@dataclass(frozen=True)
class Stream:
    group: int
    x: np.ndarray
    env: np.ndarray
    fs: float
    commands: tuple[Command, ...]
    minutes: float
    count_false: bool


def gate_grid() -> dict[str, list[Gate]]:
    return {
        "firmware": [Gate()],
        "cooldown": [Gate(cooldown_ms=c) for c in COOLDOWNS_MS],
        "gate": [Gate(**s) for s in SHAPES],
        "both": [Gate(cooldown_ms=c, **s) for c, s in product(COOLDOWNS_MS, SHAPES)],
    }


def sim_streams(cfg: SimConfig, seeds: Iterable[int]) -> Iterator[Stream]:
    for seed in seeds:
        rec = simulate(cfg, seed)
        yield Stream(
            seed,
            rec.x,
            eog_envelope(rec.x, rec.fs),
            rec.fs,
            rec.commands,
            cfg.duration_s / 60,
            True,
        )


def real_streams(root: Path, subjects: Iterable[int]) -> Iterator[Stream]:
    """Un-injected runs (false activations) and injected copies (missed commands), per channel."""
    for subject in subjects:
        data, fs = load_subject(root, subject)
        minutes = data.shape[2] / fs / 60
        for channel in range(len(CHANNELS)):
            raw_envs = [eog_envelope(run[channel], fs) for run in data]
            scale = signal_scale(raw_envs, fs)
            for run, raw_env in enumerate(raw_envs):
                x, env = data[run, channel] * scale, raw_env * scale
                yield Stream(subject, x, env, fs, (), minutes, True)
                for repeat in range(INJECTION_REPEATS):
                    rng = np.random.default_rng([subject, channel, run, repeat])
                    injected, commands = inject_commands(
                        x, env, fs, rng, DELIBERATE_GAIN, INJECTIONS_PER_RUN
                    )
                    yield Stream(
                        subject, injected, eog_envelope(injected, fs), fs, commands, minutes, False
                    )


Table = dict[tuple[float, Gate], dict[int, np.ndarray]]


def evaluate(streams: Iterable[Stream], scales: Iterable[float], gates: Iterable[Gate]) -> Table:
    scales, gates = list(scales), list(gates)
    table: Table = defaultdict(lambda: defaultdict(lambda: np.zeros(len(FIELDS))))
    for s in streams:
        for scale in scales:
            prepared = prepare(s.env, s.fs, BASE.scaled(scale))
            for gate in gates:
                result = match(decide(prepared, gate), s.commands)
                table[(scale, gate)][s.group] += counts(
                    result, s.commands, s.minutes, s.count_false
                )
    return table
