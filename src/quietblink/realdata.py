"""Real forehead EEG (PhysioNet EEGMMIDB, 160 Hz) as natural-blink background, plus injection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyedflib

from quietblink.gates import excursions
from quietblink.sim import NATURAL_PEAK, Command, deliberate_command, render_blinks

CHANNELS = ("Fp1.", "Fpz.", "Fp2.")
RUNS = (6, 10, 14)
CANDIDATE_FACTOR = 3.0
GUARD_BEFORE_S, GUARD_AFTER_S, MIN_SPACING_S, EDGE_S = 0.8, 2.4, 10.0, 3.0


def load_subject(root: Path, subject: int) -> tuple[np.ndarray, float]:
    """Return an array of shape (runs, channels, samples) in microvolts and the sample rate."""
    runs, rate = [], 0.0
    for run in RUNS:
        reader = pyedflib.EdfReader(str(root / f"S{subject:03d}R{run:02d}.edf"))
        labels = reader.getSignalLabels()
        runs.append(np.stack([reader.readSignal(labels.index(c)) for c in CHANNELS]))
        rate = reader.getSampleFrequency(labels.index(CHANNELS[0]))
        reader.close()
    length = min(r.shape[1] for r in runs)
    return np.stack([r[:, :length] for r in runs]), float(rate)


def candidates(env: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Envelope excursions above a multiple of the median: natural blinks and artifacts."""
    starts, ends, peaks = excursions(env, CANDIDATE_FACTOR * float(np.median(env)))
    return np.column_stack([starts, ends]) / fs, peaks


def signal_scale(envs: list[np.ndarray], fs: float) -> float:
    """Factor that maps a subject-channel's median candidate peak onto NATURAL_PEAK."""
    peaks = np.concatenate([candidates(e, fs)[1] for e in envs])
    return NATURAL_PEAK / float(np.median(peaks))


def inject_commands(
    x: np.ndarray,
    env: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    deliberate_gain: float,
    count: int,
) -> tuple[np.ndarray, tuple[Command, ...]]:
    """Add deliberate blinks where no candidate excursion lies within the guard window."""
    spans, _ = candidates(env, fs)
    grid = np.arange(EDGE_S, x.size / fs - EDGE_S - GUARD_AFTER_S, 0.1)
    near = (spans[None, :, 0] < grid[:, None] + GUARD_AFTER_S) & (
        spans[None, :, 1] > grid[:, None] - GUARD_BEFORE_S
    )
    chosen: list[float] = []
    for start in rng.permutation(grid[~near.any(axis=1)]):
        if len(chosen) < count and all(abs(start - c) >= MIN_SPACING_S for c in chosen):
            chosen.append(float(start))
    out, commands = x.copy(), []
    for start in sorted(chosen):
        command, blinks = deliberate_command(rng, start, int(rng.choice([2, 3])), deliberate_gain)
        render_blinks(out, fs, blinks)
        commands.append(command)
    return out, tuple(commands)
