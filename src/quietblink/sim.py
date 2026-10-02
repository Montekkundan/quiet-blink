"""Software test harness for the blink detector. Not a physiological model.

One channel in ADC-like counts (10 bit, offset 512). Blink amplitude is specified as the peak of
the firmware envelope it produces, so thresholds are in the same units for every sample rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from quietblink.detector import eog_envelope

NATURAL_PEAK = 30.0
DELIBERATE_GAIN = 1.5
BLINK_SIGMA = 0.35
DELIBERATE_SIGMA = 0.25
ADC_MAX = 1023


@dataclass(frozen=True)
class Command:
    kind: int
    onsets_ms: tuple[int, ...]


@dataclass(frozen=True)
class Recording:
    x: np.ndarray
    fs: int
    commands: tuple[Command, ...]
    blink_onsets_ms: np.ndarray


@dataclass(frozen=True)
class SimConfig:
    duration_s: float = 480.0
    fs: int = 512
    commands_per_min: float = 3.0
    blinks_per_min: float = 16.0
    natural_double_fraction: float = 0.08
    jaw_per_min: float = 0.5
    movement_per_min: float = 1.0
    dropouts_per_min: float = 0.2
    noise_scale: float = 1.0
    deliberate_gain: float = DELIBERATE_GAIN


@cache
def peak_per_amplitude(fs: float, duration_s: float) -> float:
    x = np.zeros(int(2.5 * fs))
    add_blink(x, fs, 1.0, duration_s, 1.0)
    return float(eog_envelope(x, fs).max())


def add_blink(
    x: np.ndarray, fs: float, onset_s: float, duration_s: float, amplitude: float
) -> None:
    i0 = int(np.ceil(onset_s * fs))
    i1 = min(x.size, int(np.floor((onset_s + duration_s) * fs)) + 1)
    if i1 > i0:
        u = np.clip((np.arange(i0, i1) / fs - onset_s) / duration_s, 0.0, 1.0)
        x[i0:i1] += amplitude * np.sin(np.pi * u**0.7) ** 2


def render_blinks(x: np.ndarray, fs: float, blinks: list[tuple[float, float, float]]) -> None:
    for onset_s, duration_s, peak in blinks:
        add_blink(x, fs, onset_s, duration_s, peak / peak_per_amplitude(fs, round(duration_s, 3)))


def draw_blink(rng: np.random.Generator, peak: float, sigma: float) -> tuple[float, float]:
    duration = float(np.clip(rng.normal(0.17, 0.035), 0.10, 0.30))
    return duration, peak * float(np.exp(rng.normal(0.0, sigma)))


def deliberate_command(
    rng: np.random.Generator, start_s: float, kind: int, gain: float
) -> tuple[Command, list[tuple[float, float, float]]]:
    onsets = start_s + np.concatenate(([0.0], np.cumsum(rng.uniform(0.28, 0.45, kind - 1))))
    blinks = []
    for onset in onsets:
        duration, peak = draw_blink(rng, gain * NATURAL_PEAK, DELIBERATE_SIGMA)
        blinks.append((float(onset), duration, peak))
    return Command(kind, tuple(round(b[0] * 1000) for b in blinks)), blinks


def _pink(rng: np.random.Generator, n: int) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freq = np.fft.rfftfreq(n)
    freq[0] = freq[1]
    y = np.fft.irfft(spectrum / np.sqrt(freq / freq[1]), n)
    return y / y.std()


def _poisson_times(
    rng: np.random.Generator, rate_per_min: float, lo: float, hi: float
) -> np.ndarray:
    n = rng.poisson(rate_per_min * (hi - lo) / 60)
    return np.sort(rng.uniform(lo, hi, n))


def _command_starts(rng: np.random.Generator, cfg: SimConfig) -> list[float]:
    starts, t = [], 5.0
    while True:
        t += 8.0 + rng.exponential(max(60.0 / cfg.commands_per_min - 8.0, 1.0))
        if t > cfg.duration_s - 6.0:
            return starts
        starts.append(t)


def _spontaneous(
    rng: np.random.Generator, cfg: SimConfig, guards: list[tuple[float, float]]
) -> list[float]:
    gaps = rng.gamma(3.0, 60.0 / cfg.blinks_per_min / 3.0, int(cfg.duration_s))
    onsets = 2.0 + np.cumsum(gaps)
    pairs = onsets[rng.random(onsets.size) < cfg.natural_double_fraction]
    onsets = np.concatenate([onsets, pairs + rng.uniform(0.2, 0.55, pairs.size)])
    return [
        float(t)
        for t in np.sort(onsets)
        if t < cfg.duration_s - 2.0 and not any(lo <= t <= hi for lo, hi in guards)
    ]


def simulate(cfg: SimConfig, seed: int) -> Recording:
    rng = np.random.default_rng(seed)
    fs, n = cfg.fs, int(cfg.duration_s * cfg.fs)
    t = np.arange(n) / fs
    noise = cfg.noise_scale
    x = 12.0 * noise * _pink(rng, n) + 2.0 * noise * rng.standard_normal(n)
    alpha_phase = rng.uniform(0, 2 * np.pi, 3)
    x += (
        6.0
        * noise
        * (1 + 0.5 * np.sin(2 * np.pi * 0.07 * t + alpha_phase[0]))
        * np.sin(2 * np.pi * 10.2 * t + alpha_phase[1])
    )
    x += 8.0 * noise * np.sin(2 * np.pi * 60.0 * t + alpha_phase[2])

    commands, command_blinks, guards = [], [], []
    for start in _command_starts(rng, cfg):
        command, blinks = deliberate_command(
            rng, start, int(rng.choice([2, 3])), cfg.deliberate_gain
        )
        commands.append(command)
        command_blinks += blinks
        guards.append((start - 1.0, blinks[-1][0] + 1.5))
    natural = [
        (o, *draw_blink(rng, NATURAL_PEAK, BLINK_SIGMA)) for o in _spontaneous(rng, cfg, guards)
    ]
    render_blinks(x, fs, command_blinks + natural)

    for onset in _poisson_times(rng, cfg.jaw_per_min, 1.0, cfg.duration_s - 3.0):
        i0, length = int(onset * fs), int(rng.uniform(0.4, 1.5) * fs)
        sigma = 100.0 * np.exp(rng.normal(0.0, 0.5))
        x[i0 : i0 + length] += (
            sigma * rng.standard_normal(min(length, n - i0)) * np.hanning(length)[: n - i0]
        )
    for onset in _poisson_times(rng, cfg.movement_per_min, 1.0, cfg.duration_s - 6.0):
        for step in onset + np.cumsum(rng.uniform(0.1, 0.4, rng.integers(1, 4))):
            i0 = int(step * fs)
            x[i0:] += 120.0 * rng.standard_normal() * np.exp(-(t[i0:] - step) / 0.8)
    for onset in _poisson_times(rng, cfg.dropouts_per_min, 1.0, cfg.duration_s - 3.0):
        i0 = int(onset * fs)
        x[i0 : i0 + int(rng.uniform(0.2, 1.0) * fs)] = -512.0

    x = np.rint(np.clip(512.0 + x, 0, ADC_MAX))
    onsets = np.array(sorted(round(b[0] * 1000) for b in command_blinks + natural))
    return Recording(x, fs, tuple(commands), onsets)
