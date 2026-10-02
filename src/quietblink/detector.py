"""Python port of the MindClick blink path (control/control.ino, FW_VERSION 0.2.0).

Ported: Notch -> EEGFilter -> EOGFilter chain, |x| moving-average envelope,
blinkLowerThreshold/blinkUpperThreshold band, blinkDebounceMs, double/triple windows and the
blinkCount state machine. Not ported: focus/beta logic, calibration, serial commands.

Deviations from the firmware:
- Filters are designed with scipy for any sample rate (they reproduce the 512 Hz coefficients)
  and start in the steady state of the first sample; the firmware starts at zero state.
- The band test is evaluated at sample instants; the firmware polls between samples.
- A pending double is released before a blink arriving after its window, whereas the firmware
  tests the new blink first (differs only if both fall in the same millisecond).
- BLINK_COOLDOWN_MS only gates the focus path in the firmware and is off by default, so the
  command cooldown here is a new rule, not a port.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf
from typing import NamedTuple

import numpy as np
from scipy import signal


class Event(NamedTuple):
    kind: int
    ms: int


@dataclass(frozen=True)
class Params:
    blink_lower: float = 40.0
    blink_upper: float = 55.0
    blink_debounce_ms: int = 200
    double_blink_window_ms: int = 600
    triple_blink_window_ms: int = 600
    envelope_window_ms: int = 100

    def scaled(self, scale: float) -> Params:
        return replace(
            self, blink_lower=self.blink_lower * scale, blink_upper=self.blink_upper * scale
        )


def front_end(fs: float) -> np.ndarray:
    return np.vstack(
        [
            signal.butter(2, [48, 52], "bandstop", fs=fs, output="sos"),
            signal.butter(2, 45, "lowpass", fs=fs, output="sos"),
            signal.butter(2, 5, "highpass", fs=fs, output="sos"),
        ]
    )


def to_ms(index: np.ndarray, fs: float) -> np.ndarray:
    return (np.asarray(index) * 1000 // fs).astype(np.int64)


def eog_envelope(x: np.ndarray, fs: float, p: Params = Params()) -> np.ndarray:
    sos = front_end(fs)
    y, _ = signal.sosfilt(sos, x, zi=signal.sosfilt_zi(sos) * x[0])
    window = int(p.envelope_window_ms * fs) // 1000
    running = np.cumsum(np.abs(y))
    running[window:] -= running[:-window]
    return running / window


def debounce(times_ms: np.ndarray, gap_ms: int) -> np.ndarray:
    kept: list[int] = []
    k = int(np.searchsorted(times_ms, gap_ms))
    while k < len(times_ms):
        kept.append(k)
        k = int(np.searchsorted(times_ms, times_ms[k] + gap_ms))
    return np.array(kept, dtype=np.intp)


def blink_fires(env: np.ndarray, fs: float, p: Params) -> np.ndarray:
    in_band = np.flatnonzero((env > p.blink_lower) & (env < p.blink_upper))
    return in_band[debounce(to_ms(in_band, fs), p.blink_debounce_ms)]


def count_commands(
    blinks_ms: list[int], p: Params, cooldown_ms: int = 0, ibi_ratio: float = inf
) -> list[Event]:
    double_window, triple_window = p.double_blink_window_ms, p.triple_blink_window_ms
    events: list[Event] = []
    count, first, second, quiet_until = 0, 0, 0, 0
    for t in blinks_ms:
        if count == 2 and t - second > triple_window:
            release = second + triple_window + 1
            events.append(Event(2, release))
            quiet_until, count = release + cooldown_ms, 0
        elif count == 1 and t - first > double_window:
            count = 0
        if t < quiet_until:
            continue
        if count == 0:
            first, count = t, 1
        elif count == 1:
            second, count = t, 2
        else:
            gaps = (second - first, t - second)
            if max(gaps) <= ibi_ratio * min(gaps):
                events.append(Event(3, t))
                quiet_until = t + cooldown_ms
            count = 0
    if count == 2:
        events.append(Event(2, second + triple_window + 1))
    return events
