"""Independent units are fixed here, before any modelling: subjects for real data, seeds for sim."""

from __future__ import annotations

import numpy as np

SUBJECTS = tuple(range(1, 21))
SPLIT_SEED = 2026
SIM_CALIBRATION_SEEDS = tuple(range(0, 10))
SIM_TEST_SEEDS = tuple(range(100, 110))


def split_subjects() -> tuple[tuple[int, ...], tuple[int, ...]]:
    order = np.random.default_rng(SPLIT_SEED).permutation(SUBJECTS)
    half = len(SUBJECTS) // 2
    return tuple(sorted(int(s) for s in order[:half])), tuple(sorted(int(s) for s in order[half:]))
