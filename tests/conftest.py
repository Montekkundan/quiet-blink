import numpy as np
import pytest

from quietblink.sim import render_blinks


@pytest.fixture
def train():
    def build(
        onsets_s: list[float], fs: float, peak: float = 60.0, length_s: float = 6.0
    ) -> np.ndarray:
        x = np.zeros(int(length_s * fs))
        render_blinks(x, fs, [(t, 0.17, peak) for t in onsets_s])
        return x

    return build
