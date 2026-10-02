from pathlib import Path

import numpy as np
import pyedflib
import pytest

from quietblink.detector import eog_envelope
from quietblink.experiments import real_streams
from quietblink.gates import Gate
from quietblink.realdata import (
    CHANNELS,
    RUNS,
    candidates,
    inject_commands,
    load_subject,
    signal_scale,
)
from quietblink.sim import NATURAL_PEAK, add_blink
from quietblink.splits import SUBJECTS, split_subjects

FS, SECONDS = 160.0, 60


def synthetic_run(rng: np.random.Generator) -> np.ndarray:
    x = 6.0 * rng.standard_normal((len(CHANNELS), int(FS * SECONDS)))
    for onset in np.sort(rng.uniform(2, SECONDS - 3, 12)):
        for channel in x:
            add_blink(channel, FS, float(onset), 0.17, 400.0)
    return x


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("eegbci")
    rng = np.random.default_rng(0)
    for subject in SUBJECTS:
        for run in RUNS:
            headers = [
                {
                    "label": label,
                    "dimension": "uV",
                    "sample_frequency": FS,
                    "physical_min": -2000.0,
                    "physical_max": 2000.0,
                    "digital_min": -32768,
                    "digital_max": 32767,
                    "transducer": "",
                    "prefilter": "",
                }
                for label in CHANNELS
            ]
            writer = pyedflib.EdfWriter(str(folder / f"S{subject:03d}R{run:02d}.edf"), len(headers))
            writer.setSignalHeaders(headers)
            writer.writeSamples(list(synthetic_run(rng)))
            writer.close()
    return folder


def test_load_subject_shape_and_rate(root):
    data, fs = load_subject(root, 1)
    assert data.shape == (len(RUNS), len(CHANNELS), int(FS * SECONDS))
    assert fs == FS


def test_signal_scale_maps_median_candidate_peak_to_natural_peak(root):
    data, fs = load_subject(root, 1)
    envs = [eog_envelope(run[0], fs) for run in data]
    scale = signal_scale(envs, fs)
    peaks = np.concatenate([candidates(e * scale, fs)[1] for e in envs])
    assert np.median(peaks) == pytest.approx(NATURAL_PEAK, rel=0.02)


def test_injection_avoids_natural_blinks_is_finite_and_deterministic(root):
    data, fs = load_subject(root, 2)
    env = eog_envelope(data[0, 0], fs)
    runs = [
        inject_commands(data[0, 0], env, fs, np.random.default_rng(5), 1.5, 4) for _ in range(2)
    ]
    (x1, c1), (x2, c2) = runs
    np.testing.assert_array_equal(x1, x2)
    assert c1 == c2
    assert np.isfinite(x1).all()
    assert 1 <= len(c1) <= 4
    spans = candidates(env, fs)[0]
    for command in c1:
        start, end = command.onsets_ms[0] / 1000, command.onsets_ms[-1] / 1000
        assert not np.any((spans[:, 0] < end + 1.0) & (spans[:, 1] > start - 0.5))
    changed = np.flatnonzero(x1 != data[0, 0]) / fs
    assert all(
        any(c.onsets_ms[0] / 1000 - 0.1 <= t <= c.onsets_ms[-1] / 1000 + 0.5 for c in c1)
        for t in changed
    )


def test_real_streams_only_yield_requested_subjects_and_label_correctly(root):
    streams = list(real_streams(root, [3, 4]))
    assert {s.group for s in streams} == {3, 4}
    for stream in streams:
        assert stream.count_false == (not stream.commands)
        assert stream.minutes == pytest.approx(SECONDS / 60)
    assert any(s.commands for s in streams)


def test_calibration_and_test_tables_share_no_subject(root):
    from quietblink.experiments import evaluate

    calibration, test = split_subjects()
    tables = [
        evaluate(real_streams(root, subjects), [0.8], [Gate()]) for subjects in (calibration, test)
    ]
    seen = [set(table[(0.8, Gate())]) for table in tables]
    assert seen[0] == set(calibration)
    assert seen[1] == set(test)
    assert not seen[0] & seen[1]
