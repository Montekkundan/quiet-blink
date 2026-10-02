import numpy as np
import pytest

from quietblink.detector import (
    Event,
    Params,
    blink_fires,
    count_commands,
    debounce,
    eog_envelope,
    front_end,
)
from quietblink.gates import detect

P = Params()
FIRMWARE_SOS = np.array(
    [
        [0.96588529, -1.57986211, 0.96588529, 1.0, -1.58696045, 0.96505858],
        [1.0, -1.63566226, 1.0, 1.0, -1.62761184, 0.96671306],
        [0.05421270, 0.10842539, 0.05421270, 1.0, -1.24200128, 0.45885207],
        [0.95753983, -1.91507967, 0.95753983, 1.0, -1.91327599, 0.91688335],
    ]
)


def test_front_end_reproduces_firmware_biquads_at_512_hz():
    np.testing.assert_allclose(front_end(512), FIRMWARE_SOS, atol=2e-8)


def test_envelope_matches_firmware_ring_buffer():
    rng = np.random.default_rng(0)
    x = 512 + 80 * rng.standard_normal(2000)
    env = eog_envelope(x, 512)
    window = 51
    ring, total, expected = np.zeros(window), 0.0, []
    from scipy import signal

    sos = front_end(512)
    y = signal.sosfilt(sos, x, zi=signal.sosfilt_zi(sos) * x[0])[0]
    for i, value in enumerate(np.abs(y)):
        total += value - ring[i % window]
        ring[i % window] = value
        expected.append(total / window)
    np.testing.assert_allclose(env, expected, atol=1e-9)


@pytest.mark.parametrize("level", [0.0, 512.0])
def test_flat_input_fires_nothing(level):
    env = eog_envelope(np.full(5120, level), 512)
    assert detect(env, 512, P) == []


def test_debounce_spaces_sustained_fires_by_the_debounce_interval():
    env = np.full(1024, 45.0)
    fires = blink_fires(env, 512, P)
    times = fires * 1000 // 512
    assert times[0] >= P.blink_debounce_ms
    assert np.all(np.diff(times) >= P.blink_debounce_ms)
    assert np.all(np.diff(times) <= P.blink_debounce_ms + 2)
    assert len(times) == 9


def test_debounce_greedy_keeps_first_of_each_cluster():
    assert debounce(np.array([200, 250, 399, 400, 401, 700]), 200).tolist() == [0, 3, 5]


@pytest.mark.parametrize(
    ("blinks", "expected"),
    [
        ([1000], []),
        ([1000, 1300], [Event(2, 1901)]),
        ([1000, 1300, 1600], [Event(3, 1600)]),
        ([1000, 1300, 1900], [Event(3, 1900)]),
        ([1000, 1300, 1901], [Event(2, 1901)]),
        ([1000, 1601], []),
        ([1000, 1600], [Event(2, 2201)]),
        ([1000, 1300, 1600, 1900], [Event(3, 1600)]),
        ([1000, 1300, 1600, 1900, 2200], [Event(3, 1600), Event(2, 2801)]),
        ([1000, 1300, 1600, 3000, 3300], [Event(3, 1600), Event(2, 3901)]),
    ],
)
def test_state_machine_counts_single_double_triple(blinks, expected):
    assert count_commands(blinks, P) == expected


def test_cooldown_ignores_blinks_after_a_command():
    blinks = [1000, 1300, 1600, 1900, 2200, 2500]
    assert count_commands(blinks, P) == [Event(3, 1600), Event(3, 2500)]
    assert count_commands(blinks, P, cooldown_ms=1000) == [Event(3, 1600)]


def test_regularity_rejects_uneven_triples_only():
    uneven = [1000, 1250, 1800]
    assert count_commands(uneven, P) == [Event(3, 1800)]
    assert count_commands(uneven, P, ibi_ratio=1.6) == []
    assert count_commands([1000, 1300, 1600], P, ibi_ratio=1.6) == [Event(3, 1600)]


def test_pulse_trains_give_single_double_triple(train):
    for onsets, kinds in (
        ([1.0], []),
        ([1.0, 1.35], [2]),
        ([1.0, 1.35, 1.7], [3]),
        ([1.0, 3.0], []),
    ):
        x = train(onsets, 512)
        assert [e.kind for e in detect(eog_envelope(x, 512), 512, P)] == kinds


@pytest.mark.parametrize("onsets", [[1.0], [1.0, 1.35], [1.0, 1.35, 1.7], [1.0, 1.4, 1.8, 3.0]])
def test_same_physical_train_gives_same_events_at_160_and_512_hz(train, onsets):
    runs = {fs: detect(eog_envelope(train(onsets, fs), fs), fs, P) for fs in (160.0, 512.0)}
    assert [e.kind for e in runs[160.0]] == [e.kind for e in runs[512.0]]
    for slow, fast in zip(runs[160.0], runs[512.0], strict=True):
        assert abs(slow.ms - fast.ms) <= 20
