import numpy as np

from quietblink.detector import Event, Params, blink_fires, count_commands, eog_envelope
from quietblink.gates import Gate, decide, excursions, prepare
from quietblink.sim import render_blinks

P = Params()


def test_excursions_report_bounds_and_peaks():
    env = np.array([0, 5, 9, 5, 0, 0, 7, 0, 6, 8], dtype=float)
    starts, ends, peaks = excursions(env, 4.0)
    assert starts.tolist() == [1, 6, 8]
    assert ends.tolist() == [4, 7, 10]
    assert peaks.tolist() == [9.0, 7.0, 8.0]


def test_excursions_empty_when_nothing_crosses():
    starts, ends, peaks = excursions(np.zeros(10), 1.0)
    assert starts.size == ends.size == peaks.size == 0


def test_default_gate_equals_firmware_rule(train):
    env = eog_envelope(train([1.0, 1.35, 3.0, 3.35, 3.7], 512), 512)
    firmware = count_commands((blink_fires(env, 512, P) * 1000 // 512).tolist(), P)
    assert decide(prepare(env, 512, P), Gate()) == firmware
    assert firmware == [Event(2, firmware[0].ms), Event(3, firmware[1].ms)]


def test_shape_gate_rejects_a_sustained_burst_but_keeps_blinks():
    fs = 512
    x = np.zeros(6 * fs)
    render_blinks(x, fs, [(1.0, 0.17, 60.0), (1.4, 0.17, 60.0)])
    rng = np.random.default_rng(0)
    burst = slice(3 * fs, int(4.2 * fs))
    x[burst] += 150 * rng.standard_normal(x[burst].size) * np.hanning(x[burst].size)
    env = eog_envelope(x, fs)
    scaled = P.scaled(0.5)
    plain = decide(prepare(env, fs, scaled), Gate())
    gated = decide(prepare(env, fs, scaled), Gate(max_ms=400.0, peak_cap=3.0))
    assert any(e.ms >= 3000 for e in plain)
    assert [e.kind for e in gated] == [2]
    assert gated[0].ms < 3000


def test_peak_cap_rejects_oversized_excursions(train):
    env = eog_envelope(train([1.0, 1.35], 512, peak=300.0), 512)
    prepared = prepare(env, 512, P)
    assert decide(prepared, Gate(peak_cap=1.2, max_ms=500.0)) == []
    assert [e.kind for e in decide(prepared, Gate(peak_cap=10.0, max_ms=500.0))] == [2]


def test_one_strong_blink_fires_twice_in_the_firmware_window_but_once_when_gated():
    x = np.zeros(4 * 512)
    render_blinks(x, 512, [(1.0, 0.17, 90.0)])
    env = eog_envelope(x, 512)
    scaled = P.scaled(0.8)
    fires = blink_fires(env, 512, scaled) * 1000 // 512
    assert len(fires) == 2
    assert fires[1] - fires[0] >= P.blink_debounce_ms
    prepared = prepare(env, 512, scaled)
    assert prepared.n_fires.tolist() == [2]
    assert len(prepared.end_ms) == 1
