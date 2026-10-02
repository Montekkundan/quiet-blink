import numpy as np

from quietblink.detector import eog_envelope
from quietblink.sim import NATURAL_PEAK, SimConfig, peak_per_amplitude, simulate

CLEAN = SimConfig(
    duration_s=120.0, jaw_per_min=0.0, movement_per_min=0.0, dropouts_per_min=0.0, noise_scale=0.3
)


def test_simulation_is_deterministic_and_seed_dependent():
    a, b, c = simulate(CLEAN, 3), simulate(CLEAN, 3), simulate(CLEAN, 4)
    np.testing.assert_array_equal(a.x, b.x)
    assert a.commands == b.commands
    assert not np.array_equal(a.x, c.x)


def test_signal_is_finite_adc_range():
    rec = simulate(SimConfig(duration_s=240.0), 0)
    assert np.isfinite(rec.x).all()
    assert rec.x.min() >= 0 and rec.x.max() <= 1023


def test_labelled_commands_align_with_generated_blinks_in_the_envelope():
    rec = simulate(CLEAN, 1)
    env = eog_envelope(rec.x, rec.fs)
    assert rec.commands
    for command in rec.commands:
        assert command.kind == len(command.onsets_ms)
        for onset in command.onsets_ms:
            assert onset in rec.blink_onsets_ms
            window = env[int(onset * rec.fs / 1000) : int((onset + 450) * rec.fs / 1000)]
            assert window.max() > 0.4 * NATURAL_PEAK


def test_envelope_is_quiet_away_from_blinks():
    rec = simulate(CLEAN, 2)
    env = eog_envelope(rec.x, rec.fs)
    quiet = np.ones(env.size, bool)
    for onset in rec.blink_onsets_ms:
        quiet[int((onset - 100) * rec.fs / 1000) : int((onset + 700) * rec.fs / 1000)] = False
    assert quiet.sum() > 0.3 * env.size
    assert env[quiet][int(2 * rec.fs) :].max() < 0.5 * NATURAL_PEAK


def test_blink_peak_is_calibrated_in_envelope_units_at_both_rates():
    assert abs(peak_per_amplitude(512.0, 0.17) - peak_per_amplitude(160.0, 0.17)) < 0.02
