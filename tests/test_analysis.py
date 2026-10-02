from itertools import pairwise

import numpy as np
import pytest

from quietblink.analysis import frontier, paired_reduction, select, summarize
from quietblink.experiments import SCALES, evaluate, gate_grid, sim_streams
from quietblink.gates import Gate
from quietblink.metrics import CMD2, FALSE, MINUTES, MISS2
from quietblink.sim import SimConfig


def row(false: float, missed: float, commands: float = 100.0) -> np.ndarray:
    out = np.zeros(8)
    out[FALSE], out[MINUTES], out[CMD2], out[MISS2] = false, 60.0, commands, missed
    return out


def firmware_table(values: dict[float, tuple[float, float]]) -> dict:
    return {(scale, Gate()): {0: row(*values.get(scale, (0.0, 100.0)))} for scale in SCALES}


def test_select_minimises_false_rate_within_the_missed_budget():
    table = firmware_table({0.5: (100, 10), 0.6: (50, 25), 0.7: (20, 40)})
    key, feasible = select(table, "firmware", 30.0)
    assert key == (0.6, Gate()) and feasible


def test_select_falls_back_to_lowest_missed_rate_when_budget_is_unreachable():
    table = firmware_table({0.5: (100, 10), 0.6: (50, 25)})
    key, feasible = select(table, "firmware", 5.0)
    assert key == (0.5, Gate()) and not feasible


def test_select_breaks_ties_towards_lower_missed_then_lower_scale():
    table = firmware_table({0.6: (10, 20), 0.7: (10, 20), 0.8: (10, 15)})
    assert select(table, "firmware", 30.0)[0][0] == 0.8


def test_frontier_keeps_only_non_dominated_points():
    table = firmware_table({0.5: (100, 10), 0.6: (50, 25), 0.7: (60, 40), 0.8: (20, 40)})
    points = frontier(table, "firmware")
    assert [(p["false_per_hour"], p["missed_percent"]) for p in points][:3] == [
        (0.0, 100.0),
        (20.0, 40.0),
        (50.0, 25.0),
    ]
    rates = [p["false_per_hour"] for p in points]
    missed = [p["missed_percent"] for p in points]
    assert rates == sorted(rates)
    assert all(a > b for a, b in pairwise(missed))


def test_zero_false_events_report_the_rule_of_three_bound_with_unique_minutes():
    key = (1.0, Gate())
    table = {key: {1: row(0, 10), 2: row(0, 10)}}
    summary = summarize(table, key, streams_per_recording=3)
    assert summary["false_events"] == 0
    assert summary["negative_minutes"] == pytest.approx(40.0)
    assert summary["false_per_hour_rule_of_three"] == pytest.approx(3.0 * 60.0 / 40.0)


def test_rule_of_three_is_absent_when_events_were_seen():
    key = (1.0, Gate())
    summary = summarize({key: {1: row(2, 10), 2: row(1, 10)}}, key, streams_per_recording=1)
    assert "false_per_hour_rule_of_three" not in summary
    assert summary["false_per_hour"] == pytest.approx(3 / 120 * 60)


def test_paired_reduction_compares_the_same_groups():
    base, other = (1.0, Gate()), (1.0, Gate(cooldown_ms=1000))
    table = {base: {g: row(10, 0) for g in range(6)}, other: {g: row(5, 0) for g in range(6)}}
    result = paired_reduction(table, other, base)
    assert result["value"] == pytest.approx(0.5)
    assert result["ci95"] == (pytest.approx(0.5), pytest.approx(0.5))


def test_every_variant_has_a_grid_and_variants_do_not_overlap():
    grid = gate_grid()
    flat = [g for gates in grid.values() for g in gates]
    assert len(flat) == len(set(flat))


def test_evaluation_is_deterministic():
    cfg = SimConfig(duration_s=90.0)
    gates = [Gate(), Gate(cooldown_ms=1000, max_ms=300.0, peak_cap=1.8)]
    first = evaluate(sim_streams(cfg, [0, 1]), [0.7, 0.9], gates)
    second = evaluate(sim_streams(cfg, [0, 1]), [0.7, 0.9], gates)
    assert first.keys() == second.keys()
    for key in first:
        for group in first[key]:
            np.testing.assert_array_equal(first[key][group], second[key][group])


def test_isolated_blink_refires_only_when_strong_and_refire_fraction_counts_excursions(train):
    from quietblink.analysis import refire_fraction, refire_scan
    from quietblink.detector import eog_envelope
    from quietblink.experiments import Stream

    fires = list(refire_scan(0.8).values())
    assert fires == sorted(fires)
    assert (fires[0], fires[-1]) == (0, 2)
    weak, strong = train([1.0], 512, peak=45.0), train([1.0], 512, peak=135.0)
    streams = [Stream(0, x, eog_envelope(x, 512), 512.0, (), 1.0, True) for x in (weak, strong)]
    assert refire_fraction(streams, 0.8) == 0.5
