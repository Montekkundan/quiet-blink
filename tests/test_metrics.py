import numpy as np

from quietblink.detector import Event
from quietblink.metrics import (
    CMD2,
    CMD3,
    FALSE,
    MINUTES,
    MISS2,
    MISS3,
    bootstrap,
    counts,
    false_per_hour,
    match,
    mean_latency_ms,
    missed_percent,
    missed_percent_of_kind,
    rule_of_three_per_hour,
)
from quietblink.sim import Command

COMMANDS = (Command(2, (1000, 1300)), Command(3, (20000, 20300, 20600)))


def test_match_pairs_same_kind_after_last_blink_and_flags_the_rest():
    events = [Event(2, 1901), Event(2, 9000), Event(2, 20900)]
    result = match(events, COMMANDS)
    assert result.latencies_ms == {2: [601], 3: []}
    assert result.missed == {2: 0, 3: 1}
    assert result.false_events == [Event(2, 9000), Event(2, 20900)]


def test_event_before_the_last_blink_cannot_recover_a_command():
    assert match([Event(2, 1200)], COMMANDS[:1]).missed[2] == 1


def test_one_event_recovers_at_most_one_command():
    twice = (Command(2, (1000, 1300)), Command(2, (1500, 1800)))
    assert match([Event(2, 1901)], twice).missed[2] == 1


def test_counts_and_rates():
    row = counts(match([Event(2, 1901), Event(3, 5000)], COMMANDS), COMMANDS, 30.0, True)
    assert (row[FALSE], row[MINUTES], row[CMD2], row[CMD3]) == (1, 30.0, 1, 1)
    assert (row[MISS2], row[MISS3]) == (0, 1)
    assert false_per_hour(row) == 2.0
    assert missed_percent(row) == 50.0
    assert (missed_percent_of_kind(row, 2), missed_percent_of_kind(row, 3)) == (0.0, 100.0)
    assert mean_latency_ms(row, 2) == 601.0
    assert np.isnan(mean_latency_ms(row, 3))


def test_false_events_are_not_counted_when_the_stream_carries_injections():
    row = counts(match([Event(3, 5000)], COMMANDS), COMMANDS, 30.0, False)
    assert row[FALSE] == 0
    assert row[MINUTES] == 0


def test_rule_of_three():
    assert rule_of_three_per_hour(60.0) == 3.0
    assert rule_of_three_per_hour(30.0) == 6.0


def test_bootstrap_resamples_groups_and_is_deterministic():
    groups = np.zeros((6, 8))
    groups[:, MINUTES] = 10.0
    groups[:3, FALSE] = 2.0
    first = bootstrap(groups, false_per_hour, np.random.default_rng(1))
    assert first == bootstrap(groups, false_per_hour, np.random.default_rng(1))
    assert 0.0 <= first[0] < false_per_hour(groups.sum(axis=0)) < first[1] <= 12.0
