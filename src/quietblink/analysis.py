"""Operating-point selection on the calibration split and held-out reporting."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, replace
from itertools import product
from math import inf, isnan
from pathlib import Path
from typing import Any

import numpy as np

from quietblink.detector import blink_fires, eog_envelope
from quietblink.experiments import (
    BASE,
    SCALES,
    VARIANTS,
    Stream,
    Table,
    evaluate,
    gate_grid,
    real_streams,
    sim_streams,
)
from quietblink.gates import Gate, prepare
from quietblink.metrics import (
    CMD2,
    CMD3,
    FALSE,
    FIELDS,
    MINUTES,
    bootstrap,
    false_per_hour,
    mean_latency_ms,
    missed_percent,
    missed_percent_of_kind,
    rule_of_three_per_hour,
)
from quietblink.realdata import CHANNELS
from quietblink.sim import SimConfig, render_blinks
from quietblink.splits import SIM_CALIBRATION_SEEDS, SIM_TEST_SEEDS, split_subjects

BUDGETS = (30.0, 20.0)
PRIMARY_BUDGET = BUDGETS[0]
BOOTSTRAP_SEED = 7
AXES = {
    "jaw_per_min": (0.0, 0.5, 2.0, 6.0),
    "movement_per_min": (0.0, 1.0, 4.0, 12.0),
    "noise_scale": (0.5, 1.0, 2.0, 3.0),
    "deliberate_gain": (1.0, 1.25, 1.5, 2.0),
}
Key = tuple[float, Gate]


def pooled(table: Table, key: Key, groups: set[int] | None = None) -> np.ndarray:
    rows = [row for g, row in table[key].items() if groups is None or g in groups]
    return np.sum(rows, axis=0)


def select(table: Table, variant: str, budget: float) -> tuple[Key, bool]:
    scored = []
    for scale, gate in product(SCALES, gate_grid()[variant]):
        row = pooled(table, (scale, gate))
        scored.append((false_per_hour(row), missed_percent(row), scale, (scale, gate)))
    feasible = [s for s in scored if s[1] <= budget]
    if feasible:
        return min(feasible, key=lambda s: s[:3])[3], True
    return min(scored, key=lambda s: (s[1], s[0], s[2]))[3], False


def frontier(table: Table, variant: str) -> list[dict[str, Any]]:
    points = []
    for scale, gate in product(SCALES, gate_grid()[variant]):
        row = pooled(table, (scale, gate))
        points.append((false_per_hour(row), missed_percent(row), scale, gate))
    points.sort(key=lambda p: (p[0], p[1]))
    kept, best = [], inf
    for fa, missed, scale, gate in points:
        if missed < best:
            best = missed
            kept.append(
                {"false_per_hour": fa, "missed_percent": missed, **config_json(scale, gate)}
            )
    return kept


def config_json(scale: float, gate: Gate) -> dict[str, Any]:
    return {"scale": scale, "gate": {k: (None if v == inf else v) for k, v in asdict(gate).items()}}


def summarize(table: Table, key: Key, streams_per_recording: int) -> dict[str, Any]:
    groups = np.array([table[key][g] for g in sorted(table[key])])
    row = groups.sum(axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out: dict[str, Any] = {
        "false_per_hour": false_per_hour(row),
        "false_per_hour_ci95": bootstrap(groups, false_per_hour, rng),
        "false_events": int(row[FALSE]),
        "negative_minutes": row[MINUTES] / streams_per_recording,
        "stream_minutes": row[MINUTES],
        "missed_percent": missed_percent(row),
        "missed_percent_ci95": bootstrap(groups, missed_percent, rng),
        "commands": int(row[CMD2] + row[CMD3]),
        "missed_percent_by_kind": {
            "double": missed_percent_of_kind(row, 2),
            "triple": missed_percent_of_kind(row, 3),
        },
        "false_per_hour_by_group": {
            str(g): false_per_hour(table[key][g]) for g in sorted(table[key])
        },
        "mean_latency_ms": {"double": mean_latency_ms(row, 2), "triple": mean_latency_ms(row, 3)},
    }
    if row[FALSE] == 0:
        out["false_per_hour_rule_of_three"] = rule_of_three_per_hour(out["negative_minutes"])
    return out


def false_reduction(row: np.ndarray) -> float:
    base = row[len(FIELDS) + FALSE]
    return float(1 - row[FALSE] / base) if base else float("nan")


def paired_reduction(table: Table, key: Key, base: Key) -> dict[str, Any]:
    """Relative cut in false activations versus the firmware rule, resampling the same subjects."""
    groups = sorted(table[key])
    both = np.array([np.concatenate([table[key][g], table[base][g]]) for g in groups])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return {
        "value": false_reduction(both.sum(axis=0)),
        "ci95": bootstrap(both, false_reduction, rng),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if isnan(value) or value in (inf, -inf) else float(value)
    return value


def report_domain(
    calibration: Table, test: Table, streams_per_recording: int
) -> tuple[dict[str, Any], dict[tuple[float, str], Key]]:
    domain: dict[str, Any] = {"operating_points": {}, "frontier": {}}
    chosen: dict[tuple[float, str], Key] = {}
    for budget in BUDGETS:
        entry = {}
        for variant in VARIANTS:
            key, feasible = select(calibration, variant, budget)
            chosen[(budget, variant)] = key
            entry[variant] = {
                **config_json(*key),
                "feasible_on_calibration": feasible,
                "calibration": summarize(calibration, key, streams_per_recording),
                "test": summarize(test, key, streams_per_recording),
                "false_reduction_vs_firmware_test": paired_reduction(
                    test, key, chosen[(budget, "firmware")]
                ),
            }
        domain["operating_points"][str(budget)] = entry
    domain["frontier"] = {v: frontier(test, v) for v in VARIANTS}
    default: Key = (1.0, Gate())
    merged: Table = {default: {**calibration[default], **test[default]}}
    domain["firmware_defaults_all_groups"] = summarize(merged, default, streams_per_recording)
    return domain, chosen


def nuisance_sweep(points: dict[str, Key]) -> dict[str, Any]:
    scales = sorted({scale for scale, _ in points.values()})
    gates = [gate for _, gate in points.values()]
    cache: dict[SimConfig, dict[str, dict[str, float]]] = {}

    def condition(cfg: SimConfig) -> dict[str, dict[str, float]]:
        if cfg not in cache:
            table = evaluate(sim_streams(cfg, SIM_TEST_SEEDS), scales, gates)
            cache[cfg] = {
                variant: {
                    "false_per_hour": false_per_hour(pooled(table, key)),
                    "missed_percent": missed_percent(pooled(table, key)),
                }
                for variant, key in points.items()
            }
        return cache[cfg]

    out: dict[str, Any] = {
        axis: {str(level): condition(replace(SimConfig(), **{axis: level})) for level in levels}
        for axis, levels in AXES.items()
    }
    out["no_artifacts"] = condition(
        replace(SimConfig(), jaw_per_min=0.0, movement_per_min=0.0, dropouts_per_min=0.0)
    )
    return out


def refire_fraction(streams: Iterable[Stream], scale: float) -> float:
    """Share of fired envelope excursions that fire twice (rise and fall)."""
    fired = refired = 0
    for stream in streams:
        n_fires = prepare(stream.env, stream.fs, BASE.scaled(scale)).n_fires
        fired, refired = fired + n_fires.size, refired + int((n_fires >= 2).sum())
    return refired / fired


def refire_scan(scale: float, fs: float = 512.0) -> dict[str, int]:
    """Fires caused by one isolated 170 ms blink of a given envelope peak."""
    out = {}
    for peak in (30.0, 45.0, 67.0, 90.0, 135.0):
        x = np.zeros(int(4 * fs))
        render_blinks(x, fs, [(1.0, 0.17, peak)])
        out[str(peak)] = len(blink_fires(eog_envelope(x, fs), fs, BASE.scaled(scale)))
    return out


def run_all(data_root: Path, out_dir: Path) -> dict[str, Any]:
    started, cpu = time.perf_counter(), time.process_time()
    stages: dict[str, dict[str, float]] = {}

    def timed(name: str, begin: tuple[float, float]) -> tuple[float, float]:
        now = (time.perf_counter(), time.process_time())
        stages[name] = {"wall_s": now[0] - begin[0], "cpu_s": now[1] - begin[1]}
        return now

    gates = [g for gs in gate_grid().values() for g in gs]
    calibration_subjects, test_subjects = split_subjects()
    mark = (started, cpu)
    sim_cal = evaluate(sim_streams(SimConfig(), SIM_CALIBRATION_SEEDS), SCALES, gates)
    sim_test = evaluate(sim_streams(SimConfig(), SIM_TEST_SEEDS), SCALES, gates)
    mark = timed("simulator_sweeps", mark)
    real_cal = evaluate(real_streams(data_root, calibration_subjects), SCALES, gates)
    real_test = evaluate(real_streams(data_root, test_subjects), SCALES, gates)
    mark = timed("real_background_sweeps", mark)

    sim, sim_chosen = report_domain(sim_cal, sim_test, 1)
    real, real_chosen = report_domain(real_cal, real_test, len(CHANNELS))
    real["sim_selected_on_real_test"] = {
        variant: {
            **config_json(*sim_chosen[(PRIMARY_BUDGET, variant)]),
            "test": summarize(real_test, sim_chosen[(PRIMARY_BUDGET, variant)], len(CHANNELS)),
        }
        for variant in VARIANTS
    }
    mark = timed("selection_and_bootstrap", mark)
    nuisance = nuisance_sweep({v: sim_chosen[(PRIMARY_BUDGET, v)] for v in VARIANTS})
    mark = timed("nuisance_sweep", mark)
    sim_scale = sim_chosen[(PRIMARY_BUDGET, "firmware")][0]
    real_scale = real_chosen[(PRIMARY_BUDGET, "firmware")][0]
    refire = {
        "isolated_blink_fires_by_envelope_peak": refire_scan(sim_scale),
        "scale_simulator": sim_scale,
        "scale_real": real_scale,
        "refire_fraction_simulator_test_all_excursions": refire_fraction(
            sim_streams(SimConfig(), SIM_TEST_SEEDS), sim_scale
        ),
        "refire_fraction_real_test_natural_only": refire_fraction(
            (s for s in real_streams(data_root, test_subjects) if not s.commands), real_scale
        ),
    }
    timed("refire_diagnostics", mark)

    metrics = {
        "meta": {
            "calibration_subjects": calibration_subjects,
            "test_subjects": test_subjects,
            "sim_calibration_seeds": SIM_CALIBRATION_SEEDS,
            "sim_test_seeds": SIM_TEST_SEEDS,
            "sim_config": asdict(SimConfig()),
            "scales": SCALES,
            "budgets_percent": BUDGETS,
            "channels": CHANNELS,
            "gates_in_grid": len(gates),
            "total_wall_s": time.perf_counter() - started,
            "total_cpu_s": time.process_time() - cpu,
            "stages": stages,
        },
        "simulator": sim,
        "real_background": real,
        "nuisance_sweep_simulator": nuisance,
        "double_fire_diagnostic": refire,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=1, allow_nan=False))
    return metrics
