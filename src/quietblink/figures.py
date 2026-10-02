"""Figures drawn from results/metrics.json; titles state the measured takeaway."""

from __future__ import annotations

import json
from math import inf
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quietblink.analysis import PRIMARY_BUDGET
from quietblink.detector import Params
from quietblink.experiments import VARIANTS, real_streams
from quietblink.gates import Gate, detect
from quietblink.metrics import match

COLORS = {"firmware": "#222222", "cooldown": "#56B4E9", "gate": "#009E73", "both": "#D55E00"}
MARKERS = {"firmware": "o", "cooldown": "s", "gate": "^", "both": "D"}
LABELS = {
    "firmware": "Firmware rule",
    "cooldown": "+ cooldown",
    "gate": "+ shape gate",
    "both": "+ both",
}
WINDOW_S = 30.0
DPI = 150


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def _gate(entry: dict[str, Any]) -> Gate:
    return Gate(**{k: (inf if v is None else v) for k, v in entry["gate"].items()})


def _change(reduction: float) -> str:
    return f"{-reduction * 100:+.0f}%"


def _points(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "simulator": metrics["simulator"]["operating_points"][str(PRIMARY_BUDGET)],
        "real_background": metrics["real_background"]["operating_points"][str(PRIMARY_BUDGET)],
    }


def tradeoff(metrics: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    points = _points(metrics)
    titles = {
        "simulator": f"Simulator, {len(metrics['meta']['sim_test_seeds'])} held-out seeds",
        "real_background": (
            f"Real forehead EEG, {len(metrics['meta']['test_subjects'])} held-out subjects"
        ),
    }
    for ax, domain in zip(axes, points, strict=True):
        for variant in VARIANTS:
            curve = metrics[domain]["frontier"][variant]
            ax.plot(
                [p["false_per_hour"] for p in curve],
                [p["missed_percent"] for p in curve],
                color=COLORS[variant],
                lw=1,
                alpha=0.8,
            )
            test = points[domain][variant]["test"]
            ax.plot(
                test["false_per_hour"],
                test["missed_percent"],
                MARKERS[variant],
                color=COLORS[variant],
                mec="white",
                ms=7,
                label=LABELS[variant],
            )
        ax.axhline(PRIMARY_BUDGET, color="gray", ls=":", lw=0.8)
        ax.set_xlim(0, 1.8 * max(p["test"]["false_per_hour"] for p in points[domain].values()))
        ax.set_ylim(0, 100)
        ax.set_title(titles[domain])
        ax.set_xlabel("False activations per hour (per channel)")
    axes[0].set_ylabel("Missed commands (%)")
    axes[0].legend(loc="upper right", title="markers: calibration-chosen points", fontsize=7)
    sim = points["simulator"]["both"]["false_reduction_vs_firmware_test"]["value"]
    real = points["real_background"]["both"]["false_reduction_vs_firmware_test"]["value"]
    fig.suptitle(
        f"At a {PRIMARY_BUDGET:.0f}% missed budget, cooldown + shape gate changes false "
        f"activations\nby {_change(sim)} in simulation and {_change(real)} on held-out real "
        "subjects",
        fontsize=9,
    )
    fig.savefig(out / "tradeoff.png")
    plt.close(fig)


def false_activations(metrics: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    points = _points(metrics)
    sim_min = points["simulator"]["firmware"]["test"]["negative_minutes"]
    real_min = points["real_background"]["firmware"]["test"]["negative_minutes"]
    minutes = {
        "simulator": f"{sim_min:.0f} simulated min",
        "real_background": f"{real_min:.1f} real negative min per channel",
    }
    for ax, domain in zip(axes, points, strict=True):
        for i, variant in enumerate(VARIANTS):
            test = points[domain][variant]["test"]
            lo, hi = test["false_per_hour_ci95"]
            value = test["false_per_hour"]
            ax.bar(i, value, color=COLORS[variant], width=0.65)
            ax.errorbar(i, value, yerr=[[value - lo], [hi - value]], color="black", capsize=3, lw=1)
            ax.annotate(
                f"{value:.0f}",
                (i, hi),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels(
            [
                f"{LABELS[v]}\nmissed {points[domain][v]['test']['missed_percent']:.0f}%"
                for v in VARIANTS
            ],
            fontsize=7,
        )
        ax.set_title(
            f"{'Simulator' if domain == 'simulator' else 'Real forehead EEG'}: {minutes[domain]}"
        )
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("False activations per hour (95% CI, bootstrap)")
    sim_cut = points["simulator"]["both"]["false_reduction_vs_firmware_test"]["value"]
    real_cut = points["real_background"]["both"]["false_reduction_vs_firmware_test"]
    low, high = real_cut["ci95"]
    fig.suptitle(
        f"Cooldown + shape gate changes false activations by {_change(sim_cut)} in simulation\n"
        f"and {_change(real_cut['value'])} on held-out subjects (paired 95% CI {_change(high)} to "
        f"{_change(low)})",
        fontsize=9,
    )
    fig.savefig(out / "false_activations.png")
    plt.close(fig)


def _example_window(
    root: Path, subjects: list[int], firmware: tuple[float, Gate]
) -> tuple[int, Any, int]:
    """First 30 s window of a held-out injected stream with a command and a firmware false event."""
    for subject in subjects:
        for stream in real_streams(root, [subject]):
            if not stream.commands:
                continue
            events = detect(stream.env, stream.fs, Params().scaled(firmware[0]), firmware[1])
            false = match(events, stream.commands).false_events
            for k in range(int(stream.x.size / stream.fs // WINDOW_S)):
                lo_ms, hi_ms = k * WINDOW_S * 1000, (k + 1) * WINDOW_S * 1000
                has_command = any(lo_ms <= c.onsets_ms[0] < hi_ms - 2000 for c in stream.commands)
                if has_command and any(lo_ms <= e.ms < hi_ms for e in false):
                    return subject, stream, k
    raise RuntimeError("no example window found")


def trace(metrics: dict[str, Any], root: Path, out: Path) -> None:
    points = _points(metrics)["real_background"]
    configs = {v: (points[v]["scale"], _gate(points[v])) for v in ("firmware", "both")}
    subjects = metrics["meta"]["test_subjects"]
    subject, stream, k = _example_window(root, subjects, configs["firmware"])
    t0 = k * WINDOW_S
    sl = slice(int(t0 * stream.fs), int((t0 + WINDOW_S) * stream.fs))
    t = np.arange(sl.start, sl.stop) / stream.fs - t0
    fig, (top, mid, strip) = plt.subplots(
        3, 1, figsize=(7.2, 4.4), sharex=True, height_ratios=(1, 1.2, 0.55)
    )
    top.plot(t, stream.x[sl] - np.median(stream.x[sl]), color="gray", lw=0.5)
    top.set_ylabel("Signal (units)")
    mid.plot(t, stream.env[sl], color="black", lw=0.8)
    mid.set_ylabel("Envelope (units)")
    for variant, (scale, _) in configs.items():
        mid.axhline(Params().scaled(scale).blink_lower, color=COLORS[variant], ls=":", lw=1)
    for command in stream.commands:
        if t0 * 1000 <= command.onsets_ms[0] < (t0 + WINDOW_S) * 1000:
            span = (command.onsets_ms[0] / 1000 - t0, command.onsets_ms[-1] / 1000 - t0 + 0.3)
            for ax in (top, mid, strip):
                ax.axvspan(*span, color="#0072B2", alpha=0.15)
    false_counts = {}
    for row, variant in enumerate(configs):
        scale, gate = configs[variant]
        events = detect(stream.env, stream.fs, Params().scaled(scale), gate)
        false = set(match(events, stream.commands).false_events)
        shown = [e for e in events if t0 * 1000 <= e.ms < (t0 + WINDOW_S) * 1000]
        false_counts[variant] = sum(e in false for e in shown)
        for e in shown:
            strip.plot(
                e.ms / 1000 - t0,
                1 - row,
                MARKERS[variant],
                color=COLORS[variant],
                mfc="none" if e in false else COLORS[variant],
                ms=7,
            )
            strip.text(e.ms / 1000 - t0, 1 - row + 0.35, f"x{e.kind}", ha="center", fontsize=6)
    strip.set_yticks([1, 0], [LABELS["firmware"], LABELS["both"]])
    strip.set_ylim(-0.6, 1.8)
    strip.grid(visible=False)
    strip.set_xlabel(
        "Time in window (s); blue = injected command; filled = recovered, hollow = false"
    )
    fig.suptitle(
        f"30 s of held-out subject {subject}: false activations, firmware rule "
        f"{false_counts['firmware']}, with gating {false_counts['both']}",
        fontsize=9,
    )
    fig.savefig(out / "trace.png")
    plt.close(fig)


def nuisance(metrics: dict[str, Any], out: Path) -> None:
    sweep = metrics["nuisance_sweep_simulator"]
    axes_labels = {
        "jaw_per_min": "Jaw bursts per min",
        "movement_per_min": "Baseline-step events per min",
        "noise_scale": "Background noise (x)",
        "deliberate_gain": "Deliberate / natural blink amplitude",
    }
    fig, grid = plt.subplots(2, 4, figsize=(7.2, 4.2), sharey="row")
    for col, (axis, label) in enumerate(axes_labels.items()):
        levels = sorted(sweep[axis], key=float)
        for row, metric in enumerate(("false_per_hour", "missed_percent")):
            ax = grid[row, col]
            for variant in VARIANTS:
                ax.plot(
                    [float(v) for v in levels],
                    [sweep[axis][v][variant][metric] for v in levels],
                    MARKERS[variant] + "-",
                    color=COLORS[variant],
                    ms=4,
                    lw=1,
                    label=LABELS[variant],
                )
            if row == 1:
                ax.set_xlabel(label, fontsize=7)
    grid[0, 0].set_ylabel("False activations per hour")
    grid[1, 0].set_ylabel("Missed commands (%)")
    grid[0, 0].legend(fontsize=6)
    jaw = sweep["jaw_per_min"]
    top = max(jaw, key=float)
    firmware_rate, gated_rate = (jaw[top][v]["false_per_hour"] for v in ("firmware", "both"))
    fig.suptitle(
        f"Operating points fixed at default nuisance: at {top} jaw bursts/min the firmware rule "
        f"fires {firmware_rate:.0f}/h, gated {gated_rate:.0f}/h (simulator)",
        fontsize=9,
    )
    fig.savefig(out / "nuisance.png")
    plt.close(fig)


def make_figures(metrics_path: Path, data_root: Path, out: Path) -> None:
    _style()
    metrics = json.loads(metrics_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    tradeoff(metrics, out)
    false_activations(metrics, out)
    trace(metrics, data_root, out)
    nuisance(metrics, out)
