from __future__ import annotations

import argparse
from pathlib import Path

from quietblink.analysis import run_all
from quietblink.figures import make_figures


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="quietblink", description="Run every experiment and write results/."
    )
    parser.add_argument("--data", type=Path, required=True, help="directory with S001R06.edf etc.")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_all(args.data, args.out)
    make_figures(args.out / "metrics.json", args.data, args.out / "figures")


if __name__ == "__main__":
    main()
