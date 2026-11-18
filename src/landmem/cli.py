"""Command-line entry point: ``python -m landmem <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from landmem.config import DEFAULT_CONFIG, load_config


def _setup_logging(root: Path, name: str) -> None:
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(logs / f"{name}.log")]
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="landmem", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download the registered data selection")
    p_fetch.add_argument("steps", nargs="+",
                         choices=["metadata", "era5", "truth", "forecasts", "members"])
    p_fetch.add_argument("--workers", type=int, default=4)

    sub.add_parser("covariates", help="soil moisture anomalies, coupling regime, warm season")
    sub.add_parser("extract", help="afternoon and night forecast/truth panels")
    sub.add_parser("members", help="probabilistic scores from ensemble members at lead day 5")
    p_an = sub.add_parser("analyze", help="run the registered tests")
    p_an.add_argument("--skip-members", action="store_true")
    sub.add_parser("figures", help="draw the figures from saved results")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _setup_logging(cfg.root, args.command)

    if args.command == "fetch":
        from landmem.fetch import STEPS
        for step in args.steps:
            STEPS[step](cfg, args.workers)
    elif args.command == "covariates":
        from landmem.covariates import build_covariates
        build_covariates(cfg)
    elif args.command == "extract":
        from landmem.panels import build_panels
        build_panels(cfg)
    elif args.command == "members":
        from landmem.members import build_member_scores
        build_member_scores(cfg)
    elif args.command == "analyze":
        from landmem.analysis import run_analysis
        run_analysis(cfg, skip_members=args.skip_members)
    elif args.command == "figures":
        from landmem.figures import draw_figures
        draw_figures(cfg)
    return 0
