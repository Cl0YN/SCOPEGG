"""CLI пайплайна: python -m utility_coach.run {all|ingest|...|report} [--limit N]."""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from collections.abc import Callable

STEPS: list[str] = ["ingest", "zones", "features", "model", "lineups", "rating", "findings", "advise", "report"]

RUNNERS: dict[str, tuple[str, str]] = {
    "ingest": ("utility_coach.ingest", "run_ingest"),
    "zones": ("utility_coach.zones", "run_zones"),
    "features": ("utility_coach.features", "run_features"),
    "model": ("utility_coach.model", "run_model"),
    "lineups": ("utility_coach.lineups", "run_lineups"),
    "rating": ("utility_coach.stats", "run_rating"),
    "findings": ("utility_coach.findings", "run_findings"),
    "advise": ("utility_coach.advisor", "run_advise"),
    "report": ("utility_coach.report", "build_report"),
}


def _resolve(step: str) -> Callable:
    module, name = RUNNERS[step]
    return getattr(importlib.import_module(module), name)


def run_step(step: str, limit: int | None = None):
    fn = _resolve(step)
    print(f"[{step}] старт", flush=True)
    t0 = time.perf_counter()
    result = fn(limit=limit) if step == "ingest" else fn()
    print(f"[{step}] готово за {time.perf_counter() - t0:.1f} с", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m utility_coach.run",
                                     description="Пайплайн Utility Coach для Cache")
    parser.add_argument("step", choices=["all", *STEPS], help="шаг пайплайна или all — все по порядку")
    parser.add_argument("--limit", type=int, default=None,
                        help="только для ingest: прочитать первые N матчей (для быстрой проверки)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    steps = STEPS if args.step == "all" else [args.step]
    for step in steps:
        result = run_step(step, limit=args.limit)
        if step == "report" and result is not None:
            print(f"Отчёт: {result}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
