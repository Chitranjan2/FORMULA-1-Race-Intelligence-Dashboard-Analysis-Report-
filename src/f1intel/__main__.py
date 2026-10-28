"""Command line entry point: ``python -m f1intel <command>``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="f1intel",
        description="Real-data Formula 1 race intelligence: ingest, analyse, simulate, model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="download and process OpenF1 + Jolpica data")
    build.add_argument("--seasons", default="2023,2024,2025,2026", help="comma-separated seasons")
    build.add_argument("--limit", type=int, default=None, help="max rounds per season (debugging)")

    sub.add_parser("analyze", help="compute pace, degradation and pit-loss tables")
    sub.add_parser("strategy", help="run the pit-window / undercut strategy engine")
    sub.add_parser("train", help="walk-forward validation and podium model artifact")
    sub.add_parser("report", help="render the static figures in outputs/")
    sub.add_parser("all", help="build -> analyze -> strategy -> train -> report")

    args = parser.parse_args(argv)

    from . import analysis, build as build_module, model, strategy

    if args.command == "build":
        seasons = tuple(int(s) for s in args.seasons.split(",") if s.strip())
        build_module.build(seasons, limit_per_season=args.limit)
    elif args.command == "analyze":
        analysis.run()
    elif args.command == "strategy":
        strategy.run()
    elif args.command == "train":
        model.run()
    elif args.command == "report":
        from . import report

        report.run()
    elif args.command == "all":
        seasons = tuple(int(s) for s in args.seasons.split(",") if s.strip())
        build_module.build(seasons)
        analysis.run()
        strategy.run()
        model.run()
        from . import report

        report.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
