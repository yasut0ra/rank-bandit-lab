from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from . import logging as log_utils
from .simulator import SimulationLog
from .visualize import plot_learning_curves, plot_regret_curves


@dataclass(slots=True)
class LogSummary:
    label: str
    path: str
    rounds: int
    ctr: float
    total_reward: float
    optimal_reward: float | None
    cumulative_regret: float | None
    algo: str | None
    model: str | None


def summarize(path: str) -> tuple[LogSummary, SimulationLog, dict[str, Any]]:
    log, metadata = log_utils.load_log(path)
    summary = cast(Mapping[str, Any], log.summary())

    label = cast(str, metadata.get("label") or Path(path).stem)
    algo = cast(str | None, metadata.get("algo"))
    model = cast(str | None, metadata.get("model"))
    rounds = int(cast(float | int, summary["rounds"]))
    ctr = float(cast(float | int, summary["ctr"]))
    total_reward = float(cast(float | int, summary["total_reward"]))

    log_summary = LogSummary(
        label=label,
        path=str(path),
        rounds=rounds,
        ctr=ctr,
        total_reward=total_reward,
        optimal_reward=log.optimal_reward,
        cumulative_regret=log.cumulative_regret(),
        algo=algo,
        model=model,
    )
    return log_summary, log, metadata


def summaries_to_table(summaries: Sequence[LogSummary]) -> str:
    header = f"{'Label':20} {'Algo':10} {'Model':10} {'Rounds':>8} {'CTR':>8} {'Regret':>10}"
    lines = [header, "-" * len(header)]
    for item in summaries:
        regret = f"{item.cumulative_regret:.2f}" if item.cumulative_regret is not None else "-"
        lines.append(
            f"{item.label:20} {str(item.algo or '-'):10} {str(item.model or '-'):10} "
            f"{item.rounds:8d} {item.ctr:8.4f} {regret:>10}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rank-bandit-lab-compare",
        description="Compare multiple simulation logs generated via --log-json.",
    )
    parser.add_argument("logs", nargs="+", help="Paths to JSON log files.")
    parser.add_argument(
        "--sort-by",
        choices=("ctr", "regret", "reward"),
        default="ctr",
        help="Metric to sort summaries by.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order (ascending by default).",
    )
    parser.add_argument("--out-json", help="Write summary table to JSON file.")
    parser.add_argument(
        "--plot-learning",
        metavar="PATH",
        help="Save combined learning curve plot to PATH.",
    )
    parser.add_argument(
        "--plot-regret",
        metavar="PATH",
        help="Save combined regret curve plot to PATH.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display comparison plots.",
    )
    args = parser.parse_args(argv)

    loaded: list[tuple[LogSummary, SimulationLog, dict]] = []
    for log_path in args.logs:
        try:
            loaded.append(summarize(log_path))
        except (OSError, ValueError) as exc:
            parser.error(f"Failed to read '{log_path}': {exc}")

    summaries = [item[0] for item in loaded]
    logs = [item[1] for item in loaded]
    labels = [item[0].label for item in loaded]

    key_map = {
        "ctr": lambda s: s.ctr,
        "regret": lambda s: float("inf") if s.cumulative_regret is None else s.cumulative_regret,
        "reward": lambda s: s.total_reward,
    }
    summaries.sort(key=key_map[args.sort_by], reverse=args.descending)

    print(summaries_to_table(summaries))

    if args.out_json:
        payload = [summary.__dict__ for summary in summaries]
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.plot_learning or args.plot_regret or args.show_plot:
        try:
            if args.plot_learning or args.show_plot:
                plot_learning_curves(
                    logs,
                    labels,
                    output_path=args.plot_learning,
                    show=args.show_plot,
                )
            if args.plot_regret or args.show_plot:
                plot_regret_curves(
                    logs,
                    labels,
                    output_path=args.plot_regret,
                    show=args.show_plot,
                )
        except ImportError as exc:
            parser.error(str(exc))
        except ValueError as exc:
            parser.error(str(exc))
