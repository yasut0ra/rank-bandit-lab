from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

from . import logging as log_utils
from .cli import (
    create_environment,
    create_policy,
    parse_documents,
)
from .compare import LogSummary, summaries_to_table
from .simulator import BanditSimulator
from .visualize import plot_learning_curves, plot_regret_curves


RUN_OVERRIDE_TYPES: Dict[str, callable] = {
    "algo": str,
    "epsilon": float,
    "alpha_prior": float,
    "beta_prior": float,
    "ucb_confidence": float,
    "seed": int,
    "steps": int,
    "slate_size": int,
    "model": str,
}


def parse_run_spec(spec: str) -> tuple[str, Dict[str, str]]:
    if ":" not in spec:
        raise ValueError(f"Run spec '{spec}' is missing label prefix (label:key=value,...).")
    label, remainder = spec.split(":", 1)
    if not label:
        raise ValueError("Run label cannot be empty.")
    overrides: Dict[str, str] = {}
    for token in remainder.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid token '{token}' in run spec '{spec}'. Expected key=value.")
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Missing key in run token '{token}'.")
        overrides[key] = value
    return label, overrides


def build_run_namespace(base: argparse.Namespace, overrides: Dict[str, str]) -> argparse.Namespace:
    config = {field: getattr(base, field) for field in BASE_FIELDS}
    for key, raw_value in overrides.items():
        if key not in RUN_OVERRIDE_TYPES:
            raise ValueError(f"Unsupported override '{key}'.")
        caster = RUN_OVERRIDE_TYPES[key]
        try:
            config[key] = caster(raw_value)
        except ValueError as exc:
            raise ValueError(f"Failed to parse override '{key}={raw_value}'.") from exc
    return argparse.Namespace(**config)


BASE_FIELDS = [
    "algo",
    "model",
    "steps",
    "slate_size",
    "epsilon",
    "alpha_prior",
    "beta_prior",
    "ucb_confidence",
    "doc",
    "position_biases",
    "doc_satisfaction",
    "default_satisfaction",
    "seed",
]


def run_sweep(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_specs = []
    for spec in args.run or []:
        label, overrides = parse_run_spec(spec)
        run_specs.append((label, overrides))
    if not run_specs:
        raise ValueError("At least one --run specification is required.")

    documents = parse_documents(args.doc)

    summaries: List[LogSummary] = []
    logs = []
    labels = []

    for label, overrides in run_specs:
        run_args = build_run_namespace(args, overrides)
        env = create_environment(run_args, documents)
        policy = create_policy(run_args, env.doc_ids)
        simulator = BanditSimulator(env, policy)
        log = simulator.run(run_args.steps)
        metadata = {
            "label": label,
            "algo": run_args.algo,
            "model": run_args.model,
            "steps": run_args.steps,
            "seed": run_args.seed,
            "doc_ids": list(env.doc_ids),
            "overrides": overrides,
        }
        log_path = output_dir / f"{label}.json"
        log_utils.write_log(log_path, log, metadata=metadata)

        summary = log.summary()
        log_summary = LogSummary(
            label=label,
            path=str(log_path),
            rounds=summary["rounds"],
            ctr=summary["ctr"],
            total_reward=summary["total_reward"],
            optimal_reward=log.optimal_reward,
            cumulative_regret=log.cumulative_regret(),
            algo=run_args.algo,
            model=run_args.model,
        )
        summaries.append(log_summary)
        logs.append(log)
        labels.append(label)

    key_map = {
        "ctr": lambda s: s.ctr,
        "regret": lambda s: float("inf") if s.cumulative_regret is None else s.cumulative_regret,
        "reward": lambda s: s.total_reward,
    }
    summaries.sort(key=key_map[args.sort_by], reverse=args.descending)
    print(summaries_to_table(summaries))

    if args.summary_json:
        payload = [asdict(summary) for summary in summaries]
        Path(args.summary_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.plot_learning or args.plot_regret or args.show_plot:
        if args.plot_learning or args.show_plot:
            plot_learning_curves(logs, labels, output_path=args.plot_learning, show=args.show_plot)
        if args.plot_regret or args.show_plot:
            plot_regret_curves(logs, labels, output_path=args.plot_regret, show=args.show_plot)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rank-bandit-lab-sweep",
        description="Run multiple ranking bandit configurations and record logs.",
    )
    parser.add_argument("--algo", default="epsilon", help="Default algorithm when a run does not override it.")
    parser.add_argument("--model", choices=("cascade", "position", "dependent"), default="cascade")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--slate-size", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha-prior", type=float, default=1.0)
    parser.add_argument("--beta-prior", type=float, default=1.0)
    parser.add_argument("--ucb-confidence", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--doc", action="append")
    parser.add_argument("--position-bias", action="append", dest="position_biases", type=float)
    parser.add_argument("--doc-satisfaction", action="append", dest="doc_satisfaction")
    parser.add_argument("--default-satisfaction", type=float, default=0.5)
    parser.add_argument(
        "--run",
        action="append",
        help="Run spec formatted as label:algo=epsilon,epsilon=0.05",
    )
    parser.add_argument("--output-dir", default="sweep_logs")
    parser.add_argument("--summary-json")
    parser.add_argument("--plot-learning", metavar="PATH")
    parser.add_argument("--plot-regret", metavar="PATH")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument(
        "--sort-by",
        choices=("ctr", "regret", "reward"),
        default="ctr",
        help="Metric used to sort the summary table.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_sweep(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
