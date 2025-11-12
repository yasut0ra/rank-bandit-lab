from __future__ import annotations

import argparse
from random import Random
from typing import Sequence

from . import logging as log_utils
from . import scenario_loader
from . import visualize
from .environment import (
    CascadeEnvironment,
    DependentClickEnvironment,
    PositionBasedEnvironment,
)
from .policies import (
    EpsilonGreedyRanking,
    RankingPolicy,
    SoftmaxRanking,
    ThompsonSamplingRanking,
    UCB1Ranking,
)
from .simulator import BanditSimulator
from .types import Document

DEFAULT_DOCUMENTS = (
    Document("doc-A", 0.45),
    Document("doc-B", 0.35),
    Document("doc-C", 0.25),
    Document("doc-D", 0.15),
    Document("doc-E", 0.10),
)
DEFAULT_POSITION_BIASES = (0.95, 0.85, 0.75, 0.65, 0.55)
DEFAULT_SATISFACTION = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rank-bandit-lab",
        description="Run simple ranking bandit simulations.",
    )
    parser.add_argument(
        "--algo",
        choices=("epsilon", "thompson", "ucb", "softmax"),
        default="epsilon",
    )
    parser.add_argument(
        "--model",
        choices=("cascade", "position", "dependent"),
        default="cascade",
        help="Click model / environment to simulate.",
    )
    parser.add_argument("--steps", type=int, default=2_000, help="Number of interaction rounds.")
    parser.add_argument(
        "--slate-size",
        type=int,
        default=3,
        help="Number of documents shown per round.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate for epsilon-greedy.",
    )
    parser.add_argument(
        "--alpha-prior",
        type=float,
        default=1.0,
        help="Alpha prior for Thompson sampling.",
    )
    parser.add_argument(
        "--beta-prior",
        type=float,
        default=1.0,
        help="Beta prior for Thompson sampling.",
    )
    parser.add_argument(
        "--ucb-confidence",
        type=float,
        default=1.0,
        help="Exploration scale for UCB1 (larger -> more exploration).",
    )
    parser.add_argument(
        "--softmax-temp",
        type=float,
        default=0.1,
        help="Temperature for softmax/Boltzmann exploration.",
    )
    parser.add_argument(
        "--doc",
        action="append",
        help="Document specification formatted as 'doc_id=probability'.",
    )
    parser.add_argument(
        "--position-bias",
        action="append",
        dest="position_biases",
        type=float,
        help="(PBM) Examination probability per rank position.",
    )
    parser.add_argument(
        "--doc-satisfaction",
        action="append",
        dest="doc_satisfaction",
        help="(DCM) Satisfaction probability formatted 'doc_id=probability'.",
    )
    parser.add_argument(
        "--default-satisfaction",
        type=float,
        default=DEFAULT_SATISFACTION,
        help="(DCM) Fallback satisfaction probability when not specified per doc.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument(
        "--scenario",
        choices=scenario_loader.list_scenarios(),
        help="Use a predefined scenario (overrides docs/bias defaults).",
    )
    parser.add_argument(
        "--plot-learning",
        metavar="PATH",
        help="Save learning curve plot to PATH (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-docs",
        metavar="PATH",
        help="Save seen/click distribution plot to PATH (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-regret",
        metavar="PATH",
        help="Save regret curve plot to PATH (requires matplotlib and environments with optimal reward support).",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display plots interactively (GUI/Matplotlib backend required).",
    )
    parser.add_argument(
        "--log-json",
        metavar="PATH",
        help="Write full interaction log to PATH as JSON.",
    )
    parser.add_argument(
        "--load-json",
        metavar="PATH",
        help="Load a previously saved simulation log (skips new simulation).",
    )
    return parser


def parse_documents(raw: Sequence[str] | None) -> Sequence[Document]:
    if not raw:
        return DEFAULT_DOCUMENTS
    documents: list[Document] = []
    for spec in raw:
        if "=" not in spec:
            raise ValueError(f"Invalid document spec '{spec}'. Expected 'id=prob'.")
        name, raw_prob = spec.split("=", 1)
        try:
            probability = float(raw_prob)
        except ValueError as exc:
            raise ValueError(f"Invalid probability '{raw_prob}' in '{spec}'.") from exc
        name = name.strip()
        if not name:
            raise ValueError(f"Document id missing in specification '{spec}'.")
        documents.append(Document(name, probability))
    return documents


def parse_probability_mapping(raw: Sequence[str] | None, label: str) -> dict[str, float]:
    if not raw:
        return {}
    mapping: dict[str, float] = {}
    for spec in raw:
        if "=" not in spec:
            raise ValueError(f"Invalid {label} spec '{spec}'. Expected 'id=value'.")
        name, raw_val = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"{label.title()} id missing in specification '{spec}'.")
        try:
            value = float(raw_val)
        except ValueError as exc:
            raise ValueError(f"Invalid probability '{raw_val}' in '{spec}'.") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{label.title()} probability for '{name}' must be in [0, 1], got {value}."
            )
        mapping[name] = value
    return mapping


def create_policy(args: argparse.Namespace, doc_ids: Sequence[str]) -> RankingPolicy:
    policy_rng = Random(args.seed + 1)
    if args.algo == "epsilon":
        return EpsilonGreedyRanking(
            doc_ids=doc_ids,
            slate_size=args.slate_size,
            epsilon=args.epsilon,
            rng=policy_rng,
        )
    if args.algo == "thompson":
        return ThompsonSamplingRanking(
            doc_ids=doc_ids,
            slate_size=args.slate_size,
            alpha_prior=args.alpha_prior,
            beta_prior=args.beta_prior,
            rng=policy_rng,
        )
    if args.algo == "ucb":
        return UCB1Ranking(
            doc_ids=doc_ids,
            slate_size=args.slate_size,
            confidence=args.ucb_confidence,
            rng=policy_rng,
        )
    if args.algo == "softmax":
        return SoftmaxRanking(
            doc_ids=doc_ids,
            slate_size=args.slate_size,
            temperature=args.softmax_temp,
            rng=policy_rng,
        )
    raise ValueError(f"Unsupported algorithm: {args.algo}")


def create_environment(args: argparse.Namespace, documents: Sequence[Document]):
    env_rng = Random(args.seed)
    if args.model == "cascade":
        return CascadeEnvironment(documents=documents, slate_size=args.slate_size, rng=env_rng)
    if args.model == "position":
        biases = args.position_biases or DEFAULT_POSITION_BIASES
        if len(biases) < args.slate_size:
            raise ValueError("Provide at least slate-size position biases for PBM.")
        return PositionBasedEnvironment(
            documents=documents,
            slate_size=args.slate_size,
            position_biases=biases,
            rng=env_rng,
        )
    if args.model == "dependent":
        satisfaction = parse_probability_mapping(args.doc_satisfaction, "satisfaction")
        return DependentClickEnvironment(
            documents=documents,
            slate_size=args.slate_size,
            satisfaction=satisfaction,
            default_satisfaction=args.default_satisfaction,
            rng=env_rng,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def print_summary(summary: dict[str, object], doc_ids: Sequence[str] | None, log) -> None:
    print(f"Rounds       : {summary['rounds']}")
    print(f"Total reward : {summary['total_reward']:.2f}")
    print(f"CTR          : {summary['ctr']:.4f}")
    if log.optimal_reward is not None:
        print(f"Optimal reward (per round): {log.optimal_reward:.4f}")
        cumulative_regret = log.cumulative_regret()
        if cumulative_regret is not None:
            print(f"Cumulative regret        : {cumulative_regret:.4f}")
    print("Seen counts  :")
    seen_counts = summary["seen_counts"]
    ordered_ids = list(doc_ids) if doc_ids else list(seen_counts.keys())
    for doc_id in ordered_ids:
        count = seen_counts.get(doc_id, 0)
        print(f"  {doc_id:>8} -> {count}")
    print("Click counts :")
    click_counts = summary["click_counts"]
    for doc_id in ordered_ids:
        count = click_counts.get(doc_id, 0)
        print(f"  {doc_id:>8} -> {count}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.load_json and args.log_json:
        parser.error("--log-json cannot be combined with --load-json.")

    metadata: dict[str, object] = {}
    doc_ids: Sequence[str] | None = None
    scenario_data = None
    scenario_doc_specs: Sequence[str] | None = None
    if args.scenario:
        scenario_data = scenario_loader.load_scenario(args.scenario)
        scenario_doc_specs = [
            f"{entry['id']}={entry['attraction']}"
            for entry in scenario_data.get("documents", [])
        ]
    if args.load_json:
        try:
            log, metadata = log_utils.load_log(args.load_json)
        except (OSError, ValueError) as exc:
            parser.error(f"Failed to load log: {exc}")
        doc_ids = metadata.get("doc_ids")
    else:
        doc_specs = args.doc if args.doc else scenario_doc_specs
        try:
            documents = parse_documents(doc_specs)
        except ValueError as exc:
            parser.error(str(exc))
        env_args = argparse.Namespace(**vars(args))
        if scenario_data:
            if env_args.position_biases is None and scenario_data.get("position_biases"):
                env_args.position_biases = list(scenario_data["position_biases"])
            if env_args.doc_satisfaction is None and scenario_data.get("satisfaction"):
                env_args.doc_satisfaction = [
                    f"{doc_id}={prob}"
                    for doc_id, prob in scenario_data["satisfaction"].items()
                ]
        try:
            env = create_environment(env_args, documents)
        except ValueError as exc:
            parser.error(str(exc))
        policy = create_policy(env_args, env.doc_ids)
        simulator = BanditSimulator(env, policy)
        log = simulator.run(env_args.steps)
        metadata = {
            "doc_ids": list(env.doc_ids),
            "model": env_args.model,
            "algo": env_args.algo,
            "steps": env_args.steps,
            "seed": env_args.seed,
            "scenario": args.scenario,
        }
        if args.log_json:
            try:
                log_utils.write_log(args.log_json, log, metadata=metadata)
            except OSError as exc:
                parser.error(f"Failed to write log: {exc}")
        doc_ids = env.doc_ids

    summary = log.summary()
    doc_ids_for_usage = doc_ids or metadata.get("doc_ids")
    print_summary(summary, doc_ids_for_usage, log)
    if args.plot_learning or args.plot_docs or args.plot_regret or args.show_plot:
        try:
            if args.plot_learning or args.show_plot:
                visualize.plot_learning_curve(
                    log,
                    output_path=args.plot_learning,
                    show=args.show_plot,
                )
            if args.plot_docs:
                visualize.plot_doc_distribution(
                    log,
                    doc_ids=doc_ids_for_usage or tuple(summary["seen_counts"].keys()),
                    output_path=args.plot_docs,
                    show=args.show_plot,
                )
            if args.plot_regret or args.show_plot:
                visualize.plot_regret_curve(
                    log,
                    output_path=args.plot_regret,
                    show=args.show_plot,
                )
        except ImportError as exc:
            parser.error(str(exc))
        except ValueError as exc:
            parser.error(str(exc))


if __name__ == "__main__":
    main()
