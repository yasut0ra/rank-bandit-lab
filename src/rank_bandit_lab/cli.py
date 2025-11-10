from __future__ import annotations

import argparse
from random import Random
from typing import Sequence

from . import visualize
from .environment import (
    CascadeEnvironment,
    DependentClickEnvironment,
    PositionBasedEnvironment,
)
from .policies import EpsilonGreedyRanking, RankingPolicy, ThompsonSamplingRanking
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
    parser.add_argument("--algo", choices=("epsilon", "thompson"), default="epsilon")
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


def print_summary(summary: dict[str, object], doc_ids: Sequence[str], log) -> None:
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
    for doc_id in doc_ids:
        count = seen_counts.get(doc_id, 0)
        print(f"  {doc_id:>8} -> {count}")
    print("Click counts :")
    click_counts = summary["click_counts"]
    for doc_id in doc_ids:
        count = click_counts.get(doc_id, 0)
        print(f"  {doc_id:>8} -> {count}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        documents = parse_documents(args.doc)
        env = create_environment(args, documents)
    except ValueError as exc:
        parser.error(str(exc))
    policy = create_policy(args, env.doc_ids)
    simulator = BanditSimulator(env, policy)
    log = simulator.run(args.steps)
    summary = log.summary()
    print_summary(summary, env.doc_ids, log)
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
                    doc_ids=env.doc_ids,
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
