from __future__ import annotations

import argparse
from random import Random
from typing import Iterable, Sequence

from .environment import CascadeEnvironment
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rank-bandit-lab",
        description="Run simple ranking bandit simulations.",
    )
    parser.add_argument("--algo", choices=("epsilon", "thompson"), default="epsilon")
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
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
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


def print_summary(summary: dict[str, object], doc_ids: Sequence[str]) -> None:
    print(f"Rounds       : {summary['rounds']}")
    print(f"Total reward : {summary['total_reward']:.0f}")
    print(f"CTR          : {summary['ctr']:.4f}")
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
    except ValueError as exc:
        parser.error(str(exc))
    env_rng = Random(args.seed)
    env = CascadeEnvironment(documents=documents, slate_size=args.slate_size, rng=env_rng)
    policy = create_policy(args, env.doc_ids)
    simulator = BanditSimulator(env, policy)
    log = simulator.run(args.steps)
    summary = log.summary()
    print_summary(summary, env.doc_ids)


if __name__ == "__main__":
    main()
