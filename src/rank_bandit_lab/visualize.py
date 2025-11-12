from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Sequence, cast

from .simulator import SimulationLog


@dataclass(slots=True)
class DocDistributionData:
    doc_ids: List[str]
    seen: List[float]
    clicks: List[float]


def learning_curve_data(log: SimulationLog) -> Dict[str, List[float]]:
    """Return per-round metrics for plotting learning curves."""

    metrics = log.round_metrics()
    rounds = [float(item.round_index) for item in metrics]
    reward = [float(item.reward) for item in metrics]
    cumulative = [float(item.cumulative_reward) for item in metrics]
    ctr = [float(item.ctr) for item in metrics]
    data: Dict[str, List[float]] = {
        "rounds": rounds,
        "reward": reward,
        "cumulative_reward": cumulative,
        "ctr": ctr,
    }
    if log.optimal_reward is not None:
        data["optimal_reward"] = [log.optimal_reward] * len(rounds)
    return data


def doc_distribution_data(log: SimulationLog, doc_ids: Sequence[str]) -> DocDistributionData:
    """Aggregate seen/click counts per document for plotting."""

    summary: Mapping[str, Any] = log.summary()
    seen_counts = cast(Mapping[str, int], summary["seen_counts"])
    click_counts = cast(Mapping[str, int], summary["click_counts"])
    ordered_ids = list(doc_ids)
    seen = [float(seen_counts.get(doc_id, 0)) for doc_id in ordered_ids]
    clicks = [float(click_counts.get(doc_id, 0)) for doc_id in ordered_ids]
    return DocDistributionData(doc_ids=ordered_ids, seen=seen, clicks=clicks)


def regret_curve_data(log: SimulationLog) -> Dict[str, List[float]]:
    if log.optimal_reward is None:
        raise ValueError("Regret data requires environments that expose optimal reward information.")
    metrics = log.round_metrics()
    rounds = [float(item.round_index) for item in metrics]
    instant = [float(item.instant_regret or 0.0) for item in metrics]
    cumulative = [float(item.cumulative_regret or 0.0) for item in metrics]
    return {
        "rounds": rounds,
        "instant_regret": instant,
        "cumulative_regret": cumulative,
    }


def plot_learning_curve(
    log: SimulationLog,
    output_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot cumulative reward & CTR progression."""

    data = learning_curve_data(log)
    if not data["rounds"]:
        raise ValueError("SimulationLog is empty; cannot plot learning curve.")
    plt = _require_matplotlib()
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(data["rounds"], data["cumulative_reward"], label="Cumulative Reward", color="tab:blue")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Cumulative Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(data["rounds"], data["ctr"], label="CTR", color="tab:orange")
    ax2.set_ylabel("CTR", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    if "optimal_reward" in data:
        ax1.axhline(y=data["optimal_reward"][0], color="tab:green", linestyle="--", label="Optimal Reward")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines + lines2, labels + labels2, loc="upper center", ncol=2)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show and not output_path:
        plt.show()
    elif show and output_path:
        plt.show()
    plt.close(fig)


def plot_learning_curves(
    logs: Sequence[SimulationLog],
    labels: Sequence[str],
    output_path: str | None = None,
    show: bool = False,
) -> None:
    if not logs:
        raise ValueError("No logs provided for learning curve comparison.")
    plt = _require_matplotlib()
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    colors = _color_cycle()
    for log, label in zip(logs, labels, strict=True):
        data = learning_curve_data(log)
        color = next(colors)
        ax1.plot(
            data["rounds"],
            data["cumulative_reward"],
            label=f"{label} cumulative",
            color=color,
        )
        ax2.plot(
            data["rounds"],
            data["ctr"],
            label=f"{label} CTR",
            linestyle="--",
            color=color,
        )
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Cumulative Reward")
    ax2.set_ylabel("CTR")
    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines + lines2, labels1 + labels2, loc="upper center", ncol=2)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_doc_distribution(
    log: SimulationLog,
    doc_ids: Sequence[str],
    output_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot bar chart comparing seen vs click counts per document."""

    data = doc_distribution_data(log, doc_ids)
    if not data.doc_ids:
        raise ValueError("No document ids provided; cannot plot distribution.")
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = list(range(len(data.doc_ids)))
    width = 0.4
    seen_pos = [pos - width / 2 for pos in positions]
    click_pos = [pos + width / 2 for pos in positions]
    ax.bar(seen_pos, data.seen, width=width, label="Seen", color="tab:gray")
    ax.bar(click_pos, data.clicks, width=width, label="Clicks", color="tab:green")
    ax.set_xticks(positions)
    ax.set_xticklabels(data.doc_ids)
    ax.set_ylabel("Count")
    ax.set_xlabel("Document")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show and not output_path:
        plt.show()
    elif show and output_path:
        plt.show()
    plt.close(fig)


def plot_regret_curve(
    log: SimulationLog,
    output_path: str | None = None,
    show: bool = False,
) -> None:
    data = regret_curve_data(log)
    if not data["rounds"]:
        raise ValueError("SimulationLog is empty; cannot plot regret curve.")
    plt = _require_matplotlib()
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(data["rounds"], data["cumulative_regret"], label="Cumulative Regret", color="tab:red")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Cumulative Regret", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(data["rounds"], data["instant_regret"], label="Instant Regret", color="tab:purple", alpha=0.6)
    ax2.set_ylabel("Instant Regret", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines + lines2, labels + labels2, loc="upper center", ncol=2)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_regret_curves(
    logs: Sequence[SimulationLog],
    labels: Sequence[str],
    output_path: str | None = None,
    show: bool = False,
) -> None:
    if not logs:
        raise ValueError("No logs provided for regret comparison.")
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = _color_cycle()
    for log, label in zip(logs, labels, strict=True):
        data = regret_curve_data(log)
        ax.plot(
            data["rounds"],
            data["cumulative_regret"],
            label=label,
            color=next(colors),
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Regret")
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _require_matplotlib() -> Any:
    try:
        matplotlib = importlib.import_module("matplotlib")
        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg", force=True)
        plt = importlib.import_module("matplotlib.pyplot")
        return plt
    except Exception as exc:  # pragma: no cover - depends on matplotlib presence
        raise ImportError(
            "matplotlib is required for plotting. Install it via 'pip install matplotlib'."
        ) from exc


def _color_cycle() -> Iterator[str]:
    from itertools import cycle

    palette = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    return cycle(palette)
