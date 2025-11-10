from __future__ import annotations

import importlib
import sys
from typing import Dict, Sequence

from .simulator import SimulationLog


def learning_curve_data(log: SimulationLog) -> Dict[str, list[float]]:
    """Return per-round metrics for plotting learning curves."""

    metrics = log.round_metrics()
    rounds = [item.round_index for item in metrics]
    reward = [item.reward for item in metrics]
    cumulative = [item.cumulative_reward for item in metrics]
    ctr = [item.ctr for item in metrics]
    return {
        "rounds": rounds,
        "reward": reward,
        "cumulative_reward": cumulative,
        "ctr": ctr,
    }


def doc_distribution_data(log: SimulationLog, doc_ids: Sequence[str]) -> Dict[str, list[float]]:
    """Aggregate seen/click counts per document for plotting."""

    summary = log.summary()
    seen_counts = summary["seen_counts"]
    click_counts = summary["click_counts"]
    ordered_ids = list(doc_ids)
    seen = [seen_counts.get(doc_id, 0) for doc_id in ordered_ids]
    clicks = [click_counts.get(doc_id, 0) for doc_id in ordered_ids]
    return {
        "doc_ids": ordered_ids,
        "seen": seen,
        "clicks": clicks,
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


def plot_doc_distribution(
    log: SimulationLog,
    doc_ids: Sequence[str],
    output_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot bar chart comparing seen vs click counts per document."""

    data = doc_distribution_data(log, doc_ids)
    if not data["doc_ids"]:
        raise ValueError("No document ids provided; cannot plot distribution.")
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = list(range(len(data["doc_ids"])))
    width = 0.4
    seen_pos = [pos - width / 2 for pos in positions]
    click_pos = [pos + width / 2 for pos in positions]
    ax.bar(seen_pos, data["seen"], width=width, label="Seen", color="tab:gray")
    ax.bar(click_pos, data["clicks"], width=width, label="Clicks", color="tab:green")
    ax.set_xticks(positions)
    ax.set_xticklabels(data["doc_ids"])
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


def _require_matplotlib():
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
