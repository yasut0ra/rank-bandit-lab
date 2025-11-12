from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from .simulator import SimulationLog
from .types import Interaction


def interaction_to_dict(index: int, interaction: Interaction) -> Dict[str, Any]:
    return {
        "round": index,
        "slate": list(interaction.slate),
        "seen": list(interaction.seen),
        "click_index": interaction.click_index,
        "click_positions": list(interaction.click_positions),
        "reward": interaction.reward,
    }


def dict_to_interaction(data: Dict[str, Any]) -> Interaction:
    return Interaction(
        slate=tuple(data.get("slate", ())),
        seen=tuple(data.get("seen", ())),
        click_index=data.get("click_index"),
        reward=float(data.get("reward", 0.0)),
        click_positions=tuple(data.get("click_positions", ())),
    )


def serialize_log(
    log: SimulationLog,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    record = {
        "metadata": metadata or {},
        "optimal_reward": log.optimal_reward,
        "interactions": [
            interaction_to_dict(index, interaction)
            for index, interaction in enumerate(log.interactions, start=1)
        ],
    }
    return record


def write_log(
    path: str | Path,
    log: SimulationLog,
    metadata: Dict[str, Any] | None = None,
) -> None:
    payload = serialize_log(log, metadata)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def load_log(path: str | Path) -> Tuple[SimulationLog, Dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    interactions = [
        dict_to_interaction(item) for item in data.get("interactions", [])
    ]
    log = SimulationLog(interactions, optimal_reward=data.get("optimal_reward"))
    metadata = data.get("metadata", {})
    return log, metadata
