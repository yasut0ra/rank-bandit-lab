from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from .environment import CascadeEnvironment
from .policies import RankingPolicy
from .types import Interaction


@dataclass(slots=True)
class SimulationLog:
    interactions: List[Interaction]

    @property
    def rounds(self) -> int:
        return len(self.interactions)

    @property
    def total_reward(self) -> float:
        return sum(event.reward for event in self.interactions)

    @property
    def ctr(self) -> float:
        if not self.interactions:
            return 0.0
        return self.total_reward / self.rounds

    def seen_counts(self) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for event in self.interactions:
            counter.update(event.seen)
        return dict(counter)

    def click_counts(self) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for event in self.interactions:
            counter.update(event.clicked_doc_ids)
        return dict(counter)

    def summary(self) -> Dict[str, object]:
        return {
            "rounds": self.rounds,
            "total_reward": self.total_reward,
            "ctr": self.ctr,
            "seen_counts": self.seen_counts(),
            "click_counts": self.click_counts(),
        }


class BanditSimulator:
    """Executes a ranking bandit policy inside an environment."""

    def __init__(self, env: CascadeEnvironment, policy: RankingPolicy) -> None:
        self._env = env
        self._policy = policy

    def run(self, rounds: int) -> SimulationLog:
        if rounds < 1:
            raise ValueError("rounds must be >= 1.")
        history: list[Interaction] = []
        for _ in range(rounds):
            slate = self._policy.select_slate()
            interaction = self._env.evaluate(slate)
            self._policy.update(interaction)
            history.append(interaction)
        return SimulationLog(history)
