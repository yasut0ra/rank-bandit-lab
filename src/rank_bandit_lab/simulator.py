from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Tuple

from .policies import RankingPolicy
from .types import Interaction


class RankingEnvironment(Protocol):
    doc_ids: Tuple[str, ...]

    def evaluate(self, slate: Sequence[str]) -> Interaction:
        ...

    def optimal_slate(self) -> Tuple[str, ...]:
        ...

    def expected_reward(self, slate: Sequence[str]) -> float:
        ...


@dataclass(slots=True)
class SimulationLog:
    interactions: List[Interaction]
    optimal_reward: float | None = None

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

    def round_metrics(self) -> List["RoundMetrics"]:
        total = 0.0
        metrics: list[RoundMetrics] = []
        optimal = self.optimal_reward
        cumulative_regret = 0.0
        for index, event in enumerate(self.interactions, start=1):
            total += event.reward
            instant_regret = None
            if optimal is not None:
                instant_regret = optimal - event.reward
                cumulative_regret += instant_regret
            metrics.append(
                RoundMetrics(
                    round_index=index,
                    reward=event.reward,
                    cumulative_reward=total,
                    ctr=(total / index),
                    instant_regret=instant_regret,
                    cumulative_regret=(cumulative_regret if optimal is not None else None),
                )
            )
        return metrics

    def cumulative_regret(self) -> float | None:
        if self.optimal_reward is None:
            return None
        return self.rounds * self.optimal_reward - self.total_reward


@dataclass(frozen=True, slots=True)
class RoundMetrics:
    round_index: int
    reward: float
    cumulative_reward: float
    ctr: float
    instant_regret: float | None
    cumulative_regret: float | None


class BanditSimulator:
    """Executes a ranking bandit policy inside an environment."""

    def __init__(self, env: RankingEnvironment, policy: RankingPolicy) -> None:
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
        optimal_reward = _infer_optimal_reward(self._env)
        return SimulationLog(history, optimal_reward=optimal_reward)


def _infer_optimal_reward(env: RankingEnvironment) -> float | None:
    optimal_reward = None
    optimal_slate = getattr(env, "optimal_slate", None)
    expected_reward = getattr(env, "expected_reward", None)
    if callable(optimal_slate) and callable(expected_reward):
        try:
            slate = optimal_slate()
            optimal_reward = float(expected_reward(slate))
        except Exception:  # pragma: no cover - fallback when methods raise
            optimal_reward = None
    return optimal_reward
