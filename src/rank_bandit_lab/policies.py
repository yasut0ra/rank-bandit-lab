from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import Random
from typing import Sequence, Tuple

from .types import Interaction


class RankingPolicy(ABC):
    """Interface for ranking bandit policies."""

    def __init__(
        self,
        doc_ids: Sequence[str],
        slate_size: int,
        rng: Random | None = None,
    ) -> None:
        if not doc_ids:
            raise ValueError("Policy requires at least one document id.")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("Document ids must be unique.")
        if slate_size < 1:
            raise ValueError("slate_size must be >= 1.")
        if slate_size > len(doc_ids):
            raise ValueError("slate_size cannot exceed the number of documents.")
        self._doc_ids = tuple(doc_ids)
        self._slate_size = slate_size
        self._rng = rng or Random()

    @property
    def doc_ids(self) -> Tuple[str, ...]:
        return self._doc_ids

    @property
    def slate_size(self) -> int:
        return self._slate_size

    @abstractmethod
    def select_slate(self) -> Tuple[str, ...]:
        """Return the ordered list of document ids to display."""

    @abstractmethod
    def update(self, interaction: Interaction) -> None:
        """Update the policy with fresh feedback."""


@dataclass(slots=True)
class ArmStats:
    impressions: int = 0
    clicks: int = 0


class EpsilonGreedyRanking(RankingPolicy):
    """Classic epsilon-greedy policy for ranking bandits."""

    def __init__(
        self,
        doc_ids: Sequence[str],
        slate_size: int,
        epsilon: float = 0.1,
        prior_success: float = 1.0,
        prior_failure: float = 1.0,
        rng: Random | None = None,
    ) -> None:
        super().__init__(doc_ids, slate_size, rng)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        if prior_success <= 0 or prior_failure <= 0:
            raise ValueError("prior_success and prior_failure must be > 0.")
        self._epsilon = epsilon
        self._prior_success = prior_success
        self._prior_failure = prior_failure
        self._stats = {doc_id: ArmStats() for doc_id in self.doc_ids}

    def select_slate(self) -> Tuple[str, ...]:
        rng_value = self._rng.random()
        if rng_value < self._epsilon:
            sampled = list(self.doc_ids)
            self._rng.shuffle(sampled)
            return tuple(sampled[: self.slate_size])
        ranked = sorted(self.doc_ids, key=self._score, reverse=True)
        return tuple(ranked[: self.slate_size])

    def update(self, interaction: Interaction) -> None:
        for doc_id in interaction.seen:
            stats = self._stats[doc_id]
            stats.impressions += 1
        clicked_doc = interaction.clicked_doc_id
        if clicked_doc is not None:
            self._stats[clicked_doc].clicks += 1

    def _score(self, doc_id: str) -> float:
        stats = self._stats[doc_id]
        numerator = stats.clicks + self._prior_success
        denominator = stats.impressions + self._prior_success + self._prior_failure
        return numerator / denominator


class ThompsonSamplingRanking(RankingPolicy):
    """Thompson sampling with independent Beta posteriors per document."""

    def __init__(
        self,
        doc_ids: Sequence[str],
        slate_size: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        rng: Random | None = None,
    ) -> None:
        super().__init__(doc_ids, slate_size, rng)
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("alpha_prior and beta_prior must be > 0.")
        self._alpha_prior = alpha_prior
        self._beta_prior = beta_prior
        self._successes = {doc_id: 0 for doc_id in self.doc_ids}
        self._failures = {doc_id: 0 for doc_id in self.doc_ids}

    def select_slate(self) -> Tuple[str, ...]:
        scored = []
        for doc_id in self.doc_ids:
            alpha = self._alpha_prior + self._successes[doc_id]
            beta = self._beta_prior + self._failures[doc_id]
            sample = self._rng.betavariate(alpha, beta)
            scored.append((sample, doc_id))
        scored.sort(reverse=True)
        return tuple(doc_id for _, doc_id in scored[: self.slate_size])

    def update(self, interaction: Interaction) -> None:
        if not interaction.seen:
            return
        click_index = interaction.click_index
        for position, doc_id in enumerate(interaction.seen):
            if click_index is not None and position == click_index:
                self._successes[doc_id] += 1
                break
            self._failures[doc_id] += 1
        else:
            # No click was observed; every seen item was a failure.
            pass
