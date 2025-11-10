from __future__ import annotations

from random import Random
from typing import Iterable, Mapping, Sequence, Tuple

from .types import Document, Interaction, ensure_known_documents, normalize_slate


def _prepare_documents(
    documents: Sequence[Document],
    slate_size: int,
) -> tuple[Tuple[Document, ...], dict[str, Document]]:
    if not documents:
        raise ValueError("Environment requires at least one document.")
    doc_map: dict[str, Document] = {}
    for doc in documents:
        if doc.doc_id in doc_map:
            raise ValueError(f"Duplicate document id detected: {doc.doc_id}")
        doc_map[doc.doc_id] = doc
    if slate_size < 1:
        raise ValueError("slate_size must be >= 1.")
    if slate_size > len(documents):
        raise ValueError(
            f"slate_size ({slate_size}) cannot exceed number of documents ({len(documents)})."
        )
    return tuple(documents), doc_map


class CascadeEnvironment:
    """Simulates user interactions following the cascade click model."""

    __slots__ = ("documents", "slate_size", "rng", "_doc_map")

    def __init__(
        self,
        documents: Sequence[Document],
        slate_size: int,
        rng: Random | None = None,
    ) -> None:
        documents, doc_map = _prepare_documents(documents, slate_size)
        self.documents = documents
        self._doc_map = doc_map
        self.slate_size = slate_size
        self.rng = rng or Random()

    @property
    def doc_ids(self) -> Tuple[str, ...]:
        return tuple(doc.doc_id for doc in self.documents)

    def evaluate(self, slate: Sequence[str]) -> Interaction:
        ensure_known_documents(slate, set(self._doc_map))
        normalized = normalize_slate(slate, self.slate_size)
        seen: list[str] = []
        click_index: int | None = None
        reward = 0.0
        for position, doc_id in enumerate(normalized):
            seen.append(doc_id)
            attraction = self._doc_map[doc_id].attraction
            if self.rng.random() < attraction:
                click_index = position
                reward = 1.0
                break
        click_positions = (click_index,) if click_index is not None else ()
        return Interaction(
            slate=normalized,
            seen=tuple(seen),
            click_index=click_index,
            reward=reward,
            click_positions=click_positions,
        )

    def reseed(self, seed: int | None) -> None:
        self.rng.seed(seed)

    def iter_documents(self) -> Iterable[Document]:
        return iter(self.documents)

    def optimal_slate(self) -> Tuple[str, ...]:
        ordered = sorted(self.doc_ids, key=lambda doc_id: self._doc_map[doc_id].attraction, reverse=True)
        return tuple(ordered[: self.slate_size])

    def expected_reward(self, slate: Sequence[str]) -> float:
        ensure_known_documents(slate, set(self._doc_map))
        normalized = normalize_slate(slate, self.slate_size)
        prob_no_click = 1.0
        for doc_id in normalized:
            prob_no_click *= 1.0 - self._doc_map[doc_id].attraction
        return 1.0 - prob_no_click


class PositionBasedEnvironment:
    """Position-Based Model (PBM) where examination depends on position bias."""

    __slots__ = ("documents", "slate_size", "rng", "_doc_map", "_position_biases")

    def __init__(
        self,
        documents: Sequence[Document],
        slate_size: int,
        position_biases: Sequence[float],
        rng: Random | None = None,
    ) -> None:
        documents, doc_map = _prepare_documents(documents, slate_size)
        if len(position_biases) < slate_size:
            raise ValueError("position_biases length must match or exceed slate_size.")
        validated = []
        for index, bias in enumerate(position_biases[:slate_size]):
            if not 0.0 <= bias <= 1.0:
                raise ValueError(
                    f"Position bias at index {index} must be in [0, 1], got {bias}."
                )
            validated.append(bias)
        self.documents = documents
        self._doc_map = doc_map
        self.slate_size = slate_size
        self._position_biases = tuple(validated)
        self.rng = rng or Random()

    @property
    def doc_ids(self) -> Tuple[str, ...]:
        return tuple(doc.doc_id for doc in self.documents)

    def evaluate(self, slate: Sequence[str]) -> Interaction:
        ensure_known_documents(slate, set(self._doc_map))
        normalized = normalize_slate(slate, self.slate_size)
        seen: list[str] = []
        click_positions: list[int] = []
        for position, doc_id in enumerate(normalized):
            examined = self.rng.random() < self._position_biases[position]
            if not examined:
                continue
            seen.append(doc_id)
            attraction = self._doc_map[doc_id].attraction
            if self.rng.random() < attraction:
                click_positions.append(position)
        click_index = click_positions[0] if click_positions else None
        reward = float(len(click_positions))
        return Interaction(
            slate=normalized,
            seen=tuple(seen),
            click_index=click_index,
            reward=reward,
            click_positions=tuple(click_positions),
        )

    def optimal_slate(self) -> Tuple[str, ...]:
        ordered = sorted(self.doc_ids, key=lambda doc_id: self._doc_map[doc_id].attraction, reverse=True)
        return tuple(ordered[: self.slate_size])

    def expected_reward(self, slate: Sequence[str]) -> float:
        ensure_known_documents(slate, set(self._doc_map))
        normalized = normalize_slate(slate, self.slate_size)
        reward = 0.0
        for position, doc_id in enumerate(normalized):
            reward += self._position_biases[position] * self._doc_map[doc_id].attraction
        return reward


class DependentClickEnvironment:
    """Dependent Click Model (DCM) with per-document satisfaction probabilities."""

    __slots__ = ("documents", "slate_size", "rng", "_doc_map", "_satisfaction")

    def __init__(
        self,
        documents: Sequence[Document],
        slate_size: int,
        satisfaction: Mapping[str, float] | None = None,
        default_satisfaction: float = 0.5,
        rng: Random | None = None,
    ) -> None:
        documents, doc_map = _prepare_documents(documents, slate_size)
        if not 0.0 <= default_satisfaction <= 1.0:
            raise ValueError("default_satisfaction must be in [0, 1].")
        satisfaction_map = dict(satisfaction or {})
        for doc_id, value in satisfaction_map.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Satisfaction probability for '{doc_id}' must be in [0, 1], got {value}."
                )
        resolved = {
            doc_id: satisfaction_map.get(doc_id, default_satisfaction)
            for doc_id in doc_map
        }
        self.documents = documents
        self._doc_map = doc_map
        self._satisfaction = resolved
        self.slate_size = slate_size
        self.rng = rng or Random()

    @property
    def doc_ids(self) -> Tuple[str, ...]:
        return tuple(doc.doc_id for doc in self.documents)

    def evaluate(self, slate: Sequence[str]) -> Interaction:
        ensure_known_documents(slate, set(self._doc_map))
        normalized = normalize_slate(slate, self.slate_size)
        seen: list[str] = []
        click_positions: list[int] = []
        continue_exam = True
        for position, doc_id in enumerate(normalized):
            if not continue_exam:
                break
            seen.append(doc_id)
            attraction = self._doc_map[doc_id].attraction
            if self.rng.random() < attraction:
                click_positions.append(position)
                satisfaction = self._satisfaction[doc_id]
                if self.rng.random() < satisfaction:
                    continue_exam = False
        click_index = click_positions[0] if click_positions else None
        reward = float(len(click_positions))
        return Interaction(
            slate=normalized,
            seen=tuple(seen),
            click_index=click_index,
            reward=reward,
            click_positions=tuple(click_positions),
        )

    def optimal_slate(self) -> Tuple[str, ...]:
        ordered = sorted(self.doc_ids, key=lambda doc_id: self._doc_map[doc_id].attraction, reverse=True)
        return tuple(ordered[: self.slate_size])

    def expected_reward(self, slate: Sequence[str]) -> float:
        ensure_known_documents(slate, set(self._doc_map))
        normalized = normalize_slate(slate, self.slate_size)
        reward = 0.0
        continue_prob = 1.0
        for doc_id in normalized:
            reward += continue_prob * self._doc_map[doc_id].attraction
            continue_prob *= 1.0 - self._doc_map[doc_id].attraction * self._satisfaction[doc_id]
        return reward
