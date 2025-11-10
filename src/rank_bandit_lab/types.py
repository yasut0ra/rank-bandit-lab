from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


def _validate_probability(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Attraction probability must be in [0, 1], got {value}.")
    return value


@dataclass(frozen=True, slots=True)
class Document:
    """Descriptor for a single document that can appear in a ranking slate."""

    doc_id: str
    attraction: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "attraction", _validate_probability(self.attraction))


@dataclass(frozen=True, slots=True)
class Interaction:
    """Feedback generated from playing a slate in the environment."""

    slate: Tuple[str, ...]
    seen: Tuple[str, ...]
    click_index: int | None
    reward: float

    @property
    def clicked_doc_id(self) -> str | None:
        if self.click_index is None:
            return None
        if self.click_index < 0 or self.click_index >= len(self.seen):
            return None
        return self.seen[self.click_index]

    def seen_set(self) -> set[str]:
        return set(self.seen)


def normalize_slate(doc_ids: Sequence[str], slate_size: int) -> Tuple[str, ...]:
    if slate_size < 1:
        raise ValueError("slate_size must be >= 1.")
    unique = []
    seen = set()
    for doc_id in doc_ids:
        if doc_id in seen:
            raise ValueError(f"Duplicate document id '{doc_id}' inside slate.")
        seen.add(doc_id)
        unique.append(doc_id)
        if len(unique) == slate_size:
            break
    if len(unique) < slate_size:
        raise ValueError(f"Slate has {len(unique)} documents but requires {slate_size}.")
    return tuple(unique)


def ensure_known_documents(slate: Iterable[str], known: set[str]) -> None:
    missing = [doc_id for doc_id in slate if doc_id not in known]
    if missing:
        raise KeyError(f"Unknown document ids requested: {', '.join(missing)}")
