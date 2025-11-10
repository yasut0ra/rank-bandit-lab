from __future__ import annotations

from random import Random
from typing import Iterable, Sequence, Tuple

from .types import Document, Interaction, ensure_known_documents, normalize_slate


class CascadeEnvironment:
    """Simulates user interactions following the cascade click model."""

    __slots__ = ("documents", "slate_size", "rng", "_doc_map")

    def __init__(
        self,
        documents: Sequence[Document],
        slate_size: int,
        rng: Random | None = None,
    ) -> None:
        if not documents:
            raise ValueError("Environment requires at least one document.")
        doc_map = {}
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
        self.documents = tuple(documents)
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
        return Interaction(
            slate=normalized,
            seen=tuple(seen),
            click_index=click_index,
            reward=reward,
        )

    def reseed(self, seed: int | None) -> None:
        self.rng.seed(seed)

    def iter_documents(self) -> Iterable[Document]:
        return iter(self.documents)
