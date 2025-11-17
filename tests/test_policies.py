import unittest
from random import Random

from rank_bandit_lab.policies import (
    EpsilonGreedyRanking,
    SoftmaxRanking,
    ThompsonSamplingRanking,
    UCB1Ranking,
)
from rank_bandit_lab.types import Interaction


def make_interaction(slate, seen, click_index):
    reward = 1.0 if click_index is not None else 0.0
    return Interaction(tuple(slate), tuple(seen), click_index, reward)


class PolicyTests(unittest.TestCase):
    def test_epsilon_greedy_prefers_high_ctr_doc(self) -> None:
        policy = EpsilonGreedyRanking(
            doc_ids=["a", "b", "c"],
            slate_size=2,
            epsilon=0.0,
            rng=Random(0),
        )
        policy.update(make_interaction(("a", "b"), ("a",), 0))
        policy.update(make_interaction(("b", "a"), ("b", "a"), None))
        policy.update(make_interaction(("a", "c"), ("a",), 0))
        slate = policy.select_slate()
        self.assertEqual(slate[0], "a")
        self.assertEqual(len(slate), 2)

    def test_epsilon_greedy_counts_all_clicked_docs(self) -> None:
        policy = EpsilonGreedyRanking(
            doc_ids=["a", "b", "c"],
            slate_size=3,
            epsilon=0.0,
            rng=Random(1),
        )
        interaction = Interaction(
            slate=("a", "b", "c"),
            seen=("a", "b", "c"),
            click_index=0,
            reward=2.0,
            click_positions=(0, 2),
        )
        policy.update(interaction)
        self.assertEqual(policy._stats["a"].clicks, 1)  # type: ignore[attr-defined]
        self.assertEqual(policy._stats["c"].clicks, 1)  # type: ignore[attr-defined]
        self.assertEqual(policy._stats["b"].clicks, 0)  # type: ignore[attr-defined]

    def test_thompson_sampling_tracks_successes_and_failures(self) -> None:
        policy = ThompsonSamplingRanking(
            doc_ids=["a", "b"],
            slate_size=1,
            rng=Random(5),
        )
        policy.update(make_interaction(("a",), ("a",), 0))
        policy.update(make_interaction(("b",), ("b",), None))
        self.assertEqual(policy._successes["a"], 1)  # type: ignore[attr-defined]
        self.assertEqual(policy._failures["b"], 1)  # type: ignore[attr-defined]
        slate = policy.select_slate()
        self.assertEqual(len(slate), 1)

    def test_ucb_prefers_document_with_more_clicks(self) -> None:
        policy = UCB1Ranking(
            doc_ids=["a", "b"],
            slate_size=1,
            confidence=0.5,
            rng=Random(1),
        )
        for _ in range(5):
            policy.update(make_interaction(("a",), ("a",), 0))
        for _ in range(5):
            policy.update(make_interaction(("b",), ("b",), None))
        slate = policy.select_slate()
        self.assertEqual(slate[0], "a")

    def test_softmax_gives_preference_to_high_mean(self) -> None:
        policy = SoftmaxRanking(
            doc_ids=["a", "b"],
            slate_size=1,
            temperature=0.2,
            rng=Random(2),
        )
        for _ in range(5):
            policy.update(make_interaction(("a",), ("a",), 0))
        for _ in range(5):
            policy.update(make_interaction(("b",), ("b",), None))
        weight_a = policy._weight("a")  # type: ignore[attr-defined]
        weight_b = policy._weight("b")  # type: ignore[attr-defined]
        self.assertGreater(weight_a, weight_b)
