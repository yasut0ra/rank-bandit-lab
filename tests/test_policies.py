import unittest
from random import Random

from rank_bandit_lab.policies import EpsilonGreedyRanking, ThompsonSamplingRanking
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
