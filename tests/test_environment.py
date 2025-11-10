import unittest
from random import Random

from rank_bandit_lab.environment import (
    CascadeEnvironment,
    DependentClickEnvironment,
    PositionBasedEnvironment,
)
from rank_bandit_lab.types import Document


class CascadeEnvironmentTests(unittest.TestCase):
    def test_clicks_when_doc_is_guaranteed(self) -> None:
        env = CascadeEnvironment(
            documents=[Document("a", 1.0), Document("b", 0.0)],
            slate_size=2,
            rng=Random(123),
        )
        interaction = env.evaluate(["a", "b"])
        self.assertEqual(interaction.reward, 1.0)
        self.assertEqual(interaction.click_index, 0)
        self.assertEqual(interaction.seen, ("a",))

    def test_handles_absence_of_click(self) -> None:
        env = CascadeEnvironment(
            documents=[Document("a", 0.0), Document("b", 0.0)],
            slate_size=2,
            rng=Random(999),
        )
        interaction = env.evaluate(["a", "b"])
        self.assertEqual(interaction.reward, 0.0)
        self.assertIsNone(interaction.click_index)
        self.assertEqual(interaction.seen, ("a", "b"))

    def test_position_based_environment_respects_biases(self) -> None:
        env = PositionBasedEnvironment(
            documents=[Document("a", 1.0), Document("b", 1.0), Document("c", 1.0)],
            slate_size=3,
            position_biases=(1.0, 0.0, 1.0),
            rng=Random(7),
        )
        interaction = env.evaluate(["a", "b", "c"])
        self.assertIn("a", interaction.seen)
        self.assertNotIn("b", interaction.seen)
        self.assertEqual(interaction.click_positions, (0, 2))
        self.assertEqual(interaction.reward, 2.0)

    def test_dependent_click_environment_stops_after_satisfaction(self) -> None:
        env = DependentClickEnvironment(
            documents=[Document("a", 1.0), Document("b", 1.0)],
            slate_size=2,
            satisfaction={"a": 1.0, "b": 0.0},
            rng=Random(5),
        )
        interaction = env.evaluate(["a", "b"])
        self.assertEqual(interaction.seen, ("a",))
        self.assertEqual(interaction.click_positions, (0,))
