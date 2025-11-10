import unittest
from random import Random

from rank_bandit_lab.environment import CascadeEnvironment
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
