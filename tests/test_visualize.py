import unittest

from rank_bandit_lab.simulator import SimulationLog
from rank_bandit_lab.types import Interaction
from rank_bandit_lab.visualize import doc_distribution_data, learning_curve_data


class VisualizeDataTests(unittest.TestCase):
    def test_learning_curve_data_returns_series(self) -> None:
        log = SimulationLog(
            [
                Interaction(("a",), ("a",), 0, 1.0, (0,)),
                Interaction(("a",), ("a",), None, 0.0, ()),
            ]
        )
        data = learning_curve_data(log)
        self.assertEqual(data["rounds"], [1, 2])
        self.assertEqual(data["cumulative_reward"], [1.0, 1.0])
        self.assertAlmostEqual(data["ctr"][-1], 0.5)

    def test_doc_distribution_data_orders_documents(self) -> None:
        log = SimulationLog(
            [
                Interaction(("a", "b"), ("a", "b"), 0, 1.0, (0,)),
                Interaction(("b", "a"), ("b",), None, 0.0, ()),
            ]
        )
        data = doc_distribution_data(log, ["b", "a"])
        self.assertEqual(data["doc_ids"], ["b", "a"])
        self.assertEqual(data["seen"], [2, 1])
        self.assertEqual(data["clicks"], [0, 1])
