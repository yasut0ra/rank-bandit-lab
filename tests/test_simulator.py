import unittest
from random import Random

from rank_bandit_lab.environment import CascadeEnvironment
from rank_bandit_lab.policies import ThompsonSamplingRanking
from rank_bandit_lab.simulator import BanditSimulator
from rank_bandit_lab.types import Document


class SimulatorTests(unittest.TestCase):
    def test_simulator_runs_and_reports_summary(self) -> None:
        documents = [
            Document("a", 0.5),
            Document("b", 0.2),
            Document("c", 0.1),
        ]
        env = CascadeEnvironment(documents, slate_size=2, rng=Random(42))
        policy = ThompsonSamplingRanking(["a", "b", "c"], slate_size=2, rng=Random(84))
        simulator = BanditSimulator(env, policy)

        log = simulator.run(25)
        summary = log.summary()

        self.assertEqual(summary["rounds"], 25)
        self.assertGreaterEqual(summary["ctr"], 0.0)
        self.assertLessEqual(summary["ctr"], 1.0)
        self.assertGreaterEqual(sum(summary["seen_counts"].values()), summary["rounds"])
