import tempfile
from pathlib import Path
import unittest

from rank_bandit_lab.compare import summaries_to_table, summarize
from rank_bandit_lab.logging import write_log
from rank_bandit_lab.simulator import SimulationLog
from rank_bandit_lab.types import Interaction


def build_log(reward_sequence: list[float]) -> SimulationLog:
    interactions = []
    for reward in reward_sequence:
        click_positions = (0,) if reward > 0 else ()
        interactions.append(
            Interaction(("a",), ("a",), 0 if reward > 0 else None, reward, click_positions)
        )
    return SimulationLog(interactions, optimal_reward=0.8)


class CompareTests(unittest.TestCase):
    def test_summarize_reads_metadata(self) -> None:
        log = build_log([1.0, 0.0, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            write_log(path, log, metadata={"label": "exp1", "algo": "ucb", "model": "cascade"})
            summary, _log, metadata = summarize(str(path))
        self.assertEqual(summary.label, "exp1")
        self.assertEqual(summary.rounds, 3)
        self.assertEqual(summary.algo, "ucb")
        self.assertEqual(metadata["model"], "cascade")

    def test_summaries_to_table_contains_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path1 = Path(tmp) / "log1.json"
            path2 = Path(tmp) / "log2.json"
            write_log(path1, build_log([1.0, 0.0]), metadata={"label": "best"})
            write_log(path2, build_log([0.0, 0.0]), metadata={"label": "worst"})
            summary1, _, _ = summarize(str(path1))
            summary2, _, _ = summarize(str(path2))
        table = summaries_to_table([summary1, summary2])
        self.assertIn("best", table)
        self.assertIn("worst", table)
