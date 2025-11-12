import tempfile
import unittest
from pathlib import Path

from rank_bandit_lab.logging import load_log, serialize_log, write_log
from rank_bandit_lab.simulator import SimulationLog
from rank_bandit_lab.types import Interaction


def make_log() -> SimulationLog:
    interactions = [
        Interaction(("a", "b"), ("a",), 0, 1.0, (0,)),
        Interaction(("b", "a"), ("b", "a"), None, 0.0, ()),
    ]
    return SimulationLog(interactions, optimal_reward=0.7)


class LoggingTests(unittest.TestCase):
    def test_serialize_log_contains_metadata(self) -> None:
        log = make_log()
        data = serialize_log(log, metadata={"doc_ids": ["a", "b"]})
        self.assertEqual(data["metadata"]["doc_ids"], ["a", "b"])
        self.assertEqual(len(data["interactions"]), 2)

    def test_write_and_load_log_round_trip(self) -> None:
        log = make_log()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            write_log(path, log, metadata={"doc_ids": ["a", "b"]})
            loaded_log, metadata = load_log(path)
        self.assertEqual(metadata["doc_ids"], ["a", "b"])
        self.assertEqual(len(loaded_log.interactions), 2)
        self.assertEqual(loaded_log.optimal_reward, 0.7)
