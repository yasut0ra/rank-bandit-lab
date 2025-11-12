import json
import tempfile
import unittest
from pathlib import Path

from rank_bandit_lab import sweep


class SweepTests(unittest.TestCase):
    def test_parse_run_spec_allows_missing_algo(self) -> None:
        label, overrides = sweep.parse_run_spec("eps05:epsilon=0.05")
        self.assertEqual(label, "eps05")
        self.assertEqual(overrides["epsilon"], "0.05")

    def test_run_sweep_generates_logs(self) -> None:
        parser = sweep.build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "logs"
            summary_path = Path(tmp) / "summary.json"
            args = parser.parse_args(
                [
                    "--run",
                    "eps05:algo=epsilon,epsilon=0.05,seed=1",
                    "--run",
                    "ucb07:algo=ucb,ucb_confidence=0.7,seed=2",
                    "--steps",
                    "20",
                    "--slate-size",
                    "2",
                    "--output-dir",
                    str(output_dir),
                    "--summary-json",
                    str(summary_path),
                ]
            )
            sweep.run_sweep(args)
            self.assertTrue((output_dir / "eps05.json").exists())
            self.assertTrue((output_dir / "ucb07.json").exists())
            summary = json.loads(summary_path.read_text())
            labels = {entry["label"] for entry in summary}
            self.assertIn("eps05", labels)
            self.assertIn("ucb07", labels)
