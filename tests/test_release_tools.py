from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseToolsTest(unittest.TestCase):
    def test_release_smoke(self) -> None:
        subprocess.run(
            [sys.executable, "scripts/release_smoke.py", "--config", "configs/paper_main.example.json"],
            cwd=ROOT,
            check=True,
        )

    def test_sequence_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "000000.json").write_text(
                json.dumps({"ground_truth": {"protein_seq": "ACDE"}, "prediction": {"protein_seq": "ACXE"}}),
                encoding="utf-8",
            )
            output = directory / "summary.json"
            subprocess.run(
                [sys.executable, "scripts/evaluate_sequences.py", "--predictions", str(directory), "--output", str(output)],
                cwd=ROOT,
                check=True,
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["position_recovery"], 0.75)


if __name__ == "__main__":
    unittest.main()
