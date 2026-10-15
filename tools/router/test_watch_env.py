#!/usr/bin/env python3
"""watch.sh requires operator-supplied WATCH_HOSTS."""
import os
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class WatchEnvTest(unittest.TestCase):
    def _run_watch(self, env: dict) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(HERE / "watch.sh")],
            cwd=HERE.parent.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_watch_hosts_required(self):
        env = {k: v for k, v in os.environ.items() if k != "WATCH_HOSTS"}
        env.pop("WATCH_HOSTS", None)
        proc = self._run_watch(env)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("WATCH_HOSTS", proc.stderr)

    def test_watch_hosts_whitespace_rejected(self):
        env = {k: v for k, v in os.environ.items() if k != "WATCH_HOSTS"}
        env["WATCH_HOSTS"] = "   "
        proc = self._run_watch(env)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("WATCH_HOSTS", proc.stderr)


if __name__ == "__main__":
    unittest.main()
