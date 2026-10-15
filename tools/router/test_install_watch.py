#!/usr/bin/env python3
"""install-service.sh enables the watcher only when watch.env defines WATCH_HOSTS."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


class InstallWatchGateTest(unittest.TestCase):
    def run_install(self, home: Path, watch_env: str | None) -> subprocess.CompletedProcess:
        fake_bin = home / "bin"
        fake_bin.mkdir()
        log = home / "systemctl.log"
        script = f"""#!/bin/sh
echo "$@" >> "{log}"
if [ "$1" = --user ]; then
  shift
fi
case "$1" in
  enable) echo "enabled $3" ;;
  disable) echo "disabled $3" ;;
  daemon-reload) ;;
  is-enabled) exit 1 ;;
  list-units) exit 0 ;;
  stop|reset-failed) ;;
esac
exit 0
"""
        (fake_bin / "systemctl").write_text(script)
        (fake_bin / "systemctl").chmod(0o755)
        cfg = home / ".config" / "bonsai9b"
        cfg.mkdir(parents=True)
        if watch_env is not None:
            (cfg / "watch.env").write_text(watch_env)
        env = {
            **os.environ,
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '/usr/bin')}",
        }
        return subprocess.run(
            ["bash", str(REPO / "tools/router/install-service.sh")],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO,
        )

    def test_watch_not_enabled_without_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            proc = self.run_install(home, None)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            log = (home / "systemctl.log").read_text()
            self.assertIn("enable --now bonsai9b-router.service", log)
            self.assertNotIn("enable --now bonsai9b-watch.service", log)
            self.assertIn("Fleet watcher not enabled", proc.stdout)

    def test_watch_enabled_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            proc = self.run_install(home, "WATCH_HOSTS='box-5 box-6'\n")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            log = (home / "systemctl.log").read_text()
            self.assertIn("enable --now bonsai9b-watch.service", log)
            self.assertIn("enable --now bonsai9b-router.service", log)


if __name__ == "__main__":
    unittest.main()
