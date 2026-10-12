# -*- coding: utf-8 -*-
"""云端入口最小回归：不训练，不模拟整批矩阵。"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")


@unittest.skipUnless(BASH, "入口检查需要 bash")
class CloudTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts/run_cloud.sh"
        shutil.copyfile(ROOT / "scripts/run_cloud.sh", self.script)
        self.env = {**os.environ, "PYTHON": "false", "PAR": "1"}

    def test_cuda_failure_stops_all(self):
        result = subprocess.run([BASH, self.script.as_posix(), "all"], cwd=self.root,
                                env=self.env, capture_output=True, encoding="utf-8", timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA 检查失败", result.stdout)

    def test_partial_checkpoint_and_report_are_not_complete(self):
        commands = '''source "$1" status >/dev/null
mkdir -p outputs/train/joint_f0_s0 outputs/eval/joint_f0_s0_paderborn
printf '中间轮' > outputs/train/joint_f0_s0/last.pt
printf '{}' > outputs/eval/joint_f0_s0_paderborn/metrics.json
! cell_done joint_f0_s0 && ! eval_done joint_f0_s0_paderborn'''
        result = subprocess.run([BASH, "-c", commands, "入口检查", self.script.as_posix()],
                                cwd=self.root, env=self.env, capture_output=True,
                                encoding="utf-8", timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
