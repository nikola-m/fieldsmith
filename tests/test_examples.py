"""Run every example script end-to-end; each must exit cleanly and report no [FAIL]."""
import glob
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)
EXAMPLES = sorted(glob.glob(os.path.join(HERE, "..", "examples", "[0-9][0-9]_*.py")))


@pytest.mark.slow
@pytest.mark.parametrize("path", EXAMPLES, ids=[os.path.basename(p) for p in EXAMPLES])
def test_example_runs(path):
    proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=1800,
                          cwd=os.path.dirname(path))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "[FAIL]" not in proc.stdout
    assert "[PASS]" in proc.stdout or "PASS" in proc.stdout or proc.stdout.strip()
