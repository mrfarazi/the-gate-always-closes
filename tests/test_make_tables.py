"""Regenerating the tables from results/ must reproduce the golden tables number for number."""
import glob
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "tests", "golden", "tables")


def _tokens(text):
    body = "\n".join(l for l in text.split("\n") if not l.startswith("%"))
    return re.findall(r"-?\d+(?:[.,]\d+)*", body)


@pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "results", "knockouts")), reason="results/ not present")
def test_regenerated_tables_match_golden(tmp_path):
    out = tmp_path / "tables"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_tables.py"), "--out", str(out)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    golden = sorted(glob.glob(os.path.join(GOLDEN, "*.tex")))
    assert golden, "golden tables missing"
    for g in golden:
        name = os.path.basename(g)
        new = out / name
        assert new.exists(), name
        assert _tokens(open(g).read()) == _tokens(open(new).read()), f"{name}: numbers differ"
