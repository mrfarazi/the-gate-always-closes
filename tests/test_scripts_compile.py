"""Every script and module must at least byte-compile (catches half-edited files early)."""
import glob
import os
import py_compile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = sorted(glob.glob(os.path.join(ROOT, "hra", "**", "*.py"), recursive=True)
               + glob.glob(os.path.join(ROOT, "scripts", "**", "*.py"), recursive=True))


@pytest.mark.parametrize("path", FILES, ids=[os.path.relpath(f, ROOT) for f in FILES])
def test_compiles(path):
    py_compile.compile(path, doraise=True)
