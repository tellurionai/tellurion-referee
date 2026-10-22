"""Leakage guard: only scripts/evaluate.py may load the locked `test` split.

tellurion_sim/dataset.py is also allowed, because it *writes* the splits.
If this test fails on your change, you are touching the test split somewhere you shouldn't.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "scripts/evaluate.py",
    "tellurion_sim/dataset.py",
    "scripts/make_stub_data.py",
    "tests/test_no_test_leakage.py",
}
PATTERNS = [
    re.compile(r"""split\s*(==|=|:)\s*['"]test['"]"""),   # split="test", split == "test", split: 'test'
    re.compile(r"""\[\s*['"]test['"]\s*\]"""),            # splits["test"]
    re.compile(r"""['"]test\.npz['"]|/test/|_test\.npz"""),  # test split files
    re.compile(r"""--split\s+test"""),
]
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "build", "dist", "__pycache__"}


def python_files():
    for p in ROOT.rglob("*.py"):
        if not SKIP_DIRS.intersection(p.relative_to(ROOT).parts):
            yield p


def test_only_evaluate_loads_test_split():
    offenders = []
    for p in python_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in ALLOWED:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if any(pat.search(line) for pat in PATTERNS):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "Locked test split referenced outside scripts/evaluate.py:\n" + "\n".join(offenders)
