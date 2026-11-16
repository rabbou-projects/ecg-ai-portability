"""A personal literal or a home path added to a tracked file fails CI (problem 14, "data lives in code").

``scripts/hardcode_ratchet.py`` counts, per tracked non-prose file, absolute home
paths and tailnet addresses, the owner's contact literals and IBANs, and the
owner's name. ``hardcode-baseline.json`` holds the counts; they may only fall.
Every invented value below is built for the test and is no one's.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
while not (_REPO / "scripts" / "hardcode_ratchet.py").exists():
    _REPO = _REPO.parent
_spec = importlib.util.spec_from_file_location("hardcode_ratchet", _REPO / "scripts" / "hardcode_ratchet.py")
assert _spec and _spec.loader
hr = importlib.util.module_from_spec(_spec)
sys.modules["hardcode_ratchet"] = hr
_spec.loader.exec_module(hr)


def _h(s: str) -> frozenset[str]:
    return frozenset({hashlib.sha256(s.encode()).hexdigest()[:16]})


def test_repo_counts_equal_the_baseline() -> None:
    """A new home path or personal literal in a tracked file reddens CI; a removed one must lower the baseline."""
    baseline = json.loads((_REPO / hr.BASELINE_NAME).read_text())
    assert hr.compare(hr.scan(_REPO), baseline) == []


def test_home_paths_and_tailnet_addresses_count() -> None:
    """A path into one user's home or a tailnet IP resolves on one machine only; the CI runner's home does not count."""
    text = 'a = "/Users/someone/x"\nb = "/home/alice/y"\nc = "/home/runner/work"\nd = "100.101.102.103"\ne = "100.200.1.1"\n'
    assert hr.count_text(text)["home_path"] == 3


def test_contact_literals_match_by_hash() -> None:
    """The owner's e-mail, phone (either form) and birth date are found without being written in the scanner."""
    text = "mail: Jane.Doe@example.org\ntel: +33 6 00 00 00 01 or 06.00.00.00.01\nborn 01/02/1990 and 1990-02-01\n"
    c = hr.count_text(
        text,
        emails=_h("jane.doe@example.org"),
        phones=_h("0600000001"),
        dates=_h("1990-02-01"),
    )
    assert c["contact"] == 5
    assert hr.count_text(text)["contact"] == 0


def test_only_a_checksum_valid_iban_counts() -> None:
    """An IBAN in code is an account number; a string of the same shape that fails mod 97 is not one."""
    assert hr.count_text("iban = 'GB82 WEST 1234 5698 7654 32'")["contact"] == 1
    assert hr.count_text("iban = 'GB83 WEST 1234 5698 7654 32'")["contact"] == 0


def test_name_matches_a_whole_word_only() -> None:
    """The owner's name as a word counts; the same letters inside a longer word do not."""
    names = _h("alice")
    assert hr.count_text("Alice wrote; alice's; malice; alicex", names=names)["name"] == 2


def test_compare_refuses_a_rise_a_new_file_and_a_stale_baseline() -> None:
    """The baseline may only fall, and must fall with the code: each of the three differences is an error."""
    base = {"home_path": {"a.py": 2}, "contact": {}, "name": {}}
    assert hr.compare({"home_path": {"a.py": 2}}, base) == []
    rise = hr.compare({"home_path": {"a.py": 3}}, base)
    new = hr.compare({"home_path": {"a.py": 2, "b.py": 1}}, base)
    fall = hr.compare({"home_path": {"a.py": 1}}, base)
    assert [e.split()[0] for e in rise + new + fall] == ["RISE", "RISE", "FALL"]


def _git(*args: str) -> None:
    """Run git as a test fixture; a hook's GIT_DIR must not aim it at the real repo (``git init`` would reinitialise it)."""
    subprocess.run(["git", *args], check=True, env=hr.git_env())


def test_lower_refuses_to_write_a_rise(tmp_path: Path) -> None:
    """--lower writes a fall and refuses a rise, so the baseline cannot be raised by the tool."""
    _git("init", "-q", str(tmp_path))
    f = tmp_path / "a.py"
    f.write_text('p = "/Users/someone/a"\n')
    _git("-C", str(tmp_path), "add", "a.py")
    assert hr.main(["--root", str(tmp_path), "--lower"]) == 0
    assert json.loads((tmp_path / hr.BASELINE_NAME).read_text())["home_path"] == {"a.py": 1}
    f.write_text('p = "/Users/someone/a"\nq = "/home/bob/b"\n')
    assert hr.main(["--root", str(tmp_path), "--lower"]) == 1
    assert hr.main(["--root", str(tmp_path), "--check"]) == 1
    f.write_text("p = 1\n")
    assert hr.main(["--root", str(tmp_path), "--check"]) == 1
    assert hr.main(["--root", str(tmp_path), "--lower"]) == 0
    assert hr.main(["--root", str(tmp_path), "--check"]) == 0


def test_prose_is_left_out(tmp_path: Path) -> None:
    """Markdown is prose and is not counted; the same line in a code file is."""
    _git("init", "-q", str(tmp_path))
    (tmp_path / "README.md").write_text("/Users/someone/a\n")
    (tmp_path / "run.sh").write_text("/Users/someone/a\n")
    _git("-C", str(tmp_path), "add", ".")
    assert hr.scan(tmp_path)["home_path"] == {"run.sh": 1}


if __name__ == "__main__":
    import tempfile

    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print("ok", name)
