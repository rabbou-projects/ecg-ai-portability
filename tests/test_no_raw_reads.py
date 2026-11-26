"""No code here opens a publisher's raw files: every corpus the store holds is read through it.

Ruben's rule of 2026-11-25, "nobody should read raw": PTB-XL's raw folder left the
box on 2026-11-24, so a reader of ``ptbxl_database.csv`` or of a WFDB record fails
on the next run, or reads a partial copy (the Challenge-2021 bundle on the box held
12,616 of PTB-XL's 21,837 headers on 2026-11-26).  ``ecs.store`` is the one reader.

The scan parses every module under ``src``, ``scripts`` and ``tests`` and flags:

- a call into ``wfdb``'s readers (``rdrecord``, ``rdsamp``, ``rdheader``, ``rdann``);
- a string naming a publisher file or folder that builds a path (``root / name``)
  or is passed to a call (``read_csv(name)``, ``glob(name)``); a message or a
  docstring that names a file opens nothing;
- a use of a configured raw-corpus folder.

The corpora the store does not yet serve to this repository read their raw files
from the modules in ``NOT_YET_MOVED``, each with its reason.  The list may only
shrink: a listed module that no longer reads raw fails too, so it leaves the list.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("src", "scripts", "tests")

WFDB_READERS = frozenset({"rdrecord", "rdsamp", "rdheader", "rdann"})

# Files and folders as their publishers ship them (each corpus's README or file list).
PUBLISHER_NAMES = (
    "ptbxl_database.csv",
    "scp_statements.csv",
    "records500",
    "records100",
    "SHA256SUMS.txt",
    "metadata.csv",
    "CSV/train.csv",
    "CSV/test.csv",
    "row_data",
    "data/raw",
    "*.hea",
    "exams.csv",
    "record_list.csv",
    "machine_measurements.csv",
    "attributes.csv",
    "gold_standard.csv",
    "AttributesDictionary.csv",
    "exams_part",
)

# ``ecs.config``'s folders of raw corpora, those removed by the move included, so a
# reintroduced one is caught by name.
RAW_FOLDERS = frozenset(
    {
        "PTBXL_DIR",
        "SPH_DIR",
        "ACS_DIR",
        "CHALLENGE2021_DIR",
        "MIMIC_ECG_DIR",
        "MIMIC_PATIENTS_CSV",
        "CODE15_DIR",
        "CODE15_SAMPLE",
        "CODE_TEST_DIR",
        "ZZU_DIR",
        "ZZU_RECORDS_DIR",
        "ECHONEXT_DIR",
    }
)

_CODE = "CODE-15% and CODE-test: the store holds their tables, not their tracings"
_MIMIC = "MIMIC-IV-ECG: the store holds it at 250 Hz whole and at 500 Hz for a subset only"
_ZZU = "ZZU pECG: the store holds it; the move needs its own before/after proof"
_ECHONEXT = "EchoNext: restricted; ecgdb's own loader reads it in place under data/raw"
NOT_YET_MOVED: dict[str, str] = {
    "src/ecs/config.py": f"{_CODE}; {_MIMIC}; {_ZZU}",
    "src/ecs/ingest.py": f"read_wfdb, for {_MIMIC} and {_ZZU}",
    "src/ecs/code_ecg.py": _CODE,
    "src/ecs/mimic.py": _MIMIC,
    "src/ecs/zzu.py": _ZZU,
    "src/ecs/rotation.py": f"{_CODE}; {_MIMIC}; {_ZZU}",
    "src/ecs/echonext.py": _ECHONEXT,
    "scripts/code15_extract.py": _CODE,
    "scripts/code_zzu_label_table.py": f"{_CODE}; {_ZZU}",
    "scripts/mimic_label_table.py": _MIMIC,
    "scripts/mimic_shadow.py": _MIMIC,
    "scripts/echonext_clinical.py": _ECHONEXT,
    "scripts/echonext_provenance.py": _ECHONEXT,
    "scripts/echonext_scores.py": _ECHONEXT,
    "scripts/score_pairs.py": _ECHONEXT,
    "tests/test_code_zzu.py": f"writes a synthetic table for {_ZZU}",
    "tests/test_code_zzu_labels.py": f"{_CODE}; {_ZZU}",
    "tests/test_echonext.py": f"writes a synthetic release for {_ECHONEXT}",
    "tests/test_echonext_data.py": _ECHONEXT,
    "tests/test_mimic.py": _MIMIC,
    "tests/test_no_raw_reads.py": "this guard names what it looks for",
}


def _opening(node: ast.Constant, parents: dict[int, ast.AST]) -> bool:
    """Whether a string builds a path or is handed to a call, the two ways one opens."""
    parent = parents.get(id(node))
    if isinstance(parent, ast.keyword):
        parent = parents.get(id(parent))
    if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div):
        return True
    return isinstance(parent, ast.Call) and node is not parent.func


def raw_reads(source: str) -> list[str]:
    """Each place ``source`` reads a publisher's raw files, as ``line: what``."""
    tree = ast.parse(source)
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in WFDB_READERS:
            hits.append(f"{node.lineno}: wfdb.{node.attr}")
        elif isinstance(node, ast.alias) and node.name in WFDB_READERS:
            hits.append(f"{node.lineno}: import {node.name}")
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _opening(node, parents)
        ):
            named = [name for name in PUBLISHER_NAMES if name in node.value]
            hits.extend(f"{node.lineno}: {name!r}" for name in named)
        elif isinstance(node, ast.Name) and node.id in RAW_FOLDERS:
            hits.append(f"{node.lineno}: {node.id}")
        elif isinstance(node, ast.alias) and node.name in RAW_FOLDERS:
            hits.append(f"{node.lineno}: import {node.name}")
    return sorted(set(hits))


def _modules() -> list[Path]:
    return sorted(p for folder in SCANNED for p in (ROOT / folder).rglob("*.py"))


@pytest.mark.repo_scan
def test_no_module_outside_the_list_reads_a_publishers_raw_files() -> None:
    """A raw reader outside the list breaks once its folder leaves the box, as PTB-XL's did."""
    offenders = {}
    for path in _modules():
        name = path.relative_to(ROOT).as_posix()
        if name in NOT_YET_MOVED:
            continue
        hits = raw_reads(path.read_text())
        if hits:
            offenders[name] = hits
    assert offenders == {}, f"read through ecs.store instead: {offenders}"


@pytest.mark.repo_scan
def test_every_listed_module_still_reads_raw_so_the_list_only_shrinks() -> None:
    """A moved module left on the list would let a new raw read in it pass unseen."""
    stale = [
        name
        for name in NOT_YET_MOVED
        if not (ROOT / name).is_file() or not raw_reads((ROOT / name).read_text())
    ]
    assert stale == []


class TestTheDetector:
    """Each detector fires on its own case and stays quiet on its twin."""

    def test_a_wfdb_read_is_caught(self) -> None:
        assert raw_reads("import wfdb\nr = wfdb.rdrecord('x')\n") == ["2: wfdb.rdrecord"]

    def test_a_wfdb_write_is_not_a_read(self) -> None:
        assert raw_reads("import wfdb\nwfdb.wrsamp('x', fs=500)\n") == []

    def test_a_publisher_file_in_code_is_caught(self) -> None:
        source = "import pandas as pd\nd = pd.read_csv(root / 'ptbxl_database.csv')\n"
        assert raw_reads(source) == ["2: 'ptbxl_database.csv'"]

    def test_a_publisher_file_joined_to_a_folder_is_caught(self) -> None:
        assert raw_reads("p = root / 'scp_statements.csv'\n") == ["1: 'scp_statements.csv'"]

    def test_a_publisher_file_named_in_a_docstring_or_a_message_is_not_a_read(self) -> None:
        source = (
            'def f():\n    """``database`` is ptbxl_database.csv."""\n'
            '    return {"6.6": "the size of ptbxl_database.csv in MB"}\n'
        )
        assert raw_reads(source) == []

    def test_a_raw_folder_is_caught_by_name_and_by_import(self) -> None:
        source = "from ecs.config import PTBXL_DIR\np = PTBXL_DIR / 'x'\n"
        assert raw_reads(source) == ["1: import PTBXL_DIR", "2: PTBXL_DIR"]

    def test_a_folder_of_this_repository_is_not_a_raw_folder(self) -> None:
        assert raw_reads("from ecs.config import RESULTS_DIR\np = RESULTS_DIR / 'x'\n") == []
