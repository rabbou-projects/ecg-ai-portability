"""The one way this repository reads a corpus: cardiodb's store, never a publisher's files.

PTB-XL, SPH, ACS-ECG and the Challenge-2021 partitions are read here, from the
store under ``CARDIODB_PROJECT`` (``cardiodb``'s README, "Reading tracings" and
"Querying").  ``cardiodb`` runs in its own environment, through
``cardiodb_bridge.py``: it is private and pins numpy 2, while this repository is
public and pins numpy 1 for torch 2.2.2.

What the store changes, and how this module keeps the old reading:

- a tracing is int16 microvolts at 500 Hz in the canonical lead order.  PTB-XL,
  ACS-ECG and the Challenge partitions are WFDB at 1,000 units per millivolt, so
  ``microvolts / 1000`` is the float ``wfdb.rdrecord`` returned, bit for bit;
  SPH's float16 millivolts are rounded to the nearest microvolt (0.5 µV at most);
- a non-finite sample is stored as 0 and counted (``waveform.n_nonfinite``), and a
  sample past int16 is clipped and counted (``n_clipped``).  Such a tracing comes
  back all NaN, so the chain excludes it as it excluded the publisher's record.
"""

from __future__ import annotations

import io
import json
import struct
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from .config import ACS_FLAGS, ACS_LABELLED_SPLIT, CARDIODB_PROJECT, N_LEADS, SAMPLING_RATE_HZ
from .ingest import CANONICAL_LEADS, Record

__all__ = [
    "ACS_DATASET",
    "BRIDGE",
    "CHALLENGE_SOURCE",
    "PTBXL_SOURCE",
    "SPH_SOURCE",
    "acs_table",
    "available",
    "challenge_source",
    "distributed_headers",
    "ptbxl_database",
    "ptbxl_record_id",
    "query",
    "scp_statements",
    "sph_metadata",
    "table",
    "tracings",
]

BRIDGE = Path(__file__).with_name("cardiodb_bridge.py")
MICROVOLTS_PER_MILLIVOLT = 1000.0
HEADER = struct.Struct("<qqq")

PTBXL_SOURCE = "physionet/ptb-xl"
SPH_SOURCE = "figshare/sph"
ACS_SOURCE = "figshare/acs"
ACS_DATASET = "acs"
ACS_VOCABULARY = "ACS-ECG"
CHALLENGE_SOURCE = "challenge-2021"
CHALLENGE_DX = "header comment Dx"


def _command(project: Path) -> list[str]:
    return ["uv", "run", "--project", str(project), "--no-sync", "python", "-I", str(BRIDGE)]


def available(project: Path = CARDIODB_PROJECT) -> bool:
    """Whether an cardiodb checkout sits where ``CARDIODB_PROJECT`` says (not in CI)."""
    return (project / "pyproject.toml").is_file() and (project / ".venv").is_dir()


def _ask(request: dict[str, Any], project: Path = CARDIODB_PROJECT) -> bytes:
    done = subprocess.run(
        _command(project), input=json.dumps(request).encode(), capture_output=True, check=False
    )
    if done.returncode:
        raise RuntimeError(f"cardiodb at {project}: {done.stderr.decode(errors='replace')[-2000:]}")
    return done.stdout


def table(name: str, corpus: str | None = None) -> pd.DataFrame:
    """A store table (``corpus`` names a clinical directory), as written."""
    return pd.read_parquet(io.BytesIO(_ask({"op": "table", "name": name, "corpus": corpus})))


def query(sql: str, layers: Sequence[str] = ()) -> pd.DataFrame:
    """``cardiodb query --with <layers>``: the tracings layer and the layers named."""
    request = {"op": "query", "sql": sql, "layers": list(layers)}
    return pd.read_parquet(io.BytesIO(_ask(request)))


def _read_exact(stream: IO[bytes], n: int) -> bytes:
    data = stream.read(n)
    if len(data) != n:
        raise EOFError(f"the bridge stopped after {len(data)} of {n} bytes")
    return data


def tracings(
    record_ids: Sequence[str], project: Path = CARDIODB_PROJECT
) -> Iterator[tuple[str, Record]]:
    """Each store record as a ``Record``: (12, T) float32 millivolts at 500 Hz."""
    ids = list(record_ids)
    with subprocess.Popen(
        _command(project), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as child:
        assert child.stdin is not None and child.stdout is not None and child.stderr is not None
        child.stdin.write(json.dumps({"op": "tracings", "ids": ids}).encode())
        child.stdin.close()
        try:
            for record_id in ids:
                n, nonfinite, clipped = HEADER.unpack(_read_exact(child.stdout, HEADER.size))
                raw = _read_exact(child.stdout, n * N_LEADS * 2)
                microvolts = np.frombuffer(raw, dtype="<i2").reshape(n, N_LEADS)
                signal = (microvolts / MICROVOLTS_PER_MILLIVOLT).T.astype(np.float32)
                if nonfinite or clipped:
                    signal = np.full_like(signal, np.nan)
                yield (
                    record_id,
                    Record(np.ascontiguousarray(signal), CANONICAL_LEADS, SAMPLING_RATE_HZ),
                )
        except EOFError as error:
            child.wait()
            message = child.stderr.read().decode(errors="replace")[-2000:]
            raise RuntimeError(f"cardiodb at {project}: {message}") from error


# --------------------------------------------------------------------------
# The publishers' tables, as each corpus shipped them
# --------------------------------------------------------------------------


def ptbxl_database() -> pd.DataFrame:
    """PTB-XL's ``ptbxl_database.csv``, indexed by ``ecg_id``."""
    frame = table("ptbxl_database", "ptb-xl").drop(columns="record_id")
    return frame.set_index("ecg_id")


def ptbxl_record_id(filename_hr: str) -> str:
    """The store's id of the record ``filename_hr`` names (``records500/00000/00001_hr``)."""
    return f"{PTBXL_SOURCE}:{Path(filename_hr).name}"


def scp_statements() -> pd.DataFrame:
    """PTB-XL's ``scp_statements.csv``, indexed by statement as ``index_col=0`` reads it."""
    frame = table("scp_statements", "ptb-xl").set_index("scp_code")
    frame.index.name = None
    return frame


def sph_metadata() -> pd.DataFrame:
    """SPH's ``metadata.csv``."""
    return table("metadata", "sph").drop(columns=["record_id", "native_record_id"])


def acs_table(split: str) -> pd.DataFrame:
    """ACS-ECG's ``CSV/<split>.csv``, in its row order: ``Patient_id``,
    ``ecg_row_record`` and, for the labelled split, the 0/1 columns ``ACS_FLAGS``.

    Only these columns are read here; the store keeps no ``gender``, ``age``,
    ``Time_Interval`` or ``ecg_med_record``.
    """
    # A filtered scan keeps the table's row order, which is the CSV's; a join does
    # not promise any, so the split and the patient are joined here, order kept.
    records = query(
        f"SELECT record_id, native_record_id FROM record WHERE source_id = '{ACS_SOURCE}'"
    )
    splits = query(
        f"SELECT record_id, split_published FROM split WHERE record_id LIKE '{ACS_SOURCE}:%'"
    )
    patients = query(
        f"SELECT record_id, native_patient_id FROM patient WHERE record_id LIKE '{ACS_SOURCE}:%'"
    )
    records = records.merge(splits, on="record_id", how="left", validate="1:1")
    records = records.merge(patients, on="record_id", how="left", validate="1:1")
    records = records[records["split_published"] == split].reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "Patient_id": records["native_patient_id"].to_numpy(),
            "ecg_row_record": [f"{r}.dat" for r in records["native_record_id"]],
        }
    )
    if split != ACS_LABELLED_SPLIT:
        return frame
    labels = query(
        "SELECT split_part(record_id, ':', 2) AS native, code FROM label"
        f" WHERE record_id LIKE '{ACS_SOURCE}:%' AND vocabulary = '{ACS_VOCABULARY}'",
        ["statements"],
    )
    positive = set(zip(labels["native"], labels["code"], strict=True))
    natives = list(records["native_record_id"])
    for code in ACS_FLAGS:
        frame[code] = np.array([(n, code) in positive for n in natives], dtype=np.int64)
    return frame


def challenge_source(partition: str) -> pd.DataFrame:
    """One Challenge-2021 partition as its headers state it, in path order.

    Columns: ``record_id`` (the header's), ``store_id``, ``path`` (the header
    file's path in the bundle), ``dx`` (SNOMED CT codes, as the ``# Dx:`` line
    lists them), ``sampling_rate_hz``, ``n_samples``, ``age``, ``sex``.
    """
    source = f"{CHALLENGE_SOURCE}/{partition}"
    records = query(
        "SELECT r.record_id AS store_id, r.native_record_id AS record_id,"
        " f.relative_path AS path, r.sampling_rate_hz, r.n_samples, h.age, h.sex"
        " FROM record r JOIN source_file f ON f.file_id = r.header_file_id"
        " LEFT JOIN challenge2021.header h USING (record_id)"
        f" WHERE r.source_id = '{source}' ORDER BY f.relative_path",
        ["clinical/challenge2021"],
    )
    labels = query(
        "SELECT record_id AS store_id, code FROM label"
        f" WHERE record_id LIKE '{source}:%' AND source_field = '{CHALLENGE_DX}'",
        ["statements"],
    )
    dx = labels.groupby("store_id", sort=False)["code"].agg(tuple)
    records["dx"] = [dx.get(s, ()) for s in records["store_id"]]
    return records


def distributed_headers(partition: str) -> set[str]:
    """The record names whose ``.hea`` the partition's ``SHA256SUMS.txt`` declares."""
    listed = query(
        "SELECT relative_path FROM source_file"
        f" WHERE source_id = '{CHALLENGE_SOURCE}/{partition}'"
        " AND relative_path LIKE '%.hea' AND sha256_declared IS NOT NULL"
    )
    return {Path(p).stem for p in listed["relative_path"]}
