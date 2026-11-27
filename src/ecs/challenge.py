"""The PhysioNet/CinC Challenge-2021 partitions, read where they sit on disk.

Four of the five rotation sources come out of this bundle: PTB-XL,
Chapman-Shaoxing with Ningbo, Georgia, and CPSC 2018 with its extension.  Every
record is a WFDB pair whose header carries the diagnoses as SNOMED CT codes on a
``# Dx:`` line, so the labels need no mapping of mine -- only the union that
``small_set`` defines.

Two things the bundle does not ship, and neither is guessed:

*Patient identifiers.*  A Challenge header has an age and a sex and no patient
key.  Splitting by patient therefore treats each record as its own
patient, which is right for Chapman-Shaoxing, Ningbo, Georgia and CPSC -- one
tracing per patient by construction -- and wrong for PTB-XL, whose 21,799
records come from 18,869 patients.  PTB-XL is the one corpus with a published
patient key, so its waveforms and its patients are read from the distribution
itself and only its SNOMED codes come from here.  The deviation is named in
``partition_deviations`` for every corpus that carries it.

*A count that matches the distribution.*  The bundle froze PTB-XL at 21,837
records; the distribution this study reads is v1.0.3, at 21,799.  The join keeps
the intersection and reports what fell out on either side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import MAPPINGS_DIR
from .small_set import SMALL_SET, aha_codes_for, refusals, snomed_codes_for
from .store import challenge_source, distributed_headers

__all__ = [
    "CHALLENGE_PARTITIONS",
    "completeness",
    "distributed_records",
    "PTBXL_RECORD_PREFIX",
    "challenge_small_set_labels",
    "join_ptbxl_to_challenge",
    "partition_deviations",
    "scan_source",
    "sph_small_set_labels",
]

# The Challenge directory names each rotation source is assembled from.
CHALLENGE_PARTITIONS: dict[str, tuple[str, ...]] = {
    "ptbxl": ("ptb-xl",),
    "chapman_ningbo": ("chapman_shaoxing", "ningbo"),
    "georgia": ("georgia",),
    "cpsc": ("cpsc_2018", "cpsc_2018_extra"),
}

# The bundle names each PTB-XL record HR<ecg_id zero-padded to five digits>.
# The join is not taken on trust: `test_label_map.py` compares the SNOMED labels
# it produces against PTB-XL's own SCP-ECG statements class by class, and a
# misaligned join would break that agreement rather than pass it.
PTBXL_RECORD_PREFIX = "HR"

NO_PATIENT_KEY = (
    "the Challenge bundle ships no patient identifier, so each record is its own "
    "patient for splitting"
)


def scan_source(source: str) -> pd.DataFrame:
    """Every record of one rotation source, indexed by the header's record name.

    Columns: ``partition``, ``path`` (the record's id in cardiodb's store), ``dx``
    (a tuple of SNOMED CT codes, as the ``# Dx:`` line lists them),
    ``sampling_rate_hz``, ``n_samples``, ``age``, ``sex``; in the bundle's path
    order within each partition.
    """
    frames = []
    for partition in CHALLENGE_PARTITIONS[source]:
        held = challenge_source(partition)
        if not len(held):
            raise FileNotFoundError(f"cardiodb's store holds no record of {partition}")
        frames.append(
            pd.DataFrame(
                {
                    "partition": partition,
                    "path": held["store_id"].to_numpy(),
                    "dx": held["dx"].to_numpy(),
                    "sampling_rate_hz": held["sampling_rate_hz"].astype(int).to_numpy(),
                    "n_samples": held["n_samples"].astype(int).to_numpy(),
                    "age": [_age(a) for a in held["age"]],
                    "sex": held["sex"].fillna("Unknown").to_numpy(),
                },
                index=pd.Index(held["record_id"], name="record_id"),
            )
        )
    table = pd.concat(frames)
    if table.index.has_duplicates:
        duplicates = sorted(table.index[table.index.duplicated()].unique())
        raise ValueError(f"{source}: repeated record ids {duplicates[:5]}")
    return table


def _age(stated: object) -> float:
    """The header's ``# Age:`` as years, or NaN: CPSC's ``-1`` and ``NaN`` are no age."""
    text = "" if stated is None or stated is pd.NA else str(stated).strip()
    return float(text) if text.replace(".", "", 1).isdigit() else float("nan")


def distributed_records(partition: str) -> set[str]:
    """The record ids the bundle's own SHA256SUMS.txt says this partition holds.

    cardiodb keeps each header file with the digest that manifest declares for it
    (``source_file.sha256_declared``), so the manifest is still the reference for
    whether the store's copy is complete -- a question a class count cannot answer
    on its own, because a missing record and a mis-parsed label look the same in a
    total.
    """
    listed = distributed_headers(partition)
    if not listed:
        raise ValueError(f"the store lists no .hea of {partition} with a declared digest")
    return listed


def completeness(source: str, table: pd.DataFrame) -> dict[str, list[str] | int]:
    """What the store holds of this source against what the bundle distributes.

    ``n_on_disk`` counts the records the store holds, read from the bundle on disk.
    """
    on_disk = set(table.index)
    listed: set[str] = set()
    for partition in CHALLENGE_PARTITIONS[source]:
        listed |= distributed_records(partition)
    missing = sorted(listed - on_disk)
    return {
        "n_distributed": len(listed),
        "n_on_disk": len(on_disk),
        "missing_from_this_copy": missing,
        "not_in_the_manifest": sorted(on_disk - listed),
    }


def partition_deviations(source: str, table: pd.DataFrame) -> list[str]:
    """What the chain could not make identical for this source."""
    out = [NO_PATIENT_KEY]
    rates = sorted({int(r) for r in table["sampling_rate_hz"]})
    if rates != [500]:
        out.append(f"sampling rates present: {rates}")
    lengths = sorted({int(n) for n in table["n_samples"]})
    if len(lengths) > 1:
        out.append(
            f"record lengths run from {min(lengths)} to {max(lengths)} samples; "
            "the first ten seconds are kept"
        )
    return out


def challenge_small_set_labels(table: pd.DataFrame, root: Path = MAPPINGS_DIR) -> pd.DataFrame:
    """One boolean column per small-set class, indexed like ``table``.

    A record is a positive of a class when its ``# Dx:`` line carries any of the
    SNOMED codes that class unions.
    """
    codes = {klass.key: snomed_codes_for(klass.key, root) for klass in SMALL_SET}
    columns = {
        key: table["dx"].apply(lambda dx, want=wanted: bool(want & set(dx)))
        for key, wanted in codes.items()
    }
    return pd.DataFrame(columns, index=table.index)


def sph_small_set_labels(metadata: pd.DataFrame, root: Path = MAPPINGS_DIR) -> pd.DataFrame:
    """One boolean column per small-set class for Shandong, indexed like ``metadata``.

    Shandong writes each diagnosis as an AHA code with an optional ``+modifier``;
    the base code is what carries the diagnosis (ambiguity ``aha-modifier-tokens``).
    A class with no AHA bridge is refused: its column is all-``False`` and the
    pair appears in ``small_set.refusals()``, which every reader of this table has
    to consult before using the column.
    """
    refused = {key for corpus, key in refusals() if corpus == "sph"}
    base = (
        metadata["AHA_Code"]
        .astype(str)
        .apply(
            lambda raw: {
                token.strip().split("+", 1)[0] for token in raw.split(";") if token.strip()
            }
        )
    )
    columns = {}
    for klass in SMALL_SET:
        if klass.key in refused:
            columns[klass.key] = pd.Series(False, index=metadata.index)
            continue
        wanted = aha_codes_for(klass.key, root)
        if not wanted:
            raise ValueError(
                f"{klass.key} has no AHA bridge and is not among the declared refusals"
            )
        columns[klass.key] = base.apply(lambda codes, want=wanted: bool(want & codes))
    return pd.DataFrame(columns, index=metadata.index)


def join_ptbxl_to_challenge(
    database: pd.DataFrame, challenge: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """PTB-XL's own rows carrying the Challenge's SNOMED codes.

    Returns the intersection, indexed by ``ecg_id`` and carrying ``patient_id``,
    ``strat_fold``, ``filename_hr`` and ``dx``, together with what fell out of
    each side.
    """
    numbers = []
    for record in challenge.index:
        text = str(record)
        rest = text.removeprefix(PTBXL_RECORD_PREFIX)
        if rest == text or not rest.isdigit():
            raise ValueError(
                f"{text!r} is not of the form {PTBXL_RECORD_PREFIX}<digits>; the join from the "
                "bundle's record names to PTB-XL's ecg_id assumes that form"
            )
        numbers.append(int(rest))
    ecg_id = pd.Series(numbers, index=challenge.index, name="ecg_id")
    dx_by_ecg_id = pd.Series(list(challenge["dx"]), index=np.asarray(ecg_id), name="dx")
    kept = database.index.intersection(dx_by_ecg_id.index)
    joined = database.loc[kept, ["patient_id", "strat_fold", "filename_hr"]].copy()
    joined["dx"] = dx_by_ecg_id.loc[kept]
    fell_out = {
        "in_the_distribution_not_in_the_bundle": [
            str(i) for i in database.index.difference(dx_by_ecg_id.index)
        ],
        "in_the_bundle_not_in_the_distribution": [
            f"{PTBXL_RECORD_PREFIX}{i:05d}" for i in dx_by_ecg_id.index.difference(database.index)
        ],
    }
    return joined, fell_out
