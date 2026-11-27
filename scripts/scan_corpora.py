"""Run every record of every corpus through the ingestion chain and write the
counts to results/ingest_report.json: records read, records excluded with the
reason, and the deviations the chain had to name.

Reads in chunks of a thousand records so that no corpus is ever held in memory
whole; nothing transformed is written anywhere.

Usage: .venv/bin/python scripts/scan_corpora.py
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from ecs.config import N_LEADS, RESULTS_DIR, WINDOW_SAMPLES
from ecs.ingest import Corpus, load_acs, load_ptbxl, load_sph
from ecs.store import acs_table, ptbxl_database, sph_metadata

CHUNK = 1000


def scan(
    name: str, ids: Sequence[str], load: Callable[[Sequence[str]], Corpus]
) -> dict[str, object]:
    started = time.time()
    total = Corpus(name, np.empty((0, N_LEADS, WINDOW_SAMPLES), np.float32), [], {})
    for start in range(0, len(ids), CHUNK):
        chunk = load(ids[start : start + CHUNK])
        total.ids.extend(chunk.ids)
        total.excluded.update(chunk.excluded)
        total.n_resampled += chunk.n_resampled
        total.n_cropped += chunk.n_cropped
        total.notes = chunk.notes
        done = min(start + CHUNK, len(ids))
        print(f"  {name}: {done:>6} / {len(ids)}  {time.time() - started:6.0f} s", flush=True)
    assert len(total.ids) + len(total.excluded) == len(ids)
    return {
        "n_records": len(ids),
        "n_kept": len(total.ids),
        "n_excluded": len(total.excluded),
        "excluded": total.excluded,
        "deviations": total.deviations,
        "seconds": round(time.time() - started, 1),
    }


def main() -> int:
    database = ptbxl_database()
    metadata = sph_metadata()
    train = acs_table("train")
    test = acs_table("test")
    acs_ids = [str(f).removesuffix(".dat") for f in pd.concat([train, test])["ecg_row_record"]]
    out = {
        "ptbxl": scan(
            "ptbxl",
            [str(i) for i in database.index],
            lambda ids: load_ptbxl(database, ids=[int(i) for i in ids]),
        ),
        "sph": scan("sph", list(metadata["ECG_ID"]), lambda ids: load_sph(metadata, ids=ids)),
        "acs": scan("acs", acs_ids, lambda ids: load_acs(train, ids=ids)),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "ingest_report.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    for name, counts in out.items():
        print(name, counts)
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
