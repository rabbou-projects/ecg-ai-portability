"""The half of ``ecs.store`` that runs inside cardiodb's own environment.

``ecs.store`` starts this file with cardiodb's interpreter (``uv run --project
<cardiodb> --no-sync python -I``), writes one JSON request to its stdin and reads
the answer from its stdout.  It imports ``cardiodb`` and nothing of this
repository: cardiodb is private and pins numpy 2, this repository is public and
pins numpy 1 for torch 2.2.2, so the two never share an interpreter.

Requests:

- ``{"op": "table", "name": ..., "corpus": ...}``: ``Store().read``, as Parquet.
- ``{"op": "query", "sql": ..., "layers": [...]}``: ``connect(Store(), layers)``,
  the result as Parquet.
- ``{"op": "tracings", "ids": [...]}``: for each id in order, a header of three
  little-endian int64 (samples, non-finite samples, clipped samples) and then the
  ``(samples, 12)`` little-endian int32 microvolts ``cardiodb.loader.load_many`` reads at 500 Hz in
  the canonical lead order.  An id the store does not hold at 500 Hz gets the
  header ``(NOT_HELD, 0, 0)`` and no samples, and the answer goes on, as the
  publisher's reader went on past a file it could not open.
"""

from __future__ import annotations

import io
import json
import struct
import sys
from typing import Any

MICROVOLTS_PER_MILLIVOLT = 1000
RATE_HZ = 500
WIRE = "<i4"  # int32: cardiodb holds a source past ±32,767 µV in int32
CHUNK = 5000  # ids registered per waveform-index query
NOT_HELD = -1  # the sample count written for an id the store does not hold


def _parquet(frame: Any) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _tracings(ids: list[str], out: Any) -> None:
    import numpy as np
    import pandas as pd
    from cardiodb.ingest import CANONICAL_LEADS
    from cardiodb.loader import load_many
    from cardiodb.store import Store, connect

    store = Store()
    connection = connect(store)
    for begin in range(0, len(ids), CHUNK):
        chunk = ids[begin : begin + CHUNK]
        connection.register("wanted", pd.DataFrame({"record_id": chunk}))
        counts = {
            record_id: (int(nonfinite), int(clipped))
            for record_id, nonfinite, clipped in connection.execute(
                "SELECT record_id, n_nonfinite, n_clipped FROM waveform"
                f" JOIN wanted USING (record_id) WHERE rate_hz = {RATE_HZ}"
            ).fetchall()
        }
        connection.unregister("wanted")
        held = iter(
            load_many([i for i in chunk if i in counts], RATE_HZ, list(CANONICAL_LEADS), store)
        )
        for record_id in chunk:
            if record_id not in counts:
                out.write(struct.pack("<qqq", NOT_HELD, 0, 0))
                continue
            tracing = next(held)
            if tracing.record_id != record_id or tracing.held_rate_hz != RATE_HZ:
                raise LookupError(f"{record_id}: the loader answered {tracing.record_id}")
            microvolts = np.rint(tracing.signal * MICROVOLTS_PER_MILLIVOLT).astype(WIRE)
            nonfinite, clipped = counts[record_id]
            out.write(struct.pack("<qqq", microvolts.shape[0], nonfinite, clipped))
            out.write(np.ascontiguousarray(microvolts).tobytes())


def main() -> None:
    request = json.loads(sys.stdin.read())
    out = sys.stdout.buffer
    op = request["op"]
    if op == "table":
        from cardiodb.store import Store

        out.write(_parquet(Store().read(request["name"], request.get("corpus"))))
    elif op == "query":
        from cardiodb.store import Store, connect

        connection = connect(Store(), request.get("layers", []))
        out.write(_parquet(connection.execute(request["sql"]).fetch_df()))
    elif op == "tracings":
        _tracings(list(request["ids"]), out)
    else:
        raise ValueError(f"no op {op!r}")
    out.flush()


if __name__ == "__main__":
    main()
