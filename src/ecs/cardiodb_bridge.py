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
  ``(samples, 12)`` int16 microvolts ``cardiodb.loader.load_many`` reads at 500 Hz in
  the canonical lead order.  An id the store does not hold stops the answer with
  the loader's error on stderr.
"""

from __future__ import annotations

import io
import json
import struct
import sys
from typing import Any

MICROVOLTS_PER_MILLIVOLT = 1000
RATE_HZ = 500
CHUNK = 5000  # ids registered per waveform-index query


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
        for tracing in load_many(chunk, RATE_HZ, list(CANONICAL_LEADS), store):
            if tracing.held_rate_hz != RATE_HZ:
                raise LookupError(f"{tracing.record_id}: held at {tracing.held_rate_hz:g} Hz only")
            microvolts = np.rint(tracing.signal * MICROVOLTS_PER_MILLIVOLT).astype("<i2")
            nonfinite, clipped = counts[tracing.record_id]
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
