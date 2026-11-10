"""EchoNext calibration: does each model's probability match the observed risk, by setting?

For the composite label (moderate or worse structural heart disease) and the three
models the report compares, on the calibration inpatients and on each care
setting of the test split:

  observed_expected  the observed prevalence divided by the mean predicted
                     probability, 1 when the model is calibrated in the large;
  intercept          calibration-in-the-large, the intercept of a logistic
                     model of the label with the logit of the probability as
                     offset, 0 when calibrated;
  slope              the coefficient of a logistic model of the label on the
                     logit of the probability, 1 when calibrated;
  deciles            the mean predicted probability and the observed
                     prevalence in each tenth of the scores, for a curve.

Each of the first three carries a 95% percentile interval from 2,000 draws
resampling ECGs within the setting, the unit the report's other proportions use.

Reads the stored scores under ``$ECS_ECHONEXT_DERIVED/scores`` and the EchoNext
metadata; writes ``results/echonext_calibration.json``.  Under a minute.

Usage: uv run python scripts/echonext_calibration.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))

from echonext_clinical import CONTEXTS, STRONGEST, composite_scores

from ecs.config import REPO_ROOT, RESULTS_DIR
from ecs.echonext import COMPOSITE, read_metadata, transfer_cohorts

DRAWS = 2000
DECILES = 10
# Probabilities are clipped away from 0 and 1 before the logit, so that a score
# stored as exactly 0 or 1 stays finite; 1e-6 is far below any decile's width.
EPS = 1e-6
PRODUCERS = ["scripts/echonext_calibration.py", "scripts/echonext_clinical.py"]


def logit(p: NDArray[np.float64]) -> NDArray[np.float64]:
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def logistic_fit(
    y: NDArray[np.int_], x: NDArray[np.float64], offset: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Maximum-likelihood coefficients of logit P(y) = offset + X b, by Newton's method."""
    beta = np.zeros(x.shape[1])
    for _ in range(50):
        eta = offset + x @ beta
        mu = 1 / (1 + np.exp(-eta))
        grad = x.T @ (y - mu)
        hess = x.T @ (x * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            return beta
    raise RuntimeError("logistic fit did not converge in 50 Newton steps")


def statistics(p: NDArray[np.float64], y: NDArray[np.int_]) -> dict[str, float]:
    z = logit(p)
    ones = np.ones((len(y), 1))
    intercept = logistic_fit(y, ones, z)[0]
    slope = logistic_fit(y, np.column_stack([ones[:, 0], z]), np.zeros(len(y)))[1]
    return {
        "observed_expected": float(y.mean() / p.mean()),
        "intercept": float(intercept),
        "slope": float(slope),
    }


def deciles(p: NDArray[np.float64], y: NDArray[np.int_]) -> list[dict[str, float]]:
    edges = np.quantile(p, np.linspace(0, 1, DECILES + 1))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, DECILES - 1)
    return [
        {
            "n": int((which == k).sum()),
            "mean_predicted": float(p[which == k].mean()),
            "observed": float(y[which == k].mean()),
        }
        for k in range(DECILES)
        if (which == k).any()
    ]


def calibration(p: NDArray[np.float64], y: NDArray[np.int_], seed: int = 0) -> dict[str, Any]:
    estimate = statistics(p, y)
    rng = np.random.default_rng(seed)
    drawn: dict[str, list[float]] = {name: [] for name in estimate}
    for _ in range(DRAWS):
        i = rng.integers(0, len(y), len(y))
        for name, value in statistics(p[i], y[i]).items():
            drawn[name].append(value)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_predicted": float(p.mean()),
    }
    for name, value in estimate.items():
        low, high = np.percentile(drawn[name], [2.5, 97.5])
        out[name] = {"estimate": value, "low": float(low), "high": float(high)}
    out["deciles"] = deciles(p, y)
    return out


def measure(meta: pd.DataFrame) -> dict[str, Any]:
    cohorts = transfer_cohorts(meta, "inpatient", CONTEXTS)
    rows = {"calibration_inpatients": cohorts.calibration} | {
        context: cohorts.targets[context] for context in CONTEXTS
    }
    y = meta[COMPOSITE].to_numpy(dtype=int)
    out: dict[str, Any] = {}
    for arm in STRONGEST:
        probs = composite_scores(arm, meta)
        out[arm] = {}
        for name, r in rows.items():
            p = probs[r]
            if np.isnan(p).any():
                raise ValueError(f"{arm} leaves ECGs of {name} unscored")
            out[arm][name] = calibration(p, y[r])
    return out


def main() -> None:
    from echonext_clinical import inputs

    from ecs.provenance import provenance_block

    start = time.time()
    result: dict[str, Any] = {
        "question": (
            "Calibration of each model's composite probability on the calibration inpatients "
            "and in each care setting of the test split."
        ),
        "label": COMPOSITE,
        "interval_rule": "2.5th and 97.5th percentiles of 2,000 draws resampling ECGs",
        "arms": measure(read_metadata()),
        "inputs": inputs(),
        "provenance": provenance_block(PRODUCERS),
    }
    result["seconds"] = round(time.time() - start, 1)
    path = RESULTS_DIR / "echonext_calibration.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path.relative_to(REPO_ROOT), result["seconds"], "s")


if __name__ == "__main__":
    main()
