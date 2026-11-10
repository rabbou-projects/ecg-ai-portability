"""The calibration statistics recover known miscalibration on simulated labels."""

from __future__ import annotations

import numpy as np
import pytest
from echonext_calibration import logit, statistics


def labels_from(true_risk: np.ndarray, seed: int = 1) -> np.ndarray:
    return (np.random.default_rng(seed).random(len(true_risk)) < true_risk).astype(int)


def test_calibrated_probabilities_give_unit_slope_and_zero_intercept() -> None:
    p = np.random.default_rng(0).uniform(0.02, 0.98, 200_000)
    s = statistics(p, labels_from(p))
    assert s["slope"] == pytest.approx(1.0, abs=0.02)
    assert s["intercept"] == pytest.approx(0.0, abs=0.02)
    assert s["observed_expected"] == pytest.approx(1.0, abs=0.01)


def test_known_slope_and_offset_are_recovered() -> None:
    # True risk on the logit scale is 0.5 * logit(p) - 0.4: the model is overconfident
    # (slope 0.5) and the logistic fit of the label on logit(p) must return those two.
    p = np.random.default_rng(0).uniform(0.02, 0.98, 200_000)
    risk = 1 / (1 + np.exp(-(0.5 * logit(p) - 0.4)))
    s = statistics(p, labels_from(risk))
    assert s["slope"] == pytest.approx(0.5, abs=0.02)
    assert s["observed_expected"] == pytest.approx(risk.mean() / p.mean(), abs=0.01)
