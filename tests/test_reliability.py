"""Unit tests for src/reliability.py on simulated scores where the right answer is known."""
import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score

from src.reliability import (calibration_intercept, calibration_metrics, calibration_slope,
                             cluster_bootstrap_delta, conformal_summary, crossfit_platt,
                             mondrian_conformal_sets)


@pytest.fixture(scope="module")
def calibrated():
    """y drawn from p itself -> perfectly calibrated by construction. 5 rows per 'patient'."""
    rng = np.random.default_rng(0)
    p = rng.beta(2, 5, 20000)
    y = rng.binomial(1, p)
    groups = np.repeat(np.arange(4000), 5)
    return y, p, groups


def test_perfect_calibration_is_recognised(calibrated):
    y, p, _ = calibrated
    m = calibration_metrics(y, p)
    assert abs(m["cal_intercept"]) < 0.05
    assert abs(m["cal_slope"] - 1) < 0.08
    assert m["ece"] < 0.02


def test_overprediction_gives_negative_intercept(calibrated):
    y, p, _ = calibrated
    assert calibration_intercept(y, expit(logit(p) + 1.0)) < -0.8


def test_overconfident_scores_give_slope_below_one(calibrated):
    y, p, _ = calibrated
    assert calibration_slope(y, expit(2 * logit(p))) == pytest.approx(0.5, abs=0.06)


def test_crossfit_platt_repairs_shifted_scores(calibrated):
    y, p, g = calibrated
    shifted = expit(logit(p) + 1.5)
    fixed = crossfit_platt(y, shifted, g)
    assert abs(calibration_intercept(y, fixed)) < 0.05
    assert abs(fixed.mean() - y.mean()) < 0.01


def test_conformal_hits_target_coverage_for_both_classes(calibrated):
    y, p, g = calibrated
    for alpha in (0.1, 0.2):
        s = conformal_summary(mondrian_conformal_sets(y, p, g, alpha=alpha))
        assert s["coverage_y0"] >= 1 - alpha - 0.02
        assert s["coverage_y1"] >= 1 - alpha - 0.02
        shares = s["share_low"] + s["share_high"] + s["share_uncertain"] + s["share_empty"]
        assert shares == pytest.approx(1.0)


def test_bootstrap_identical_models_give_zero_delta(calibrated):
    y, p, g = calibrated
    r = cluster_bootstrap_delta(y[:2000], p[:2000], p[:2000], g[:2000], n_boot=200)
    assert r["delta"] == 0 and r["delta_lo"] == 0 and r["delta_hi"] == 0
    assert r["n_patients"] == 400  # resamples patients, not rows


def test_bootstrap_detects_a_better_model(calibrated):
    y, p, g = calibrated
    rng = np.random.default_rng(1)
    noisy = expit(logit(p) + rng.normal(0, 2.0, len(p)))  # same signal, much more noise
    r = cluster_bootstrap_delta(y, noisy, p, g, metric=average_precision_score, n_boot=300)
    assert r["delta"] > 0 and r["delta_lo"] > 0
