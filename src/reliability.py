"""Reliability layer: can MOSAIC's risk scores be trusted, and how sure are we about H1?

Three questions, three tools:

R1  Calibration    - when the model says 30% risk, do ~30% of those patients deteriorate?
                     Brier score, Brier skill, calibration intercept and slope, ECE,
                     reliability curves; cross-fitted Platt recalibration.
R2  Uncertainty    - is the H1 gain (longitudinal - isolated) bigger than chance?
                     Patient-level (cluster) bootstrap 95% CI for the paired difference.
R3  Conformal      - for each discharge, can the model commit to "low" or "high" risk
                     at a guaranteed error rate, or must it say "uncertain"?
                     Class-conditional (Mondrian) cross-conformal prediction sets.

Every resampling / calibration split is grouped by patient, matching the rest of MOSAIC.
All functions take plain arrays so they work on any model's predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

EPS = 1e-6
# Logistic fits below use C=1e10 (effectively unpenalised; works on every sklearn version)


def _lp(p) -> np.ndarray:
    """Linear predictor (log-odds) with clipping so 0/1 probabilities don't explode."""
    return logit(np.clip(np.asarray(p, float), EPS, 1 - EPS))


# --------------------------------------------------------------------------- R1
def calibration_intercept(y, p) -> float:
    """Calibration-in-the-large: a in logit P(y=1) = a + logit(p), slope fixed at 1.

    0 = well calibrated on average; < 0 = model over-predicts risk; > 0 = under-predicts.
    """
    y, lp = np.asarray(y, float), _lp(p)
    f = lambda a: np.sum(y - expit(a + lp))  # score equation, monotone in a
    return float(brentq(f, -20, 20))


def calibration_slope(y, p) -> float:
    """b in logit P(y=1) = a + b * logit(p).

    1 = ideal; < 1 = predictions too extreme (typical of overfitting); > 1 = too timid.
    """
    lr = LogisticRegression(C=1e10, max_iter=1000)
    lr.fit(_lp(p).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0, 0])


def reliability_curve(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned predicted vs observed risk, with Wilson 95% CI on the observed rate."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), q=min(n_bins, len(d)), labels=False)
    g = d.groupby("bin").agg(p_mean=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    z = 1.96
    centre = (g["obs"] + z**2 / (2 * g["n"])) / (1 + z**2 / g["n"])
    half = z * np.sqrt(g["obs"] * (1 - g["obs"]) / g["n"] + z**2 / (4 * g["n"] ** 2)) / (1 + z**2 / g["n"])
    g["obs_lo"], g["obs_hi"] = centre - half, centre + half
    return g.reset_index(drop=True)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    c = reliability_curve(y, p, n_bins)
    return float(np.sum(c["n"] * np.abs(c["obs"] - c["p_mean"])) / c["n"].sum())


def calibration_metrics(y, p, n_bins: int = 10) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    prev = y.mean()
    brier = brier_score_loss(y, p)
    ref = prev * (1 - prev)  # Brier of always predicting the prevalence
    return {
        "brier": brier,
        "brier_skill": 1 - brier / ref if ref > 0 else np.nan,
        "cal_intercept": calibration_intercept(y, p),
        "cal_slope": calibration_slope(y, p),
        "ece": expected_calibration_error(y, p, n_bins),
        "mean_predicted": p.mean(),
        "observed_rate": prev,
    }


def crossfit_platt(y, p, groups, n_splits: int = 5) -> np.ndarray:
    """Platt-recalibrate out-of-fold: each patient's scores are mapped by a logistic
    fit learned on OTHER patients only, so the recalibrated metrics stay honest."""
    y, lp, groups = np.asarray(y), _lp(p).reshape(-1, 1), np.asarray(groups)
    out = np.empty(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(lp, y, groups):
        lr = LogisticRegression(C=1e10, max_iter=1000).fit(lp[tr], y[tr])
        out[te] = lr.predict_proba(lp[te])[:, 1]
    return out


# --------------------------------------------------------------------------- R2
def _patient_index(groups) -> list[np.ndarray]:
    groups = np.asarray(groups)
    _, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, bounds)


def cluster_bootstrap_delta(y, p_a, p_b, groups, metric=average_precision_score,
                            n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired patient-level bootstrap of metric(p_b) - metric(p_a).

    Patients (not rows) are resampled with replacement, because one patient's several
    admissions are correlated. Both models are scored on the SAME resample (paired).
    """
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    rows = _patient_index(groups)
    rng = np.random.default_rng(seed)
    a_vals, b_vals = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows), len(rows))
        idx = np.concatenate([rows[i] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        a_vals.append(metric(y[idx], p_a[idx]))
        b_vals.append(metric(y[idx], p_b[idx]))
    a_vals, b_vals = np.array(a_vals), np.array(b_vals)
    d = b_vals - a_vals
    return {
        "a": metric(y, p_a), "a_lo": np.percentile(a_vals, 2.5), "a_hi": np.percentile(a_vals, 97.5),
        "b": metric(y, p_b), "b_lo": np.percentile(b_vals, 2.5), "b_hi": np.percentile(b_vals, 97.5),
        "delta": metric(y, p_b) - metric(y, p_a),
        "delta_lo": np.percentile(d, 2.5), "delta_hi": np.percentile(d, 97.5),
        "share_boot_delta_gt0": float((d > 0).mean()),
        "n_boot": len(d), "n_patients": len(rows),
        "_deltas": d,  # kept for the histogram figure; dropped before saving
    }


# --------------------------------------------------------------------------- R3
SET_LABELS = {(False, False): "empty", (True, False): "low risk {0}",
              (False, True): "high risk {1}", (True, True): "uncertain {0,1}"}


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def mondrian_conformal_sets(y, p, groups, alpha: float = 0.1, n_splits: int = 5) -> pd.DataFrame:
    """Class-conditional cross-conformal prediction sets for a binary risk score.

    Nonconformity of label c is 1 - P(c). For each patient fold, thresholds are set on the
    other patients, separately for y=0 and y=1 ("Mondrian"), so coverage >= 1 - alpha holds
    for deteriorating patients too, not just on average.
    Returns one row per admission: contains_0, contains_1, set_label, covered.
    """
    y, p, groups = np.asarray(y), np.asarray(p, float), np.asarray(groups)
    s0, s1 = p, 1 - p  # nonconformity for label 0 and label 1
    has0, has1 = np.zeros(len(y), bool), np.zeros(len(y), bool)
    for cal, te in GroupKFold(n_splits=n_splits).split(p, y, groups):
        q0 = _conformal_quantile(s0[cal][y[cal] == 0], alpha)
        q1 = _conformal_quantile(s1[cal][y[cal] == 1], alpha)
        has0[te], has1[te] = s0[te] <= q0, s1[te] <= q1
    out = pd.DataFrame({"y": y, "contains_0": has0, "contains_1": has1})
    out["set_label"] = [SET_LABELS[(a, b)] for a, b in zip(has0, has1)]
    out["covered"] = np.where(y == 1, has1, has0)
    return out


def conformal_summary(sets: pd.DataFrame) -> dict:
    share = sets["set_label"].value_counts(normalize=True)
    pos = sets[sets["y"] == 1]
    return {
        "coverage": sets["covered"].mean(),
        "coverage_y0": sets.loc[sets["y"] == 0, "covered"].mean(),
        "coverage_y1": pos["covered"].mean(),
        "share_low": share.get("low risk {0}", 0.0),
        "share_high": share.get("high risk {1}", 0.0),
        "share_uncertain": share.get("uncertain {0,1}", 0.0),
        "share_empty": share.get("empty", 0.0),
        # of patients who DID deteriorate, how many were confidently flagged high risk?
        "events_flagged_high": (pos["set_label"] == "high risk {1}").mean() if len(pos) else np.nan,
        # when the model commits, how often is it right? (a weak model's commitments are
        # mostly its allowed alpha-errors, so always read these next to the shares)
        "ppv_high": sets.loc[sets["set_label"] == "high risk {1}", "y"].mean(),
        "npv_low": 1 - sets.loc[sets["set_label"] == "low risk {0}", "y"].mean(),
    }


# ----------------------------------------------------------------------- helper
def average_oof(oof: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Average out-of-fold probabilities over CV repeats -> one score per (model, admission).

    Adds subject_id so downstream splits/bootstraps can group by patient.
    """
    m = (oof.groupby(["model", "hadm_id"], as_index=False)
            .agg(label=("label", "first"), p=("p", "mean")))
    return m.merge(df[["hadm_id", "subject_id"]], on="hadm_id", how="left")


def auroc_safe(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
