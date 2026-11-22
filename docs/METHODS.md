# MOSAIC — Methods (as implemented)

## Data
MIMIC-IV `hosp` and `icu` modules, loaded as DuckDB views over the CSVs (`src/db.py`). Tables: patients, admissions, diagnoses_icd, labevents, prescriptions, transfers, icustays.

## Event timeline (`src/timeline.py`)
One row per event: `subject_id, hadm_id, event_time, event_type, code, value`.
Event types: admission, discharge, diagnosis (ICD, anchored at discharge time), lab (itemid + value), medication, icu_admit.

## Cohort and outcome (`src/labels.py`)
- **Index admission:** age ≥ 18 at admission, discharged alive.
- **Label = 1** if any later admission starts within 365 days after index discharge AND includes an ICU stay or in-hospital death.
- Age = `anchor_age + (year(admittime) − anchor_year)`.

## Features (`src/features.py`)
All features use only data up to the index discharge; no post-discharge information.

| Set | Features |
| --- | --- |
| Isolated (index admission only) | age, gender, admission type, length of stay, ICU this stay, # diagnoses, # distinct drugs, # labs, last value of creatinine, haemoglobin, WBC, sodium, potassium, glucose, lactate |
| Longitudinal (adds history) | # prior admissions, # prior ICU stays, # prior emergency admissions, total prior length of stay, days since last discharge, cumulative distinct diagnoses, mean and slope (per day) of each key lab across all admissions to date |

Missing values: median imputation plus missing-indicator columns; one-hot for categoricals; scaling for logistic regression only.

## Models (`src/train.py`)
- Logistic regression (balanced class weights).
- LightGBM (300 trees, learning rate 0.05, 15 leaves, balanced).

## Evaluation (`src/evaluate.py`)
- 5 repeats × 5-fold `StratifiedGroupKFold`, grouped by patient.
- Metrics: AUPRC (primary), AUROC; Brier score in the single-split run.
- **H1:** paired difference longitudinal − isolated per repeat, same folds.
- **H2:** AUPRC by number of prior admissions (0 / 1–2 / 3+).
- **H3:** mean |SHAP| (TreeExplainer) on the longitudinal LightGBM.

## Reliability (`src/reliability.py`)
Input: out-of-fold probabilities from the same 5 × 5 patient-grouped CV, **averaged over the 5 repeats** to give one score per index admission per model and feature set.

**R1 – Calibration.**
- Brier score and Brier skill score (1 − Brier / Brier of predicting the prevalence; > 0 means better than the base rate).
- Calibration intercept (calibration-in-the-large): `a` in logit P(y) = a + logit(p), slope fixed at 1. < 0 means risk is over-predicted.
- Calibration slope: `b` in logit P(y) = a + b·logit(p). < 1 means predictions are too extreme.
- ECE over 10 quantile bins; reliability curves with Wilson 95% CIs.
- Recalibration: Platt scaling (logistic fit on logit p), **cross-fitted** with 5-fold `GroupKFold` by patient, so each patient's score is mapped by a fit learned on other patients only. Isotonic regression is not used because it overfits at demo size.
- Expected finding: both models use `class_weight="balanced"`, which pushes predicted risk towards 50% regardless of the true rate, so raw intercepts should be clearly negative. This weighting does not affect AUPRC/AUROC, but must be corrected before probabilities are shown to anyone.

**R2 – Patient-level bootstrap for H1.**
Patients (not admissions) are resampled with replacement, 2,000 times; each resample takes all of that patient's index admissions. Isolated and longitudinal scores are evaluated on the same resample (paired), giving a 95% percentile CI for ΔAUPRC and ΔAUROC, plus the share of resamples with Δ > 0. This replaces the split-to-split SD as the uncertainty estimate for H1, since the SD ignores sampling variation across patients.

**R3 – Conformal prediction sets.**
Nonconformity for label c is 1 − P(c). Thresholds are set separately for y = 0 and y = 1 (Mondrian, class-conditional), so the coverage target also holds for patients who deteriorate. Cross-conformal: 5-fold `GroupKFold` by patient; each fold's thresholds are set on the other patients' out-of-fold scores. Reported at α = 0.1 and 0.2: coverage (overall, per class), shares of {low}, {high}, {uncertain} and empty sets, the share of events flagged high, and the PPV of "high" / NPV of "low" calls.
Caveat: the out-of-fold scores come from models trained on overlapping data, so exchangeability, and therefore the coverage guarantee, is approximate. Coverage is measured empirically and reported.

## Tests (`tests/`, `pytest -q`)
- Unit tests on simulated scores with known answers (perfect calibration, shifted and overconfident scores, conformal coverage, bootstrap behaviour).
- Cohort rules (adults, discharged alive, one row per index admission).
- **Future-leakage test:** injects a new admission, an ICU stay and extreme labs after every patient's last discharge. Labels may change, but every feature of every original index admission must be identical. Checked by mutation: removing the `charttime <= dischtime` guard makes this test fail.
- End-to-end run of `run_reliability.py` on fake data.
- CI: GitHub Actions on Python 3.11 and 3.12 for every push.

## Reproduce
```bash
python run_pipeline.py      # single split, quick check
python run_analysis.py      # full analysis, figures, reports/summary.md
python run_reliability.py   # R1-R3, figures 05-07, reports/reliability_summary.md
pytest -q                   # tests on fake data
```
