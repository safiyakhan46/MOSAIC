# MOSAIC — Progress Log

Newest first. One entry per working session: what was done, what was decided, what's next.

## 2026-11-22 — Reliability layer + tests
**Done**
- `src/reliability.py` + `run_reliability.py`: R1 calibration (Brier skill, intercept, slope, ECE, reliability curves, cross-fitted Platt), R2 patient-level paired bootstrap for the H1 difference, R3 class-conditional cross-conformal prediction sets.
- Figures 05–07; `reports/reliability_summary.md`.
- Test suite (10 tests) incl. a future-leakage test, which was mutation-checked; GitHub Actions CI.
- Verified on fake data: conformal coverage hits 0.90 and 0.80 targets; recalibrated intercepts ≈ 0.

**Decisions**
- Uncertainty for H1 now comes from patient-level bootstrap, not split-to-split SD (addresses the limitation noted in RESULTS_demo.md).
- Keep `class_weight="balanced"` for ranking; report calibration both raw and recalibrated rather than changing the models.
- Platt over isotonic (sample too small for isotonic).

**Findings on the MIMIC-IV demo**
- H1 (logreg): ΔAUPRC +0.099, patient-bootstrap 95% CI −0.010 to +0.185 (96% of resamples > 0). The "5/5 repeats" result overstated certainty; the gain is promising but not conclusive at n = 95.
- Raw logreg over-predicts risk (mean 0.42 vs 0.235 observed); after recalibration, longitudinal logreg is the only model with clearly positive Brier skill (0.10).
- Conformal, 90% coverage: history raises the share of confident "low risk" calls from 18% to 28% (NPV 0.87 → 0.92).
- H1–H3 re-run with v0.3 reproduces v0.2 exactly.

**Next**
- Push v0.3 to GitHub (no data) and confirm the CI badge is green.
- Related-work section; Project 2 (FHIR fragmentation and linkage) as the future data-ingestion layer.

## 2026-11-21 — Pilot on MIMIC-IV demo
**Done**
- Loaded MIMIC-IV Clinical Database Demo v2.2 (100 patients) via DuckDB views; no database server needed.
- Built the event timeline: 79,307 lab, 18,087 medication, 4,506 diagnosis, 275 admission and 140 ICU events.
- Built cohort: 260 index admissions, 95 patients, 61 outcome events (23.5%).
- Added `src/evaluate.py`: 5×5 patient-grouped CV, paired H1 comparison, H2 subgroups, H3 SHAP.
- Added `src/figures.py` and `run_analysis.py`; all outputs aggregate-only in `reports/`.
- Wrote `docs/RESULTS_demo.md`.

**Findings (pilot)**
- Logistic regression: AUPRC 0.374 → 0.477 with history (+0.102, 5/5 repeats).
- LightGBM: no gain (−0.012) → likely overfitting at n=260.
- Gain concentrated in patients with 3+ prior admissions (0.383 → 0.624).
- Top two SHAP features are cross-visit (prior length of stay, prior ICU count).

**Decisions**
- Switched primary evaluation from one train/test split to repeated grouped CV (a single split had only 50 test rows).
- Patient mosaic figure uses a synthetic patient, never a real record.

**Blockers**
- PhysioNet credentialing needs a supervisor reference (searching).

**Next**
- Calibration curves; censoring sensitivity analysis.
- Streamlit mosaic demo on synthetic data.
- Related-work section (5–8 papers).

## 2026-11-20 — Scaffold
- Repo structure, config, DuckDB loader, timeline builder, cohort/labels, isolated + longitudinal features, models.
- Fake MIMIC-style generator for smoke tests; pipeline ran end to end.
- PhysioNet account created; credentialing form drafted (awaiting supervisor). CITI course completed (Oct 2026).

## 2026-11-15 — Project defined
- Capstone brief, proposal (RQ, H1–H3, methods, evaluation, ethics), 8-week plan, tech stack.
