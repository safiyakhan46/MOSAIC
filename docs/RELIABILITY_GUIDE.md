# Reliability guide — how to run, read and explain R1–R3

## 1. Run it (about 1 minute on the demo)
```bash
pip install -r requirements.txt pytest
pytest -q                      # 10 tests on fake data, should all pass
python run_reliability.py      # uses mimic_path in config.yaml (the demo)
```
Outputs: `reports/reliability_summary.md`, `reports/r1-r3_*.csv` (git-ignored), and `reports/figures/05-07_*.png`.

If the CI badge at the top of the README stays grey, check that the repo name in the badge URL matches your GitHub repo (it assumes `safiyakhan46/mosaic`).

## 2. Read the outputs

### R1 – Calibration: "does 30% mean 30%?"
| Number | Good value | What a bad value means |
| --- | --- | --- |
| `cal_intercept` | ≈ 0 | **< 0: over-predicts risk.** Expect this on raw scores, because `class_weight="balanced"` acts as if half the patients deteriorate. |
| `cal_slope` | ≈ 1 | < 1: predictions too extreme (overfitting, likely for LightGBM on 260 rows). > 1: too timid. |
| `brier_skill` | > 0 | ≤ 0: no better than telling everyone the average risk. |
| `ece` | small (< 0.05) | average gap between predicted and observed risk. |

**Key idea:** ranking (AUROC/AUPRC) and calibration are different properties. A model can rank patients well and still give wrong probabilities. Platt recalibration changes the probabilities but never the ranking, so AUROC stays the same and the intercept moves to about 0. If the recalibrated *slope* is still far from 1, the model's spread is wrong, and a single rescale can only partly fix that.

### R2 – Patient-level bootstrap: "is the H1 gain real?"
- `delta`, `delta_lo`, `delta_hi`: ΔAUPRC with its 95% CI. **If the CI excludes 0, H1 is supported at this sample size.** If it crosses 0, say "a positive trend, not yet significant at n = 95 patients". That is honest, and it is the reason full MIMIC-IV is needed.
- Why patients and not rows: one patient's admissions are correlated. Resampling rows would treat 3 admissions of the same person as 3 independent pieces of evidence, which makes the CI too narrow.
- Expect a much wider CI than the ±0.03 SD in RESULTS_demo.md. That is the point: the SD measured only how much the score moved between CV splits.

### R3 – Conformal sets: "when can the model commit?"
Each discharge gets a set: **{low}**, **{high}** or **{uncertain}**. At α = 0.1 the true outcome is inside the set ≥ 90% of the time, separately for patients who did and did not deteriorate.
- `share_uncertain`: how often the model honestly says "I can't tell". With AUROC ≈ 0.7, most patients will be uncertain at 90% coverage. **That is a correct result, not a bug.**
- Compare isolated vs longitudinal: **if history lets the model commit for more patients at the same guaranteed coverage, that is a clinically meaningful way to state H1.**
- Always read `ppv_high` with `share_high`. For a weak model, many "high" calls are the α errors it is allowed to make, so PPV can sit close to the base rate.

## 3. Add to RESULTS_demo.md
Add one short section per R, each with a figure, a table row or two, and a "Reading:" paragraph, in the same style as H1–H3. Add one line to Limitations: conformal coverage is approximate because out-of-fold scores come from overlapping training sets.

## 4. How to explain it (interviews, SOP, supervisor emails)
- "I separated *discrimination* from *calibration*. The class-balanced models ranked well but over-predicted risk, so I added cross-fitted recalibration and reported both."
- "My first uncertainty estimate came from CV split variance, which understates it. I replaced it with a paired patient-level bootstrap."
- "I used Mondrian conformal prediction, so the coverage guarantee also holds for the minority class, which is the patients who deteriorate."
- "I wrote an automated test that injects future events and checks that no feature changes. I confirmed it fails when I deliberately remove the leakage guard."

## 5. Concepts to read up on (about 2 hours total)
- Van Calster et al., "Calibration: the Achilles heel of predictive analytics", *BMC Medicine* 2019.
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction", arXiv 2107.07511 (sections 1–2, plus the class-conditional part).
- Efron & Tibshirani on the bootstrap. Any short note on the *cluster* bootstrap is enough.
