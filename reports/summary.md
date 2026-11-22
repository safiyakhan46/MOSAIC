# MOSAIC results (auto-generated)

Data: `data/mimic-iv-clinical-database-demo-2.2`

## Cohort
|   n_index |   n_patients |   n_events |   pos_rate |
|----------:|-------------:|-----------:|-----------:|
|       260 |           95 |         61 |      0.235 |

## H1 – performance (mean, SD over 5 repeats of 5-fold patient-grouped CV)
|                              |   auprc_mean |   auprc_std |   auroc_mean |   auroc_std |
|:-----------------------------|-------------:|------------:|-------------:|------------:|
| ('isolated', 'lightgbm')     |        0.331 |       0.022 |        0.648 |       0.02  |
| ('isolated', 'logreg')       |        0.374 |       0.009 |        0.669 |       0.009 |
| ('longitudinal', 'lightgbm') |        0.319 |       0.021 |        0.631 |       0.016 |
| ('longitudinal', 'logreg')   |        0.477 |       0.031 |        0.714 |       0.028 |

## H1 – paired difference (longitudinal − isolated)
| model    |   delta_auprc_mean |   delta_auprc_std |   delta_auprc_min |   delta_auprc_max |   delta_auroc_mean |   delta_auroc_std |   delta_auroc_min |   delta_auroc_max | repeats_improved_auprc   |
|:---------|-------------------:|------------------:|------------------:|------------------:|-------------------:|------------------:|------------------:|------------------:|:-------------------------|
| lightgbm |             -0.012 |             0.01  |            -0.019 |             0.005 |             -0.017 |             0.027 |            -0.05  |             0.02  | 1/5                      |
| logreg   |              0.102 |             0.029 |             0.066 |             0.133 |              0.045 |             0.031 |             0.009 |             0.083 | 5/5                      |

## H2 – AUPRC by prior admissions
| model    | group     |   auprc |   auroc |   n |   pos | features     |
|:---------|:----------|--------:|--------:|----:|------:|:-------------|
| lightgbm | 0 prior   |   0.269 |   0.618 |  95 |    19 | isolated     |
| lightgbm | 1-2 prior |   0.314 |   0.647 |  69 |    14 | isolated     |
| lightgbm | 3+ prior  |   0.438 |   0.649 |  96 |    28 | isolated     |
| logreg   | 0 prior   |   0.362 |   0.658 |  95 |    19 | isolated     |
| logreg   | 1-2 prior |   0.494 |   0.758 |  69 |    14 | isolated     |
| logreg   | 3+ prior  |   0.383 |   0.597 |  96 |    28 | isolated     |
| lightgbm | 0 prior   |   0.247 |   0.541 |  95 |    19 | longitudinal |
| lightgbm | 1-2 prior |   0.28  |   0.591 |  69 |    14 | longitudinal |
| lightgbm | 3+ prior  |   0.483 |   0.722 |  96 |    28 | longitudinal |
| logreg   | 0 prior   |   0.358 |   0.683 |  95 |    19 | longitudinal |
| logreg   | 1-2 prior |   0.387 |   0.678 |  69 |    14 | longitudinal |
| logreg   | 3+ prior  |   0.624 |   0.762 |  96 |    28 | longitudinal |

## H3 – top SHAP features
| feature           |   mean_abs_shap | type                  |
|:------------------|----------------:|:----------------------|
| prior_los_days    |          0.9027 | history (cross-visit) |
| n_prior_icu       |          0.86   | history (cross-visit) |
| icu_this_stay     |          0.8553 | index admission       |
| last_hemoglobin   |          0.6585 | index admission       |
| n_prior_emergency |          0.5942 | history (cross-visit) |
| n_dx_cumulative   |          0.5726 | history (cross-visit) |
| slope_potassium   |          0.5317 | history (cross-visit) |
| last_glucose      |          0.4619 | index admission       |
| age               |          0.4436 | index admission       |
| mean_hemoglobin   |          0.358  | history (cross-visit) |
| mean_potassium    |          0.35   | history (cross-visit) |
| n_dx              |          0.3483 | index admission       |
| slope_wbc         |          0.325  | history (cross-visit) |
| mean_wbc          |          0.2912 | history (cross-visit) |
| mean_glucose      |          0.2775 | history (cross-visit) |