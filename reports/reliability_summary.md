# MOSAIC reliability results (auto-generated)

Data: `data/mimic-iv-clinical-database-demo-2.2` · 260 index admissions, 95 patients · out-of-fold scores averaged over 5 repeats of 5-fold patient-grouped CV · 2000 patient-level bootstrap resamples

## R1 – calibration
| features     | model    | stage        |   brier |   brier_skill |   cal_intercept |   cal_slope |   ece |   mean_predicted |   observed_rate |
|:-------------|:---------|:-------------|--------:|--------------:|----------------:|------------:|------:|-----------------:|----------------:|
| isolated     | logreg   | raw          |   0.222 |        -0.237 |          -1.153 |       0.451 | 0.196 |            0.42  |           0.235 |
| isolated     | logreg   | recalibrated |   0.172 |         0.04  |          -0.006 |       0.731 | 0.046 |            0.236 |           0.235 |
| isolated     | lightgbm | raw          |   0.205 |        -0.14  |           0.2   |       0.256 | 0.151 |            0.213 |           0.235 |
| isolated     | lightgbm | recalibrated |   0.174 |         0.031 |           0.009 |       0.849 | 0.09  |            0.233 |           0.235 |
| longitudinal | logreg   | raw          |   0.196 |        -0.091 |          -0.772 |       0.367 | 0.156 |            0.319 |           0.235 |
| longitudinal | logreg   | recalibrated |   0.161 |         0.102 |          -0     |       0.813 | 0.063 |            0.235 |           0.235 |
| longitudinal | lightgbm | raw          |   0.199 |        -0.108 |           0.782 |       0.244 | 0.147 |            0.156 |           0.235 |
| longitudinal | lightgbm | recalibrated |   0.174 |         0.029 |          -0.007 |       0.803 | 0.036 |            0.236 |           0.235 |

## R2 – H1 paired difference, patient-level bootstrap 95% CI
| model    | metric   |   isolated |   longitudinal |   delta |   delta_lo |   delta_hi |   share_boot_delta_gt0 |
|:---------|:---------|-----------:|---------------:|--------:|-----------:|-----------:|-----------------------:|
| logreg   | auprc    |      0.377 |          0.475 |   0.099 |     -0.01  |      0.185 |                  0.962 |
| logreg   | auroc    |      0.676 |          0.728 |   0.052 |     -0.019 |      0.132 |                  0.922 |
| lightgbm | auprc    |      0.344 |          0.331 |  -0.013 |     -0.144 |      0.126 |                  0.417 |
| lightgbm | auroc    |      0.663 |          0.654 |  -0.008 |     -0.091 |      0.063 |                  0.424 |

## R3 – conformal prediction sets (class-conditional)
| features     | model    |   alpha |   coverage |   coverage_y1 |   share_low |   share_uncertain |   share_high |   events_flagged_high |   ppv_high |   npv_low |
|:-------------|:---------|--------:|-----------:|--------------:|------------:|------------------:|-------------:|----------------------:|-----------:|----------:|
| isolated     | logreg   |     0.1 |      0.904 |         0.902 |       0.177 |             0.696 |        0.127 |                 0.23  |      0.424 |     0.87  |
| isolated     | logreg   |     0.2 |      0.796 |         0.803 |       0.381 |             0.358 |        0.262 |                 0.443 |      0.397 |     0.879 |
| isolated     | lightgbm |     0.1 |      0.912 |         0.902 |       0.231 |             0.669 |        0.1   |                 0.148 |      0.346 |     0.9   |
| isolated     | lightgbm |     0.2 |      0.812 |         0.82  |       0.412 |             0.358 |        0.231 |                 0.361 |      0.367 |     0.897 |
| longitudinal | logreg   |     0.1 |      0.896 |         0.902 |       0.277 |             0.592 |        0.131 |                 0.213 |      0.382 |     0.917 |
| longitudinal | logreg   |     0.2 |      0.808 |         0.82  |       0.45  |             0.281 |        0.269 |                 0.508 |      0.443 |     0.906 |
| longitudinal | lightgbm |     0.1 |      0.912 |         0.918 |       0.15  |             0.735 |        0.115 |                 0.197 |      0.4   |     0.872 |
| longitudinal | lightgbm |     0.2 |      0.796 |         0.787 |       0.388 |             0.377 |        0.235 |                 0.344 |      0.344 |     0.871 |