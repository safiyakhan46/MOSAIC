"""MOSAIC reliability analysis: R1 calibration, R2 patient-level bootstrap for H1, R3 conformal.

Usage: python run_reliability.py [--config config.yaml] [--n-boot 2000]
Outputs (aggregate only): reports/r*_*.csv(.md), reports/figures/05-07_*.png,
                          reports/reliability_summary.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src import figures
from src.db import connect, load_config
from src.evaluate import oof_predictions
from src.features import build_feature_sets
from src.labels import build_cohort
from src.reliability import (auroc_safe, average_oof, calibration_metrics, cluster_bootstrap_delta,
                             conformal_summary, crossfit_platt, mondrian_conformal_sets,
                             reliability_curve)
from src.timeline import build_timeline

MODELS = ("logreg", "lightgbm")


def main(config_path: str, n_boot: int | None = None, n_repeats: int = 5):
    cfg = load_config(config_path)
    rcfg = cfg.get("reliability", {})
    n_boot = n_boot or rcfg.get("n_boot", 2000)
    alphas = rcfg.get("alphas", [0.1, 0.2])
    n_bins = rcfg.get("n_bins", 10)
    out, figdir = Path("reports"), Path("reports/figures")
    figdir.mkdir(parents=True, exist_ok=True)

    con = connect(cfg)
    build_timeline(con)
    build_cohort(con, cfg)
    iso, longi = build_feature_sets(con, cfg)

    print(f"Out-of-fold predictions (5-fold x {n_repeats} repeats, grouped by patient)...")
    preds = {
        "isolated": average_oof(oof_predictions(iso, cfg, n_repeats=n_repeats), iso),
        "longitudinal": average_oof(oof_predictions(longi, cfg, n_repeats=n_repeats), longi),
    }

    # R1 calibration ---------------------------------------------------------
    cal_rows, curves = [], {}
    for fs, d in preds.items():
        for m in MODELS:
            dm = d[d.model == m]
            y, p, g = dm["label"].values, dm["p"].values, dm["subject_id"].values
            p_rec = crossfit_platt(y, p, g)
            for stage, pp in [("raw", p), ("recalibrated", p_rec)]:
                cal_rows.append({"features": fs, "model": m, "stage": stage,
                                 "auprc": average_precision_score(y, pp), "auroc": auroc_safe(y, pp),
                                 **calibration_metrics(y, pp, n_bins)})
                if m == "logreg":
                    curves[(fs, stage)] = reliability_curve(y, pp, n_bins)
    cal = pd.DataFrame(cal_rows)

    # R2 patient-level bootstrap of the H1 difference ------------------------
    boot_rows, deltas = [], {}
    for m in MODELS:
        a = preds["isolated"].query("model == @m").set_index("hadm_id")
        b = preds["longitudinal"].query("model == @m").set_index("hadm_id").loc[a.index]
        for name, fn in [("auprc", average_precision_score), ("auroc", auroc_safe)]:
            r = cluster_bootstrap_delta(a["label"].values, a["p"].values, b["p"].values,
                                        a["subject_id"].values, metric=fn, n_boot=n_boot,
                                        seed=cfg["random_state"])
            if name == "auprc":
                deltas[m] = r["_deltas"]
            r.pop("_deltas")
            boot_rows.append({"model": m, "metric": name, **r})
    boot = pd.DataFrame(boot_rows).rename(columns={
        "a": "isolated", "a_lo": "isolated_lo", "a_hi": "isolated_hi",
        "b": "longitudinal", "b_lo": "longitudinal_lo", "b_hi": "longitudinal_hi"})

    # R3 conformal prediction sets -------------------------------------------
    conf_rows = []
    for fs, d in preds.items():
        for m in MODELS:
            dm = d[d.model == m]
            for alpha in alphas:
                sets = mondrian_conformal_sets(dm["label"].values, dm["p"].values,
                                               dm["subject_id"].values, alpha=alpha)
                conf_rows.append({"features": fs, "model": m, "alpha": alpha,
                                  "target_coverage": 1 - alpha, **conformal_summary(sets)})
    conf = pd.DataFrame(conf_rows)

    # save -------------------------------------------------------------------
    cal.round(4).to_csv(out / "r1_calibration.csv", index=False)
    boot.round(4).to_csv(out / "r2_h1_bootstrap.csv", index=False)
    conf.round(4).to_csv(out / "r3_conformal.csv", index=False)
    figures.fig_reliability(curves, figdir / "05_r1_calibration.png")
    figures.fig_bootstrap(deltas, figdir / "06_r2_h1_bootstrap.png")
    figures.fig_conformal(conf, figdir / "07_r3_conformal.png")

    n_pat = preds["isolated"]["subject_id"].nunique()
    show_cal = ["features", "model", "stage", "brier", "brier_skill", "cal_intercept", "cal_slope",
                "ece", "mean_predicted", "observed_rate"]
    show_boot = ["model", "metric", "isolated", "longitudinal", "delta", "delta_lo", "delta_hi",
                 "share_boot_delta_gt0"]
    show_conf = ["features", "model", "alpha", "coverage", "coverage_y1", "share_low",
                 "share_uncertain", "share_high", "events_flagged_high", "ppv_high", "npv_low"]
    md = ["# MOSAIC reliability results (auto-generated)", "",
          f"Data: `{cfg['mimic_path']}` · {len(preds['isolated']) // len(MODELS)} index admissions, "
          f"{n_pat} patients · out-of-fold scores averaged over {n_repeats} repeats of 5-fold "
          f"patient-grouped CV · {n_boot} patient-level bootstrap resamples", "",
          "## R1 – calibration", cal[show_cal].round(3).to_markdown(index=False), "",
          "## R2 – H1 paired difference, patient-level bootstrap 95% CI",
          boot[show_boot].round(3).to_markdown(index=False), "",
          "## R3 – conformal prediction sets (class-conditional)",
          conf[show_conf].round(3).to_markdown(index=False)]
    (out / "reliability_summary.md").write_text("\n".join(md))
    print("\n".join(md))
    return cal, boot, conf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    main(a.config, a.n_boot, a.repeats)
