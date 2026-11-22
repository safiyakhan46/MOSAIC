"""MOSAIC full analysis: H1 (repeated grouped CV), H2 subgroups, H3 SHAP, figures.

Usage: python run_analysis.py [--config config.yaml]
Outputs (aggregate only): reports/*.csv, reports/figures/*.png, reports/summary.md
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src import figures
from src.db import connect, load_config
from src.evaluate import h1_compare, h2_subgroups, h3_shap, oof_predictions
from src.features import build_feature_sets
from src.labels import build_cohort
from src.timeline import build_timeline


def main(config_path):
    cfg = load_config(config_path)
    out, figdir = Path("reports"), Path("reports/figures")
    figdir.mkdir(parents=True, exist_ok=True)

    con = connect(cfg)
    build_timeline(con)
    build_cohort(con, cfg)
    iso, longi = build_feature_sets(con, cfg)
    cohort = con.execute("SELECT COUNT(*) n_index, COUNT(DISTINCT subject_id) n_patients, "
                         "SUM(label) n_events, ROUND(AVG(label),3) pos_rate FROM cohort").df()

    print("Running 5x5 grouped CV (isolated, longitudinal)...")
    oof_iso, oof_long = oof_predictions(iso, cfg), oof_predictions(longi, cfg)
    summary, paired = h1_compare(oof_iso, oof_long)
    sub = h2_subgroups(oof_iso, oof_long, longi)
    imp = h3_shap(longi, cfg)

    summary.to_csv(out / "h1_summary.csv")
    paired.to_csv(out / "h1_paired_difference.csv")
    sub.to_csv(out / "h2_subgroups.csv", index=False)
    imp.to_csv(out / "h3_shap_top.csv", index=False)

    figures.fig_cohort(con, figdir / "01_admissions_per_patient.png")
    figures.fig_h1(summary, figdir / "02_h1_isolated_vs_longitudinal.png")
    figures.fig_h2(sub, figdir / "03_h2_subgroups.png")
    figures.fig_h3(imp, figdir / "04_h3_shap.png")

    md = ["# MOSAIC results (auto-generated)", "",
          f"Data: `{cfg['mimic_path']}`", "", "## Cohort", cohort.to_markdown(index=False), "",
          "## H1 – performance (mean, SD over 5 repeats of 5-fold patient-grouped CV)",
          summary.to_markdown(), "", "## H1 – paired difference (longitudinal − isolated)",
          paired.to_markdown(), "", "## H2 – AUPRC by prior admissions",
          sub.round(3).to_markdown(index=False), "", "## H3 – top SHAP features",
          imp.round(4).to_markdown(index=False)]
    (out / "summary.md").write_text("\n".join(md))
    (out / "cohort.json").write_text(json.dumps(cohort.iloc[0].to_dict(), default=float))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
