"""MOSAIC end-to-end: tables -> timeline -> cohort/labels -> features -> models.

Usage:  python run_pipeline.py                 (uses config.yaml)
        python run_pipeline.py --config other.yaml
"""
import argparse
from pathlib import Path

from src.db import connect, load_config
from src.features import build_feature_sets
from src.labels import build_cohort
from src.timeline import build_timeline, summary
from src.train import compare


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    con = connect(cfg)

    build_timeline(con)
    print("\n== Timeline ==\n", summary(con).to_string(index=False))

    build_cohort(con, cfg)
    c = con.execute(
        "SELECT COUNT(*) n_index, COUNT(DISTINCT subject_id) n_patients, "
        "AVG(label) pos_rate FROM cohort").df()
    print("\n== Cohort ==\n", c.to_string(index=False))

    iso, longi = build_feature_sets(con, cfg)
    print(f"\nFeatures: isolated={iso.shape[1] - 3}, longitudinal={longi.shape[1] - 3}")

    results = compare(iso, longi, cfg)
    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/results.csv", index=False)
    print("\n== Results (test patients) ==\n",
          results.round(3).to_string(index=False) if len(results) else "no results")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
