"""Integration tests on fake data: cohort rules, no future leakage, reliability run end to end."""
import shutil

import numpy as np
import pandas as pd
import pytest

from src.db import connect
from src.features import build_feature_sets
from src.labels import build_cohort
from src.timeline import build_timeline


def _features(cfg):
    con = connect(cfg)
    build_timeline(con)
    build_cohort(con, cfg)
    iso, longi = build_feature_sets(con, cfg)
    return con, iso, longi


def test_cohort_rules(cfg):
    con, _, _ = _features(cfg)
    c = con.execute("SELECT * FROM cohort").df()
    assert (c["age"] >= cfg["min_age"]).all()
    assert (c["hospital_expire_flag"] == 0).all()
    assert set(c["label"].unique()) <= {0, 1}
    assert c["hadm_id"].is_unique


def test_no_leakage_from_future_events(cfg, fake_data, tmp_path):
    """Add a new admission, ICU stay and extreme labs AFTER each patient's last discharge.
    Labels may change; features of the original index admissions must not change at all."""
    _, iso0, long0 = _features(cfg)

    fut = tmp_path / "future"
    shutil.copytree(fake_data, fut)
    adm = pd.read_csv(fut / "hosp/admissions.csv.gz", parse_dates=["admittime", "dischtime"])
    last = adm.sort_values("dischtime").groupby("subject_id").tail(1)
    new = last.copy()
    new["hadm_id"] = np.arange(90_000_000, 90_000_000 + len(new))
    new["admittime"] = last["dischtime"].values + pd.Timedelta(days=30)
    new["dischtime"] = new["admittime"] + pd.Timedelta(days=5)
    new["hospital_expire_flag"] = 0
    pd.concat([adm, new]).to_csv(fut / "hosp/admissions.csv.gz", index=False)

    icu = pd.read_csv(fut / "icu/icustays.csv.gz")
    icu_new = pd.DataFrame({"subject_id": new["subject_id"], "hadm_id": new["hadm_id"],
                            "stay_id": new["hadm_id"] + 10**8, "first_careunit": "MICU",
                            "intime": new["admittime"], "outtime": new["dischtime"], "los": 5.0})
    pd.concat([icu, icu_new]).to_csv(fut / "icu/icustays.csv.gz", index=False)

    labs = pd.read_csv(fut / "hosp/labevents.csv.gz")
    lab_new = pd.DataFrame({"subject_id": new["subject_id"], "hadm_id": new["hadm_id"],
                            "itemid": 50912, "charttime": new["admittime"] + pd.Timedelta(hours=2),
                            "valuenum": 999.0})
    pd.concat([labs, lab_new]).to_csv(fut / "hosp/labevents.csv.gz", index=False)

    _, iso1, long1 = _features({**cfg, "mimic_path": str(fut)})
    for before, after in [(iso0, iso1), (long0, long1)]:
        after = after[after["hadm_id"].isin(before["hadm_id"])]
        a = before.drop(columns="label").set_index("hadm_id").sort_index()
        b = after.drop(columns="label").set_index("hadm_id").sort_index()[a.columns]
        pd.testing.assert_frame_equal(a, b, check_dtype=False)

    # sanity: the injected future ICU stays DID reach some labels
    merged = long0[["hadm_id", "label"]].merge(long1[["hadm_id", "label"]], on="hadm_id")
    assert (merged["label_y"] >= merged["label_x"]).all()
    assert (merged["label_y"] > merged["label_x"]).any()


@pytest.mark.slow
def test_reliability_runs_end_to_end(cfg, tmp_path, monkeypatch):
    import yaml
    import run_reliability

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    cal, boot, conf = run_reliability.main(str(cfg_path), n_boot=100, n_repeats=1)

    assert len(cal) == 8 and len(boot) == 4 and len(conf) == 8
    rec = cal[cal.stage == "recalibrated"]
    assert (rec["cal_intercept"].abs() < 0.1).all()
    assert ((conf["coverage"] - conf["target_coverage"]).abs() < 0.05).all()
    for f in ["r1_calibration.csv", "r2_h1_bootstrap.csv", "r3_conformal.csv",
              "reliability_summary.md", "figures/05_r1_calibration.png"]:
        assert (tmp_path / "reports" / f).exists()
