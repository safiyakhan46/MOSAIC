"""Generate a tiny FAKE dataset with MIMIC-IV column names, for smoke-testing.

Not real patients. Usage: python tests/make_fake_mimic.py data/fake_mimic
Then set mimic_path: data/fake_mimic in a config and run the pipeline.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/fake_mimic")
(out / "hosp").mkdir(parents=True, exist_ok=True)
(out / "icu").mkdir(parents=True, exist_ok=True)

LABS = [50912, 51222, 51301, 50983, 50971, 50931, 50813]
pats, adms, dxs, labs, rxs, trs, icus = [], [], [], [], [], [], []
hadm = 20000000
for sid in range(10000001, 10000401):
    frailty = rng.random()
    pats.append(dict(subject_id=sid, gender=rng.choice(["M", "F"]),
                     anchor_age=int(rng.integers(20, 90)), anchor_year=2150, dod=None))
    t = pd.Timestamp("2150-01-01") + pd.Timedelta(days=int(rng.integers(0, 300)))
    for k in range(int(rng.integers(1, 6))):
        hadm += 1
        los = float(rng.gamma(2, 2))
        dis = t + pd.Timedelta(days=los)
        icu = rng.random() < 0.1 + 0.3 * frailty * k / 3
        died = rng.random() < 0.02 + 0.05 * frailty
        adms.append(dict(subject_id=sid, hadm_id=hadm, admittime=t, dischtime=dis,
                         deathtime=dis if died else None,
                         admission_type=rng.choice(["EW EMER.", "ELECTIVE", "URGENT"]),
                         discharge_location="HOME", hospital_expire_flag=int(died)))
        for _ in range(int(rng.integers(2, 10))):
            dxs.append(dict(subject_id=sid, hadm_id=hadm, seq_num=1,
                            icd_code=f"I{rng.integers(10, 99)}", icd_version=10))
        for _ in range(20):
            iid = int(rng.choice(LABS))
            labs.append(dict(subject_id=sid, hadm_id=hadm, itemid=iid,
                             charttime=t + pd.Timedelta(hours=float(rng.uniform(0, los * 24))),
                             valuenum=float(rng.normal(1 + frailty * k, 0.3))))
        rxs.append(dict(subject_id=sid, hadm_id=hadm, starttime=t, drug="Drug" + str(rng.integers(1, 20))))
        trs.append(dict(subject_id=sid, hadm_id=hadm, eventtype="admit", careunit="Medicine", intime=t, outtime=dis))
        if icu:
            icus.append(dict(subject_id=sid, hadm_id=hadm, stay_id=hadm + 10**8,
                             first_careunit="MICU", intime=t, outtime=dis, los=los))
        if died:
            break
        t = dis + pd.Timedelta(days=int(rng.integers(20, 400)))

for name, rows in [("hosp/patients", pats), ("hosp/admissions", adms),
                   ("hosp/diagnoses_icd", dxs), ("hosp/labevents", labs),
                   ("hosp/prescriptions", rxs), ("hosp/transfers", trs),
                   ("icu/icustays", icus)]:
    pd.DataFrame(rows).to_csv(out / f"{name}.csv.gz", index=False)
print(f"Fake MIMIC-style data written to {out}")
