"""Cohort and outcome.

Index event = discharge alive from an adult admission.
Label = 1 if ANY later admission starting within `horizon_days` of that discharge
        involves an ICU stay or in-hospital death.
"""
import duckdb


def build_cohort(con: duckdb.DuckDBPyConnection, cfg: dict) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE adm AS
        SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime,
               a.admission_type, a.hospital_expire_flag,
               p.gender,
               p.anchor_age + (year(a.admittime) - p.anchor_year) AS age,
               EXISTS (SELECT 1 FROM icustays i WHERE i.hadm_id = a.hadm_id) AS had_icu,
               ROW_NUMBER() OVER (PARTITION BY a.subject_id ORDER BY a.admittime) AS adm_seq
        FROM admissions a JOIN patients p USING (subject_id);

        CREATE OR REPLACE TABLE cohort AS
        SELECT k.*,
               MAX(CASE WHEN n.hadm_id IS NOT NULL
                         AND (n.had_icu OR n.hospital_expire_flag = 1)
                        THEN 1 ELSE 0 END) AS label
        FROM adm k
        LEFT JOIN adm n
          ON n.subject_id = k.subject_id
         AND n.admittime > k.dischtime
         AND n.admittime <= k.dischtime + INTERVAL {int(cfg['horizon_days'])} DAY
        WHERE k.age >= {int(cfg['min_age'])}
          AND k.hospital_expire_flag = 0
        GROUP BY ALL;
        """
    )
