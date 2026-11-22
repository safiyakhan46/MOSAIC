"""Build the MOSAIC event timeline: one ordered row per clinical event per patient.

Columns: subject_id, hadm_id, event_time, event_type, code, value
This is the "mosaic" — fragments from every admission stitched into one table.
"""
import duckdb
import pandas as pd

TIMELINE_SQL = """
CREATE OR REPLACE TABLE timeline AS
SELECT subject_id, hadm_id, admittime AS event_time, 'admission' AS event_type,
       admission_type AS code, NULL::DOUBLE AS value
FROM admissions
UNION ALL
SELECT subject_id, hadm_id, dischtime, 'discharge', discharge_location, NULL
FROM admissions
UNION ALL
-- diagnoses carry no timestamp in MIMIC-IV; anchor them at discharge
SELECT d.subject_id, d.hadm_id, a.dischtime, 'diagnosis',
       d.icd_version::VARCHAR || ':' || d.icd_code, NULL
FROM diagnoses_icd d JOIN admissions a USING (hadm_id)
UNION ALL
SELECT subject_id, hadm_id, charttime, 'lab', itemid::VARCHAR, valuenum
FROM labevents
WHERE hadm_id IS NOT NULL
UNION ALL
SELECT subject_id, hadm_id, starttime, 'medication', drug, NULL
FROM prescriptions
UNION ALL
SELECT subject_id, hadm_id, intime, 'icu_admit', first_careunit, los
FROM icustays
"""


def build_timeline(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(TIMELINE_SQL)


def patient_timeline(con: duckdb.DuckDBPyConnection, subject_id: int) -> pd.DataFrame:
    """Return one patient's full ordered history (for exploration and the demo)."""
    return con.execute(
        "SELECT * FROM timeline WHERE subject_id = ? ORDER BY event_time, event_type",
        [subject_id],
    ).df()


def summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT event_type, COUNT(*) AS n_events,
               COUNT(DISTINCT subject_id) AS n_patients
        FROM timeline GROUP BY event_type ORDER BY n_events DESC
        """
    ).df()
