"""Two feature sets for the same cohort rows.

isolated      -> uses ONLY the index admission k        (baseline, H1 comparator)
longitudinal  -> isolated + everything up to discharge of k across ALL prior admissions

No information after the index discharge is ever used (prevents leakage).
"""
import duckdb
import pandas as pd


def isolated_features(con: duckdb.DuckDBPyConnection, cfg: dict) -> pd.DataFrame:
    # last recorded value of each key lab during admission k
    lab_sql = ",\n".join(
        f"arg_max(CASE WHEN l.itemid = {iid} THEN l.valuenum END, "
        f"CASE WHEN l.itemid = {iid} THEN l.charttime END) AS last_{name}"
        for name, iid in cfg["labs"].items()
    )
    return con.execute(
        f"""
        WITH lab AS (
            SELECT c.hadm_id, {lab_sql}, COUNT(l.itemid) AS n_labs
            FROM cohort c LEFT JOIN labevents l ON l.hadm_id = c.hadm_id
            GROUP BY c.hadm_id
        ),
        dx AS (
            SELECT hadm_id, COUNT(DISTINCT icd_code) AS n_dx
            FROM diagnoses_icd GROUP BY hadm_id
        ),
        rx AS (
            SELECT hadm_id, COUNT(DISTINCT drug) AS n_drugs
            FROM prescriptions GROUP BY hadm_id
        )
        SELECT c.subject_id, c.hadm_id, c.label,
               c.age, c.gender, c.admission_type,
               date_diff('hour', c.admittime, c.dischtime) / 24.0 AS los_days,
               c.had_icu::INT AS icu_this_stay,
               COALESCE(dx.n_dx, 0) AS n_dx,
               COALESCE(rx.n_drugs, 0) AS n_drugs,
               lab.* EXCLUDE (hadm_id)
        FROM cohort c
        LEFT JOIN lab USING (hadm_id)
        LEFT JOIN dx USING (hadm_id)
        LEFT JOIN rx USING (hadm_id)
        ORDER BY c.subject_id, c.admittime
        """
    ).df()


def history_features(con: duckdb.DuckDBPyConnection, cfg: dict) -> pd.DataFrame:
    """Cross-visit features: counts, gaps, prior ICU, lab trends (H3)."""
    trend_sql = ",\n".join(
        f"regr_slope(CASE WHEN l.itemid = {iid} THEN l.valuenum END, "
        f"CASE WHEN l.itemid = {iid} THEN epoch(l.charttime) / 86400.0 END) AS slope_{name},\n"
        f"AVG(CASE WHEN l.itemid = {iid} THEN l.valuenum END) AS mean_{name}"
        for name, iid in cfg["labs"].items()
    )
    return con.execute(
        f"""
        WITH prior AS (
            SELECT c.hadm_id,
                   COUNT(p.hadm_id) AS n_prior_adm,
                   SUM(p.had_icu::INT) AS n_prior_icu,
                   SUM(CASE WHEN p.admission_type ILIKE '%EMER%' THEN 1 ELSE 0 END) AS n_prior_emergency,
                   SUM(date_diff('hour', p.admittime, p.dischtime)) / 24.0 AS prior_los_days,
                   date_diff('day', MAX(p.dischtime), c.admittime) AS days_since_last_disch
            FROM cohort c
            LEFT JOIN adm p ON p.subject_id = c.subject_id AND p.dischtime <= c.admittime
            GROUP BY c.hadm_id, c.admittime
        ),
        hist_dx AS (
            SELECT c.hadm_id, COUNT(DISTINCT d.icd_code) AS n_dx_cumulative
            FROM cohort c
            JOIN adm a ON a.subject_id = c.subject_id AND a.dischtime <= c.dischtime
            JOIN diagnoses_icd d ON d.hadm_id = a.hadm_id
            GROUP BY c.hadm_id
        ),
        hist_lab AS (
            SELECT c.hadm_id, {trend_sql}
            FROM cohort c
            JOIN labevents l ON l.subject_id = c.subject_id AND l.charttime <= c.dischtime
            GROUP BY c.hadm_id
        )
        SELECT c.hadm_id,
               COALESCE(prior.n_prior_adm, 0) AS n_prior_adm,
               COALESCE(prior.n_prior_icu, 0) AS n_prior_icu,
               COALESCE(prior.n_prior_emergency, 0) AS n_prior_emergency,
               COALESCE(prior.prior_los_days, 0) AS prior_los_days,
               prior.days_since_last_disch,
               COALESCE(hist_dx.n_dx_cumulative, 0) AS n_dx_cumulative,
               hist_lab.* EXCLUDE (hadm_id)
        FROM cohort c
        LEFT JOIN prior USING (hadm_id)
        LEFT JOIN hist_dx USING (hadm_id)
        LEFT JOIN hist_lab USING (hadm_id)
        """
    ).df()


def build_feature_sets(con, cfg):
    iso = isolated_features(con, cfg)
    hist = history_features(con, cfg)
    longi = iso.merge(hist, on="hadm_id", how="left")
    return iso, longi
