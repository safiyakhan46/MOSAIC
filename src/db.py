"""Load MIMIC-IV CSV tables into an in-memory DuckDB connection as views."""
from pathlib import Path

import duckdb
import yaml

TABLES = {
    "patients": "hosp/patients",
    "admissions": "hosp/admissions",
    "diagnoses_icd": "hosp/diagnoses_icd",
    "labevents": "hosp/labevents",
    "prescriptions": "hosp/prescriptions",
    "transfers": "hosp/transfers",
    "icustays": "icu/icustays",
}


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _find(root: Path, stem: str) -> Path:
    for ext in (".csv.gz", ".csv"):
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {stem}.csv(.gz) under {root}")


def connect(cfg: dict) -> duckdb.DuckDBPyConnection:
    """Register each MIMIC table as a DuckDB view (data stays on disk)."""
    root = Path(cfg["mimic_path"])
    con = duckdb.connect()
    for name, stem in TABLES.items():
        path = _find(root, stem).as_posix()
        con.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{path}', header=true)"
        )
    return con
