"""Shared fixtures: every test runs on FAKE MIMIC-style data, never real patients."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fake_data(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("fake_mimic")
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_fake_mimic.py"), str(out)],
                   check=True, capture_output=True)
    return out


@pytest.fixture(scope="session")
def cfg(fake_data) -> dict:
    with open(ROOT / "config.yaml") as f:
        c = yaml.safe_load(f)
    c["mimic_path"] = str(fake_data)
    return c
