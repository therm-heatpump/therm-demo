"""
therm demo — Streamlit Community Cloud entry point.

Loads bundled sample data (shifted to end yesterday) and runs therm
with data sources pre-configured in demo mode.
"""
from __future__ import annotations

import io
import json
import os
import runpy
import sys
import tempfile
from datetime import date
from pathlib import Path

# Locate root directory and application code:
# - In therm-demo: streamlit_app.py is in the repo root next to app.py
# - In therm-dev-demo: streamlit_app.py is in demo/ and app.py is in the repo root
# - In therm-public (if used): app.py is in therm/app/
FILE_DIR = Path(__file__).resolve().parent
if (FILE_DIR / "app.py").exists():
    ROOT = FILE_DIR
    APP_DIR = FILE_DIR
elif (FILE_DIR.parent / "app.py").exists():
    ROOT = FILE_DIR.parent
    APP_DIR = ROOT
elif (FILE_DIR.parent / "therm" / "app" / "app.py").exists():
    ROOT = FILE_DIR.parent
    APP_DIR = ROOT / "therm" / "app"
else:
    ROOT = FILE_DIR
    APP_DIR = ROOT

os.chdir(APP_DIR)
for p in (APP_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Use ephemeral directories for demo data, profiles and secrets
DEMO_DIR = Path(tempfile.gettempdir()) / "therm-demo"
os.environ["THERM_DATA_DIR"] = str(DEMO_DIR / "data")
os.environ["THERM_SECRET_DIR"] = str(DEMO_DIR / "secret")
os.environ["THERM_DEMO"] = "1"

import pandas as pd
import streamlit as st

import data_loader
import profile_store
import source_ui
from sources.influx import InfluxSettings

st.secrets = {}  # never read external credentials


class _Upload(io.BytesIO):
    """Minimal file-like wrapper with name and size for data_loader."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name
        self.size = len(data)


def _find_sample_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "sample_data",
        ROOT / "sample_data",
        APP_DIR / "sample_data",
    ]
    for c in candidates:
        if c.exists() and (c / "grafana_numeric_longterm.csv").exists():
            return c
    raise FileNotFoundError("Could not find sample_data directory.")


@st.cache_resource(show_spinner="Loading demo heat pump data...")
def _load_demo_frames():
    """The sample, shifted to end yesterday (cached across sessions)."""
    sample_dir = _find_sample_dir()
    num_bytes = (sample_dir / "grafana_numeric_longterm.csv").read_bytes()
    state_bytes = (sample_dir / "grafana_state_longterm.csv").read_bytes()

    numeric, state = data_loader._read_grafana_csvs([
        _Upload(num_bytes, "grafana_numeric_longterm.csv"),
        _Upload(state_bytes, "grafana_state_longterm.csv"),
    ])
    last = max(f["Time"].max() for f in numeric)
    shift = pd.Timestamp(date.today()).normalize() - pd.Timestamp(last).normalize() - pd.Timedelta(days=1)
    for f in numeric + state:
        f["Time"] = f["Time"] + shift

    entities = sorted(set(pd.concat(numeric + state)["entity_id"]))
    meta = [
        {
            "entity_id": e,
            "domain": "binary_sensor" if "pump" in e else "sensor",
            "unit": "°C" if "temp" in e else None,
        }
        for e in entities
    ]
    return numeric, state, entities, meta


numeric, state, entities, meta = _load_demo_frames()

source_ui.running_in_addon = lambda: True
source_ui.available_sources = lambda: [source_ui.SOURCE_INFLUX]
source_ui.influx_settings = lambda: InfluxSettings(host="demo.invalid")
source_ui.ha_settings = lambda: None
source_ui.available_entities = lambda: entities
source_ui.ha_available_entities = lambda: entities
source_ui.setup_entities = lambda source: entities
source_ui.setup_entity_metadata = lambda source: meta
source_ui._fetch_influx_cached = lambda *a, **k: (numeric, state)
source_ui.home_assistant_defaults = lambda: {"time_zone": "Europe/Dublin", "currency": "EUR"}
# Behave as if "Load latest data" had been pressed, so the dashboard shows data straight away.
source_ui._loaded = lambda *a, **k: 1

sample_dir = _find_sample_dir()
profile_path = sample_dir / "therm_profile_Samsung_-_Sample_Profile.json"
if profile_path.exists() and not profile_store.list_profiles():
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_store.save_profile(profile)
    profile_store.update_state(active_profile=profile["profile_name"], data_source=source_ui.SOURCE_INFLUX)

runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
