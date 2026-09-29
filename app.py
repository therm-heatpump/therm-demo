# app.py
import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import traceback  # NEW: for detailed debug
import hashlib
from pathlib import Path
from typing import Optional

import view_trends
import view_runs
import view_quality
import mapping_ui
import inspector
import data_loader
import ha_loader
import processing
import baselines as heartbeat_baselines
import heartbeat_store
import debug_exports
import engine_context
import pipeline
import chunked
import source_ui
import profile_store
import processed_cache
import memory_guard
import config_manager
from config import cop_provenance

_is_demo = bool(os.environ.get("THERM_DEMO"))


def _ui_message(level: str, msg: str) -> None:
    """Route engine / loader messages to Streamlit UI."""
    if level == "error":
        st.error(msg)
    elif level == "warning":
        st.warning(msg)
    else:
        st.info(msg)

# ----------------------------------------------------------------------
# Lightweight console logger with timestamps (UTC)
# ----------------------------------------------------------------------
def _log(msg: str) -> None:
    """Console logging (disabled by default)."""
    return


# --- FIX: Console Error Suppression ---
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
pd.set_option("future.no_silent_downcasting", True)

# --- UI helpers ---------------------------------------------------------------
def _run_parent_script(script: str, key: str) -> None:
    """
    Run a trusted, static script that acts on the app page (Streamlit does not run
    <script> in st.markdown). st.iframe replaces the components html function, which
    Streamlit deprecated for removal after 2026-06-01; it needs a height of at least
    1 px, so the keyed container is taken out of the layout by CSS (.st-key-therm-js-*).
    Never pass user-supplied content here: the iframe has same-origin access to the app.
    """
    with st.container(key=key):
        st.iframe(f"<script>{script}</script>", height=1)


def _scroll_to_top_if_requested() -> None:
    """
    If a previous step requested a scroll-to-top (e.g. after 'Process Uploaded Data'),
    inject a small JS snippet once and then clear the flag.

    This targets the main Streamlit app container in the parent document,
    which is the element the user actually scrolls.
    """
    if st.session_state.get("scroll_to_top"):
        _run_parent_script(
            """try {
              const app = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
              if (app && app.scrollTo) { app.scrollTo({top: 0}); } else { window.parent.scrollTo(0, 0); }
            } catch (e) { /* cross-origin or selector change: leave the scroll position */ }""",
            key="therm-js-scroll-request",
        )
        st.session_state["scroll_to_top"] = False
# --- end scroll helper --------------------------------------------------------


# No ⋮ menu in Streamlit's top strip: its items (rerun, clear cache, settings, print, record) are
# developer tools, and the strip then only holds the sidebar's reopen button (see the CSS below).
st.set_option("client.toolbarMode", "minimal")
st.set_page_config(page_title="therm", layout="wide", page_icon="assets/therm_logo_browser_tab.png")
# Reduce top padding in main and sidebar to pull content up
st.markdown(
    """
    <style>
    /* Streamlit's top strip (3.75rem) holds only the sidebar's reopen button once the ⋮ menu is off
       (toolbarMode). While the sidebar is open the strip is empty: make it see-through and click-through
       and start the page near the top. While it is collapsed (phones, or closed by hand) the strip keeps
       its background and the page starts below it, so the reopen button never covers content. */
    .block-container { padding-top: 3.5rem !important; }
    [data-testid="stApp"]:has([data-testid="stSidebar"][aria-expanded="true"]) header[data-testid="stHeader"] {
        background: transparent !important; pointer-events: none; }
    [data-testid="stApp"]:has([data-testid="stSidebar"][aria-expanded="true"]) header[data-testid="stHeader"] * {
        pointer-events: auto; }
    @media (min-width: 768px) {
        [data-testid="stApp"]:has([data-testid="stSidebar"][aria-expanded="true"]) .block-container {
            padding-top: 1.25rem !important; }
    }
    /* Desktop: a little less empty margin either side, so the five headline cards have room. */
    @media (min-width: 992px) { .block-container { padding-left: 2.5rem !important; padding-right: 2.5rem !important; } }
    /* Headline cards: values that fit a fifth of the width, and plain grey sub-lines (not a badge). */
    [data-testid="stMetricValue"] { font-size: 1.65rem !important; }
    [data-testid="stMetricDelta"] { background: none !important; padding-left: 0 !important; }
    /* Headline cards: one height for the row, compact padding. */
    .st-key-therm-kpis [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
    .st-key-therm-kpis [data-testid="stElementContainer"],
    .st-key-therm-kpis [data-testid="stMetric"] { height: 100% !important; min-height: 6.5rem !important; }
    .st-key-therm-kpis [data-testid="stMetric"] > div { padding: 0.55rem 0.85rem !important; }
    .st-key-therm-kpis [data-testid="stMetricValue"] { padding-bottom: 0 !important; line-height: 1.25 !important; }
    /* The page's navigation (Trends | Run Inspector): flat, the open view shaded, a line under the row. */
    .st-key-therm-nav { border-bottom: 1px solid rgba(49, 51, 63, 0.12); padding-bottom: 0.6rem; margin-bottom: 0.2rem; }
    .st-key-therm-nav [role="radiogroup"] { border: none !important; gap: 0.25rem; }
    .st-key-therm-nav button[data-variant="segmented_control"] {
        border: none !important; background: none !important; border-radius: 0.5rem !important;
        color: rgba(49, 51, 63, 0.7) !important; }
    .st-key-therm-nav button[data-variant="segmented_control"][aria-checked="true"] {
        background: rgba(49, 51, 63, 0.07) !important; color: rgb(49, 51, 63) !important; font-weight: 600; }
    /* Phones: the headline cards go two to a row and the page title is smaller. (Streamlit stacks every
       column below 640px.) */
    @media (max-width: 640px) {
        .st-key-therm-kpis [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.6rem !important; }
        .st-key-therm-kpis [data-testid="stColumn"] { min-width: calc(50% - 0.3rem) !important;
                                                      flex: 1 1 calc(50% - 0.3rem) !important; }
        h1 { font-size: 1.9rem !important; }
        .st-key-therm-kpis [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
        .st-key-therm-kpis [data-testid="stElementContainer"],
        .st-key-therm-kpis [data-testid="stMetric"] { min-height: 5.55rem !important; }
        .st-key-therm-kpis [data-testid="stMetric"] > div { padding: 0.4rem 0.65rem !important; }
        .st-key-therm-kpis [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
        .st-key-therm-kpis [data-testid="stMetricLabel"] p { font-size: 0.78rem !important; }
        .st-key-therm-kpis [data-testid="stMetricDelta"] { font-size: 0.72rem !important; }
    }
    /* Script-only iframes (_run_parent_script): out of the layout, no gap. */
    [class*="st-key-therm-js-"] { position: absolute !important; width: 0 !important;
                                  height: 0 !important; overflow: hidden !important; }
    /* The source choices are stacked, so the open sidebar no longer needs the
       default wide column. Keep the collapsed state and its toggle untouched. */
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 16.5rem !important;
        max-width: 16.5rem !important;
    }
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 16.5rem !important;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 0.75rem !important; }
    /* Nudge sidebar logo upward a bit more and center it */
    [data-testid="stSidebar"] img { margin-top: -22px !important; display: block !important; margin-left: auto !important; margin-right: auto !important; max-width: 150px !important; }
    /* Move the sidebar tagline only */
    .sidebar-tagline { margin-top: -30px !important; display: block; text-align: center;
                       font-size: 0.78rem; line-height: 1.25; }
    /* Sticky wrapper for System Setup header + actions */
    .setup-sticky {
        position: sticky;
        top: 0;
        z-index: 20;
        background: inherit;
        padding-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    /* Reduce spacing after radio button in sidebar - but NOT in expanders */
    [data-testid="stSidebar"] > div > div > [data-testid="stRadio"] {
        margin-bottom: -2rem !important;
    }
    /* Restore normal spacing for radio buttons inside sidebar expanders */
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stRadio"] {
        margin-bottom: 0 !important;
    }
    /* Reduce spacing before Global Stats in sidebar */
    [data-testid="stSidebar"] .global-stats {
        margin-top: -5rem !important;
    }
    /* Reduce spacing around all h3 elements (subheaders) - but NOT in sidebar */
    .main h3 {
        margin-bottom: -1rem !important;
    }
    /* Reduce spacing around Plotly charts - only in main content */
    .main [data-testid="stPlotlyChart"] {
        margin-top: -1rem !important;
        margin-bottom: -1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Auto-load the active saved profile once per session, so a returning user goes
# straight to analysis ($THERM_DATA_DIR/profiles; see profile_store).
if "system_config" not in st.session_state and not st.session_state.get("profile_autoload_done"):
    st.session_state["profile_autoload_done"] = True
    _active_profile = profile_store.load_active_profile()
    if _active_profile and _active_profile.get("mapping"):
        st.session_state["system_config"] = _active_profile


def _save_active_profile(config: dict) -> None:
    """Persist a confirmed Setup as the active profile (same content as the JSON download)."""
    export = config_manager.export_config_for_sharing(config)
    export["rooms_per_zone"] = config.get("rooms_per_zone", {})
    export["config_history"] = config.get("config_history", [])
    try:
        profile_store.save_profile(export)
        profile_store.update_state(active_profile=export["profile_name"])
    except OSError as e:
        st.warning(f"Profile could not be saved to therm's storage: {e}")


def _dataset_source_id(source_choice: str) -> str:
    """Translate a UI source label to the identifier stored with processed results."""
    if source_choice == source_ui.SOURCE_HA:
        return "ha_api"
    if source_choice == source_ui.SOURCE_INFLUX:
        return "influx"
    return "grafana"


_SETUP_DRAFT_KEYS = {
    "autodetect_auto_done",
    "autodetect_result",
    "autodetect_replace",
    "config_hist_error",
    "config_history",
    "history_signature",
    "loaded_profile_signature",
    "new_change_date",
    "new_change_note",
    "new_change_tag",
    "new_change_time",
    "profile_name_input",
    "profile_switched_to",
    "saved_profile_pick",
    "setup_new_profile",
    "tariff_add_change",
    "tariff_editor",
    "tariff_new_from",
    "tariff_rows_df",
    "tariff_rows_signature",
}


def _discard_setup_draft() -> None:
    """Remove every unsaved Setup value so reopening starts from the saved profile."""
    import setup_sections

    setup_sections.reset_widgets()
    prefixes = ("onb_",)
    for key in list(st.session_state):
        if key in _SETUP_DRAFT_KEYS or key.startswith(prefixes):
            st.session_state.pop(key, None)
    for key in (setup_sections.DRAFT_KEY, setup_sections.DRAFT_BASE_KEY, "setup_tab"):
        st.session_state.pop(key, None)


def _open_data_quality() -> None:
    """Sidebar button: show Data Quality, with neither view tab selected."""
    st.session_state["last_view"] = "Data Quality"
    st.session_state["view_mode"] = None


_HEARTBEAT_SESSION_KEYS = ("heartbeat_baseline", "heartbeat_baseline_source", "heartbeat_baseline_meta",
                           "heartbeat_upload_identity", "heartbeat_baseline_path", "heartbeat_profile")


def _drop_session_heartbeat() -> None:
    """Forget the session's active heartbeat (it belongs to one profile)."""
    for key in _HEARTBEAT_SESSION_KEYS:
        st.session_state.pop(key, None)


# Decide whether we're in System Setup (no processed config yet)
_log("app_run_start")
force_system_setup = st.session_state.get("force_system_setup", False)
in_system_setup = force_system_setup or "system_config" not in st.session_state

# Apply any one-shot scroll request
_scroll_to_top_if_requested()

# Versioned key to allow hard-reset of the file uploader widget
if "csv_uploader_version" not in st.session_state:
    st.session_state["csv_uploader_version"] = 0
if "heartbeat_uploader_version" not in st.session_state:
    st.session_state["heartbeat_uploader_version"] = 0


# === SIDEBAR HEADER ===
col1, col2, col3 = st.sidebar.columns([0.9, 2.2, 0.9])
with col2:
    st.image("assets/therm_logo.png")
st.sidebar.markdown(
    "<div class='sidebar-tagline'><strong>Thermal Health & Efficiency Reporting Module (Beta)</strong></div>",
    unsafe_allow_html=True,
)

# Top of the main panel: only the view toggle, so the first chart fits on screen.
# Period, Load, Setup and Data Quality live in the sidebar.
if _is_demo:
    st.info(
        "💡 **Interactive Demo:** Exploring sample heat pump data. "
        "Hover over charts, switch between **Trends** and **Run Inspector**, or filter by Space Heating vs Hot Water."
    )
main_nav = st.container(key="therm-nav")  # styled as the page's navigation (CSS above)

_direct_only = source_ui.running_in_addon() or _is_demo
# Add-on: the source choice sits openly at the top of the sidebar (no uploads to hide).
_source_box = (st.sidebar.container() if _direct_only else
               st.sidebar.expander("Data source & files", expanded=in_system_setup))
with _source_box:
    if _is_demo:
        st.markdown("**Data source:** Sample Data (Demo)")
        source_choice = source_ui.SOURCE_INFLUX
    else:
        # In the HA add-on: Home Assistant or InfluxDB, read directly (no uploads, which
        # HA's Ingress caps at 16 MB). Standalone: CSV exports or InfluxDB (+ HA when
        # THERM_HA_URL/TOKEN are set). The last choice is remembered.
        _source_options = source_ui.available_sources()
        _saved_source = profile_store.get_state().get("data_source")
        if _saved_source in _source_options:
            _default_source = _saved_source
        elif source_ui.running_in_addon():
            _default_source = source_ui.SOURCE_HA
        elif source_ui.influx_settings() is not None:
            _default_source = source_ui.SOURCE_INFLUX
        else:
            _default_source = _source_options[0]
        source_choice = st.radio(
            "Data source",
            _source_options,
            index=_source_options.index(_default_source),
            horizontal=False,
            key="data_source_choice",
        )
        if source_choice != _saved_source:
            try:
                profile_store.update_state(data_source=source_choice)
            except OSError:
                pass
    is_direct_source = True if _is_demo else (source_choice != source_ui.SOURCE_CSV)
    source_payload = None

    if is_direct_source:
        uploaded_files = []
        show_inspector = False
        if in_system_setup:
            direct_entities = source_ui.setup_entities(source_choice)
            if direct_entities:
                # Feed the Setup dropdowns; the key stops mapping_ui re-scanning files.
                st.session_state["available_sensors"] = direct_entities
                st.session_state["available_sensors_files_key"] = [(source_choice, len(direct_entities))]
                # Units / device classes / integrations for "Detect my heat pump" in Setup.
                st.session_state["entity_metadata"] = source_ui.setup_entity_metadata(source_choice)
                st.caption(f"{len(direct_entities):,} sensors available in {source_choice}.")
    else:
        _log("sidebar_upload_start")
        # Use a versioned key so we can hard-reset the uploader
        uploaded_files = st.file_uploader(
            "Upload CSV(s)",
            accept_multiple_files=True,
            type="csv",
            key=f"csv_uploader_{st.session_state['csv_uploader_version']}",
        )
        _log(f"sidebar_upload_done files={len(uploaded_files) if uploaded_files else 0}")

        show_inspector = st.checkbox("Show File Inspector", value=False)

    # Persist the filenames so we can show them even after processing
    if uploaded_files:
        st.session_state["uploaded_filenames"] = [f.name for f in uploaded_files]

        # ------------------------------------------------------------------
        # New: auto-detect per-file source (Grafana vs Home Assistant),
        # with manual override.
        # ------------------------------------------------------------------
        file_sources = st.session_state.get("file_sources", {})
        source_detection_cache = st.session_state.get("file_source_detection_cache", {})

        st.markdown("**File source type:**")
        for idx, f in enumerate(uploaded_files):
            fname = getattr(f, "name", f"file_{idx}")

            # Reading the first CSV rows is unnecessary on every Streamlit rerun.
            # file_id changes when an upload is replaced; size/name are a safe fallback.
            detection_key = (
                str(getattr(f, "file_id", "")),
                fname,
                int(getattr(f, "size", 0) or 0),
            )
            if detection_key not in source_detection_cache:
                source_detection_cache[detection_key] = data_loader.detect_file_source(f)
            auto_detected = source_detection_cache[detection_key]
            # Prefer previously selected source, else auto-detected, else grafana
            default_source = file_sources.get(fname, auto_detected or "grafana")

            source_label = st.radio(
                label=f"Source for `{fname}`",
                options=["Grafana / Influx CSV", "Home Assistant CSV"],
                index=0 if default_source == "grafana" else 1,
                key=f"file_source_{idx}",
                help=f"Auto-detected: {auto_detected or 'unknown'}",
            )

            internal_code = "grafana" if source_label.startswith("Grafana") else "ha"
            file_sources[fname] = internal_code

        st.session_state["file_sources"] = file_sources
        st.session_state["file_source_detection_cache"] = source_detection_cache
        # ------------------------------------------------------------------

    if not is_direct_source and "uploaded_filenames" in st.session_state:
        st.markdown("**Loaded files:**")
        for name in st.session_state["uploaded_filenames"]:
            st.markdown(f"- `{name}`")

    # Standalone only: in the add-on, uploads are awkward (Ingress) and a heartbeat
    # generated in Data Quality stays active for the session.
    if not source_ui.running_in_addon():
        st.markdown("**Reusable heartbeat baseline (optional):**")
        heartbeat_upload = st.file_uploader(
            "Load a heartbeat learned from a longer analysis",
            type="json",
            key=f"heartbeat_uploader_{st.session_state['heartbeat_uploader_version']}",
            help=(
                "Load a baseline generated from a 3–12 month analysis before "
                "processing a shorter dataset."
            ),
        )
        if heartbeat_upload is not None:
            upload_bytes = heartbeat_upload.getvalue()
            upload_identity = (
                getattr(heartbeat_upload, "name", None),
                hashlib.sha256(upload_bytes).hexdigest(),
            )
            if st.session_state.get("heartbeat_upload_identity") != upload_identity:
                loaded, label, meta = heartbeat_baselines.load_heartbeat_baseline(
                    heartbeat_upload
                )
                if loaded:
                    st.session_state["heartbeat_baseline"] = loaded
                    st.session_state["heartbeat_baseline_source"] = label
                    st.session_state["heartbeat_baseline_meta"] = meta
                    st.session_state["heartbeat_upload_identity"] = upload_identity
                    # A different baseline changes reporting-pattern fusion and DQ.
                    st.session_state.pop("cached", None)
                else:
                    st.warning("The selected JSON file contains no usable heartbeat baseline.")

    active_heartbeat = st.session_state.get("heartbeat_baseline") or {}
    # Add-on: the heartbeat is automatic and shown under Data Quality → Heartbeats;
    # the sidebar keeps it only for the standalone app's manual upload workflow.
    if active_heartbeat and not source_ui.running_in_addon():
        source = st.session_state.get("heartbeat_baseline_source", "current session")
        st.success(f"Heartbeat active: {len(active_heartbeat)} sensors ({source})")
        if st.button("Clear heartbeat baseline", type="secondary"):
            owner = st.session_state.get("heartbeat_profile") or (
                (st.session_state.get("system_config") or {}).get("profile_name"))
            _drop_session_heartbeat()
            st.session_state.pop("cached", None)
            st.session_state["heartbeat_uploader_version"] += 1
            # Add-on: don't reload this profile's saved heartbeat again this session.
            st.session_state["heartbeat_auto_off"] = owner
            st.rerun()

    # Optional: allow the user to hard-reset everything (uploads only; direct sources
    # have nothing to clear)
    if not is_direct_source and st.button("Clear files and start again", type="secondary"):
        for key in [
            "system_config",
            "cached",
            "uploaded_filenames",
            "capabilities",
            "loaded_profile_signature",      # NEW: forget which profile was loaded
            "available_sensors",             # optional: force re-scan
            "available_sensors_files_key",   # optional: force re-scan
            "raw_history_df",
            "debug_download_bundles",
            "file_source_detection_cache",
        ]:
            st.session_state.pop(key, None)

        # Force the file_uploader to re-mount with a fresh key
        st.session_state["csv_uploader_version"] += 1
        st.rerun()

# Direct sources: Period and Load latest data, under the source choice.
side_load = st.sidebar.container()

# --- Sample data download (standalone only; the add-on reads HA/InfluxDB directly) ---
if in_system_setup and not source_ui.running_in_addon():
    with st.sidebar.expander("Download Sample Data", expanded=False):
        samples = [
            ("grafana_numeric_longterm.csv", "sample_data/grafana_numeric_longterm.csv", "text/csv"),
            ("grafana_state_longterm.csv", "sample_data/grafana_state_longterm.csv", "text/csv"),
            ("therm_profile_Samsung_-_Sample_Profile.json", "sample_data/therm_profile_Samsung_-_Sample_Profile.json", "application/json"),
        ]
        for label, rel_path, mime in samples:
            p = Path(rel_path)
            if p.exists():
                st.download_button(
                    f"Download {label}",
                    data=p.read_bytes(),
                    file_name=label,
                    mime=mime,
                )
            else:
                st.caption(f"{label} (file not found)")
# Month results of long analyses kept on disk (chunked.py): five years of one profile, or
# a year or two each of a few profiles/settings; about 6–14 MB per month for ~30 sensors.
# Oldest-used months go first.
MONTH_RESULTS_KEPT = 60


def _month_store() -> processed_cache.Store:
    return processed_cache.Store("processed-months", MONTH_RESULTS_KEPT)


def _payload_months(payload) -> list:
    """Calendar months of a direct-source load (its "period" is inclusive local dates)."""
    import datetime as _dt

    period = (payload or {}).get("period")
    if not period:
        return []
    start = _dt.date.fromisoformat(period[0])
    end = _dt.date.fromisoformat(period[1]) + _dt.timedelta(days=1)
    return chunked.month_windows(start, end)


def _load_processed(disk_key) -> Optional[dict]:
    """A saved whole-period result, or None if there is none or it can no longer be used: a
    month it refers to has gone from the month store (the run details would be empty), so the
    period is processed again instead of reopening it."""
    stored = processed_cache.load(disk_key)
    if stored is not None and isinstance(stored.get("df"), chunked.MonthFrames) and not stored["df"].is_complete():
        return None
    return stored


# === MANUAL CACHING LOGIC ===
def get_processed_data(files, user_config):
    """
    Manually manages the cache to prevent UI re-renders of the loading screen.
    Now also routes between Grafana/Influx and Home Assistant CSV modes.
    """
    if not files:
        return None

    import time
    def _log(msg: str) -> None:
        # Silenced console logging to reduce noise in terminal output.
        return

    # ------------------------------------------------------------------
    # 1. Decide which data source is being used for this run
    # ------------------------------------------------------------------
    # A dict is a payload from a direct source (source_ui): InfluxDB or HA history.
    payload = files if isinstance(files, dict) else None
    is_influx = bool(payload) and payload.get("kind") == "influx"
    if payload:
        dataset_source = "influx" if is_influx else "ha_api"
        # Entity names adapted to the source (sensor.x vs x); roles are unchanged.
        user_config = payload.get("config") or user_config
    else:
        # file_sources is set in the sidebar "File source type" radios
        file_sources = st.session_state.get("file_sources", {})
        sources = set()
        for f in files:
            fname = getattr(f, "name", None)
            src = file_sources.get(fname, "grafana")
            sources.add(src)

        if len(sources) > 1:
            st.error(
                "Mixed data sources detected.\n\n"
                "For a given run, please set **all files** to either "
                "`Grafana / Influx CSV` or `Home Assistant CSV`, not a mixture."
            )
            return None

        dataset_source = sources.pop() if sources else "grafana"
    # Store for downstream debug bundle
    st.session_state["dataset_source"] = dataset_source


    # ------------------------------------------------------------------
    # 2. Manual cache key (include dataset_source so HA vs Grafana
    #    runs don't collide in the cache)
    # ------------------------------------------------------------------
    # Identify uploads by Streamlit's per-upload file_id (a replaced file with
    # the same name and size gets a new id); fall back to a content hash.
    def _file_identity(f):
        fid = getattr(f, "file_id", None)
        if fid:
            return (f.name, f.size, str(fid))
        try:
            import hashlib
            return (f.name, f.size, hashlib.sha256(f.getvalue()).hexdigest())
        except Exception:
            return (f.name, f.size)

    files_key = (
        (payload["key"], payload.get("fingerprint")) if payload
        else tuple(sorted(_file_identity(f) for f in files))
    )
    # Use a stable, name-sensitive cache key. json.dumps with sort_keys ensures we
    # don't accidentally reuse a cache entry when the profile name changes.
    import json as _json

    config_key = _json.dumps(user_config, sort_keys=True, default=str)
    # Add-on: the heartbeat is saved per profile after long analyses and loaded
    # automatically here (heartbeat_store), replacing the manual upload.
    auto_heartbeat = bool(payload) and source_ui.running_in_addon()
    profile_name = user_config.get("profile_name")
    if auto_heartbeat:
        # A heartbeat belongs to one profile: drop another
        # profile's before this analysis, then load this profile's saved one
        # unless the user cleared it for this profile.
        if (st.session_state.get("heartbeat_baseline")
                and st.session_state.get("heartbeat_profile") != profile_name):
            _drop_session_heartbeat()
        # A heartbeat saved by the last analysis applies once different data is requested;
        # reruns of the result on screen keep the heartbeat it was computed with.
        shown = st.session_state.get("cached") or {}
        if (st.session_state.get("heartbeat_pending") == profile_name
                and (shown.get("key") or (None,))[0] != files_key):
            _drop_session_heartbeat()
            st.session_state.pop("heartbeat_pending", None)
        # (Still pending here means the result on screen is being shown again: loading the new
        # heartbeat now would change its key and process the same data again.)
        if (not st.session_state.get("heartbeat_baseline")
                and st.session_state.get("heartbeat_auto_off") != profile_name
                and st.session_state.get("heartbeat_pending") != profile_name):
            saved_hb, saved_meta = heartbeat_store.load(profile_name, user_config.get("mapping"))
            if saved_hb:
                st.session_state["heartbeat_baseline"] = saved_hb
                st.session_state["heartbeat_baseline_source"] = heartbeat_store.describe(saved_meta)
                st.session_state["heartbeat_baseline_meta"] = saved_meta
                st.session_state["heartbeat_profile"] = profile_name
    active_baseline = st.session_state.get("heartbeat_baseline") or {}
    baseline_key = _json.dumps(active_baseline, sort_keys=True, default=str)
    combined_key = (files_key, config_key, dataset_source, baseline_key)

    if "cached" in st.session_state:
        cached = st.session_state["cached"]
        if cached.get("key") == combined_key and cached.get("profile_name") == user_config.get("profile_name"):
            return st.session_state["cached"]

    # Direct sources: reuse results processed earlier (any session) for the same
    # data, profile, engine version and heartbeat (processed_cache).
    disk_key = None
    if payload:
        import config as _therm_config

        disk_key = processed_cache.cache_key(combined_key, getattr(_therm_config, "CALC_VERSION", ""))
        stored = _load_processed(disk_key)
        if stored is not None:
            baseline_data = stored.pop("_baseline_data", None)
            baseline_path = stored.pop("_baseline_path", None)
            stored["key"] = combined_key
            # Old disk entries predate the explicit source field; the cache key that
            # loaded them already proves which source they belong to.
            stored["source"] = stored.get("source", dataset_source)
            st.session_state["analysis_summary"] = _analysis_summary(
                stored.get("df"), payload.get("label"), stored.get("processed_at"))
            _remember_last_result(disk_key, payload, user_config, stored.get("processed_at"))
            return _install_processed(stored, baseline_data, baseline_path)

    # One heavy analysis at a time in this therm process (memory_guard.ANALYSIS_LOCK):
    # a second browser tab or user waits instead of doubling peak memory.
    if not memory_guard.ANALYSIS_LOCK.acquire(blocking=False):
        st.warning("Another analysis is running in therm (another browser tab or user). "
                   "Try again when it has finished.")
        return None

    # The progress panel lives in a slot that is removed once everything is done
    # because collapsing it above the dashboard left the page scrolled
    # past the top, and "complete" was shown before saving had finished.
    status_slot = st.empty()
    status_container = status_slot.status("Processing data…", expanded=True)

    try:
        engine_context.set_debug(
            st.session_state.get("debug_engine", False),
            st.session_state.setdefault("engine_debug_traces", []),
        )
        t_start = time.time()
        # ------------------------------------------------------------------
        # 3. Source-specific loading
        # ------------------------------------------------------------------
        if dataset_source in ("ha", "ha_api"):
            # Home Assistant history: CSV export or read directly from HA
            t_ha = time.time()

            def progress_cb_ha(label: str, frac: float) -> None:
                """Progress callback expected by ha_loader."""
                try:
                    pct = int(max(0, min(frac * 100.0, 100.0)))
                except Exception:
                    pct = 0
                status_container.write(f"{label}…")

            if payload:
                status_container.write(f"Preparing {payload['label']}...")
                res = ha_loader.process_ha_frame(
                    payload["frame"].copy(),
                    user_config,
                    progress_cb=progress_cb_ha,
                    heartbeat_baseline=active_baseline,
                )
            else:
                status_container.write("Loading Home Assistant history CSV data...")
                res = ha_loader.process_ha_files(
                    files,
                    user_config,
                    progress_cb=progress_cb_ha,
                    heartbeat_baseline=active_baseline,
                )
            _log(f"ha_loader.process_ha_files secs={time.time()-t_ha:.3f}")
            if not res or res.get("df") is None or res["df"].empty:
                status_container.update(
                    label="Error: No usable Home Assistant data found.",
                    state="error",
                )
                return None

            # ha_loader already applies physics + runs + daily
            df = res["df"]
            runs = res["runs"]
            daily = res["daily"]
            patterns = res.get("patterns")
            raw_history = res.get("raw_history")
            raw_events = res.get("raw_events")
            provenance = res.get("provenance")
            provenance_summary = res.get("provenance_summary", {})
            unmapped_entities = res.get("unmapped_entities", [])
            baseline_data = res.get("baselines") or active_baseline
            baseline_path = st.session_state.get("heartbeat_baseline_path")

        else:
            # Grafana CSV or direct InfluxDB: same long-form frames, same pipeline
            progress_cb = lambda t, p: status_container.write(f"Reading: {t}")

            t_load = time.time()
            months = _payload_months(payload) if is_influx else []
            if len(months) > 1:
                # Long InfluxDB periods: one calendar month at a time (chunked.py), so
                # memory stays at about one month's worth; finished months are saved
                # and reused by later loads.
                import config as _therm_config

                status_container.write(f"Processing {payload['label']} month by month…")
                bar = status_container.progress(0.0)

                def month_progress(i: int, n: int, label: str) -> None:
                    bar.progress(min((i - 1) / n, 1.0), text=f"{label} · month {i} of {n}")

                result = chunked.process_period(
                    files["numeric_dfs"], files["state_dfs"], user_config, months[0][0], months[-1][1],
                    heartbeat_baseline=active_baseline,
                    calc_version=getattr(_therm_config, "CALC_VERSION", ""),
                    store=_month_store(),
                    progress=month_progress,
                )
                bar.empty()
                if result is None:
                    status_container.update(label="Error: No data found", state="error")
                    return None
                res = {"baseline_path": None}
            elif is_influx:
                status_container.write(f"Preparing {files['label']}...")
                res = data_loader.load_and_clean_frames(
                    files["numeric_dfs"],
                    files["state_dfs"],
                    user_config,
                    progress_cb,
                    heartbeat_baseline=active_baseline,
                    on_message=_ui_message,
                )
            else:
                status_container.write("Loading and merging files (Numeric + State)...")
                res = data_loader.load_and_clean_data(
                    files,
                    user_config,
                    progress_cb,
                    heartbeat_baseline=active_baseline,
                    on_message=_ui_message,
                )
            _log(f"load secs={time.time()-t_load:.3f}")
            if len(months) <= 1:
                if not res or res.get("df") is None or res["df"].empty:
                    status_container.update(label="Error: No data found", state="error")
                    return None

                status_container.write("Applying physics, detecting runs and calculating daily stats...")
                result = pipeline.run_frames_pipeline(res, user_config)
            df = result["df"]
            runs = result["runs"]
            daily = result["daily"]
            raw_events = result["raw_events"]
            patterns = result["patterns"]
            raw_history = result["raw_history"]
            provenance = result["provenance"]
            provenance_summary = result["provenance_summary"]
            unmapped_entities = result["unmapped_entities"]
            baseline_data = result["baselines"] or active_baseline
            baseline_path = res.get("baseline_path")

        # ------------------------------------------------------------------
        # 4. Finalise & cache (capabilities are derived in _install_processed)
        # ------------------------------------------------------------------
        status_container.write("Finalising results…")
        global_stats = (result["global_stats"] if isinstance(df, chunked.MonthFrames)
                        else processing.compute_global_stats(df))
        cache = {
            "key": combined_key,
            "source": dataset_source,
            "df": df,
            "runs": runs,
            "daily": daily,
            "patterns": patterns,
            # keep pre-physics merged dataframe for deep debugging
            "raw_history": raw_history,
            "raw_events": raw_events,
            "provenance": provenance,
            "provenance_summary": provenance_summary,
            "unmapped_entities": unmapped_entities,
            "global_stats": global_stats,
            # guard against stale profile names being reused
            "profile_name": user_config.get("profile_name") if isinstance(user_config, dict) else None,
            # shown as "Showing … · processed at …" when the result is reopened later
            "label": payload["label"] if payload else None,
            "processed_at": _now_label(),
        }
        if disk_key:
            status_container.write("Saving processed results for next time...")
            processed_cache.save(disk_key, {
                **{k: v for k, v in cache.items() if k != "key"},
                "_baseline_data": baseline_data,
                "_baseline_path": baseline_path,
            })
            if isinstance(df, chunked.MonthFrames):
                # Its saved months must outlive the month store's pruning while this result is kept.
                _month_store().pin(disk_key, df.keys())
        _log(f"total secs={time.time()-t_start:.3f} source={dataset_source} rows={len(df) if df is not None else 0}")
        installed = _install_processed(cache, baseline_data, baseline_path)
        if auto_heartbeat:
            status_container.write("Updating the sensor heartbeat…")
            _auto_save_heartbeat(user_config, raw_events)
        # Only now is everything finished: remove the panel, confirm briefly, and
        # bring the dashboard into view from its top.
        status_slot.empty()
        summary = _analysis_summary(df, cache["label"], cache["processed_at"])
        st.session_state["analysis_summary"] = summary
        if disk_key:
            _remember_last_result(disk_key, payload, user_config, cache["processed_at"])
        st.toast(f"✅ Analysis complete: {summary['days']} days analysed")
        _scroll_main_to_top()
        return installed

    except Exception as e:
        status_container.update(label="Processing Failed", state="error")
        st.error(f"An error occurred: {e}")

        # Detailed traceback for debugging
        tb = traceback.format_exc()
        with st.expander("Debug: Full traceback", expanded=False):
            st.code(tb)

        return None
    finally:
        memory_guard.ANALYSIS_LOCK.release()

def _now_label() -> str:
    """'28 Sep 10:06': when a result was processed (shown when it is reopened later)."""
    import datetime as _dt

    return _dt.datetime.now().strftime("%d %b %H:%M")


def _analysis_summary(df, label=None, processed_at=None) -> dict:
    """What the current analysis covers, for the one-line summary above the dashboard."""
    if isinstance(df, chunked.MonthFrames):
        days, first, last = df.days, df.first, df.last
    else:
        days = int(pd.Index(df.index.normalize()).nunique()) if df is not None and len(df) else 0
        first = df.index.min() if days else None
        last = df.index.max() if days else None
    return {"days": days, "first": first, "last": last, "label": label,
            "processed_at": processed_at or _now_label()}


def _remember_last_result(disk_key: str, payload: dict, user_config: dict, processed_at) -> None:
    """Record the analysis so the next visit can reopen it straight away (processed_cache)."""
    try:
        entry = {
            "disk_key": disk_key,
            "source": payload.get("source"),
            "profile": (user_config or {}).get("profile_name"),
            "label": payload.get("label"),
            "processed_at": processed_at,
        }
        # Keep a result for every source/profile pair. ``last_result`` remains for
        # backward compatibility with installations written by earlier versions.
        state = profile_store.get_state()
        all_results = dict(state.get("last_results") or {})
        by_profile = dict(all_results.get(entry["source"]) or {})
        by_profile[entry["profile"]] = entry
        all_results[entry["source"]] = by_profile
        profile_store.update_state(last_result=entry, last_results=all_results)
    except OSError:
        pass


def _restore_last_result(source: str, profile_name) -> Optional[dict]:
    """
    Reopen the last analysis of this profile and source from disk, so a returning
    user sees results immediately instead of a blank page. It is shown
    as "your last analysis"; Load latest data refreshes it.
    """
    state = profile_store.get_state()
    last = ((state.get("last_results") or {}).get(source) or {}).get(profile_name)
    if not last:
        # Backward-compatible fallback for state written before per-source history.
        last = state.get("last_result") or {}
    if not last.get("disk_key") or last.get("source") != source or last.get("profile") != profile_name:
        return None
    stored = _load_processed(last["disk_key"])
    if stored is None:
        return None
    baseline_data = stored.pop("_baseline_data", None)
    baseline_path = stored.pop("_baseline_path", None)
    if source_ui.running_in_addon():
        # The result reopens with the heartbeat it was computed with. If a newer one has been
        # saved since (by that or a later analysis), it must apply from the next load of new
        # data; the session flag that said so did not survive the restart.
        saved_hb, _meta = heartbeat_store.load(
            profile_name, (st.session_state.get("system_config") or {}).get("mapping"))
        if saved_hb and saved_hb != (baseline_data or {}):
            st.session_state["heartbeat_pending"] = profile_name
    stored["key"] = ("restored", last["disk_key"])
    stored["source"] = stored.get("source", _dataset_source_id(source))
    stored["restored"] = True
    st.session_state["dataset_source"] = stored["source"]
    st.session_state["analysis_summary"] = _analysis_summary(
        stored.get("df"), stored.get("label") or last.get("label"), stored.get("processed_at") or last.get("processed_at"))
    return _install_processed(stored, baseline_data, baseline_path)


def _scroll_main_to_top() -> None:
    """Scroll the main panel to the top (after processing finishes)."""
    _run_parent_script(
        """try {
          const app = window.parent.document.querySelector('[data-testid="stAppViewContainer"]')
                      || window.parent.document.querySelector('section.main');
          if (app && app.scrollTo) { app.scrollTo({top: 0}); } else { window.parent.scrollTo(0, 0); }
        } catch (e) { /* cross-origin or layout change: leave the scroll position */ }""",
        key="therm-js-scroll-done",
    )


def _auto_save_heartbeat(user_config: dict, raw_events) -> None:
    """Add-on: keep the profile's heartbeat current from long analyses (heartbeat_store).
    The new heartbeat applies to the next data loaded; never fails the analysis."""
    from config import SENSOR_ROLES

    profile = user_config.get("profile_name")
    try:
        meta = heartbeat_store.maybe_update(profile, raw_events, SENSOR_ROLES)
    except Exception as e:  # noqa: BLE001 - a heartbeat problem must not lose the results
        _log(f"heartbeat auto-save failed: {e}")
        return
    if meta:
        # Not activated in this session yet: the active heartbeat is part of the processed
        # result's key, so swapping it now made the next click (e.g. Run Inspector) process
        # the whole period again. get_processed_data switches to it for the next new data.
        st.session_state["heartbeat_pending"] = profile
        if st.session_state.get("heartbeat_auto_off") == profile:
            st.session_state.pop("heartbeat_auto_off", None)
        st.toast(f"Sensor heartbeat updated from {meta.get('days_analyzed')} days of data; "
                 "it applies from the next load of new data.")


def _install_processed(cache: dict, baseline_data, baseline_path) -> dict:
    """Make a processed result (fresh or from processed_cache) the session's active data."""
    df = cache["df"]
    if isinstance(df, chunked.MonthFrames):  # long analysis: minutes stay on disk per month
        has_flowrate, has_heat_sensor = df.flags["has_flowrate"], df.flags["has_heat_sensor"]
    else:
        has_flowrate = "FlowRate" in df.columns and df["FlowRate"].notna().any()
        has_heat_sensor = (
            "Heat" in df.columns
            and pd.to_numeric(df["Heat"], errors="coerce").fillna(0).abs().sum() > 0
        )
    caps = st.session_state.get("capabilities", {})
    caps["has_flowrate"] = has_flowrate
    caps["has_heat_sensor"] = has_heat_sensor
    caps["has_energy_channel"] = has_flowrate or has_heat_sensor
    st.session_state["capabilities"] = caps

    # Release ZIP bytes tied to a previous dataset before installing the new entry.
    st.session_state.pop("debug_download_bundles", None)
    st.session_state["cached"] = cache
    st.session_state["raw_history_df"] = cache.get("raw_events")
    if baseline_data:
        st.session_state["heartbeat_baseline"] = baseline_data
        # The results were computed with this heartbeat for this profile.
        st.session_state["heartbeat_profile"] = cache.get("profile_name")
    st.session_state["heartbeat_baseline_path"] = baseline_path
    _release_memory()
    return cache


def _release_memory() -> None:
    """Return freed heap pages to the OS after heavy processing (glibc; no-op elsewhere),
    so the add-on's resident memory drops back instead of staying at its peak."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _build_engine_debug_traces(df: pd.DataFrame, max_rows: int = 50) -> list[dict]:
    """
    Build a compact set of 'interesting' engine rows for JSON debug traces.

    Heuristics:
    - COP_Real < 0 or > 7
    - Heat < 0
    - is_heating and is_DHW both True
    Falls back to the last `max_rows` rows if nothing matches.
    """
    if df is None or df.empty:
        return []

    # Candidate columns – we keep whichever are present
    candidate_cols = [
        "Heat", "Heat_Heating", "Heat_DHW",
        "Power", "Power_Heating", "Power_DHW",
        "FlowTemp", "ReturnTemp", "DeltaT", "FlowRate",
        "ValveMode", "DHW_Mode",
        "is_heating", "is_DHW", "is_active",
        "COP_Real", "COP_Graph",
    ]
    cols = [c for c in candidate_cols if c in df.columns]

    if not cols:
        return []

    # Build masks for "interesting" rows
    idx = df.index
    cop = pd.to_numeric(df.get("COP_Real", pd.Series(index=idx)), errors="coerce")
    heat = pd.to_numeric(df.get("Heat", pd.Series(index=idx)), errors="coerce")

    masks = []

    if not cop.isna().all():
        masks.append(cop < 0)
        masks.append(cop > 7)

    if not heat.isna().all():
        masks.append(heat < 0)

    if "is_heating" in df.columns and "is_DHW" in df.columns:
        masks.append(df["is_heating"] & df["is_DHW"])

    if masks:
        mask = masks[0]
        for m in masks[1:]:
            mask = mask | m
        interesting = df[mask]
    else:
        interesting = df

    if interesting.empty:
        interesting = df.tail(max_rows)
    else:
        interesting = interesting.head(max_rows)

    # Serialise to a JSON-friendly list of dicts, including the index as Time_index
    records: list[dict] = []
    for idx, row in interesting[cols].iterrows():
        rec: dict[str, object] = {}
        # index as a label
        if isinstance(idx, pd.Timestamp):
            rec["Time_index"] = idx.isoformat()
        else:
            rec["Time_index"] = str(idx)

        for c in cols:
            val = row[c]
            if isinstance(val, pd.Timestamp):
                val = val.isoformat()
            elif isinstance(val, (np.floating, float, int, np.integer)):
                val = float(val)
            rec[c] = val
        records.append(rec)

    return records


# === MAIN LOGIC ===
if uploaded_files or is_direct_source:
    if show_inspector:
        st.title("Pre-Flight Inspector")
        summary, details_all = inspector.inspect_raw_files(uploaded_files)

        # Store inspector outputs for downstream debug bundle (optional, AI-facing)
        st.session_state["inspector_summary"] = summary
        st.session_state["inspector_details"] = details_all

        st.dataframe(summary, width="stretch")
        file_details = details_all.get("file_details", {})
        sensor_debug = details_all.get("sensor_debug", {})

        # Keep the entity list in sync with the mapping UI options
        try:
            all_entities = []
            for info in file_details.values():
                ents = info.get("entities_found") or []
                all_entities.extend(ents)
            if all_entities:
                # Stable cache key to avoid stale entities across file changes
                files_key = sorted((getattr(f, "name", ""), getattr(f, "size", 0)) for f in uploaded_files)
                st.session_state["available_sensors"] = sorted(set(all_entities))
                st.session_state["available_sensors_files_key"] = files_key
        except Exception:
            pass

        # ---- FILE-LEVEL DETAILS ----
        for fname, info in file_details.items():
            with st.expander(f"File Details: {fname}", expanded=False):
                entities = info.get("entities_found", [])
                st.write(f"Entities found: {len(entities)}")
                if entities:
                    st.code("\n".join(entities))

                ranges = info.get("entity_date_ranges") or {}
                if ranges:
                    st.markdown("**Entity Date Ranges:**")
                    st.dataframe(
                        pd.DataFrame.from_dict(ranges, orient="index"),
                        width="stretch",
                    )

                # Optional: show raw column list
                cols = info.get("columns_raw", [])
                if cols:
                    st.markdown("**Columns:**")
                    st.code("\n".join(cols))

        # ---- SENSOR-LEVEL DEBUG ----
        for fname, sensors in sensor_debug.items():
            with st.expander(f"Sensor Debug: {fname}", expanded=False):
                if "error" in sensors:
                    st.error(sensors["error"])
                    continue

                st.write(f"Sensors detected: {len(sensors)}")
                st.json(sensors)

    else:
        # --- CONFIGURATION WORKFLOW ---
        if force_system_setup or "system_config" not in st.session_state:
            _log("config_render_start")
            if force_system_setup and "system_config" in st.session_state:
                # Leave Setup without saving (otherwise the only way out is Save).
                if main_nav.button("← Back to analysis (discard changes)"):
                    _discard_setup_draft()
                    st.session_state.pop("force_system_setup", None)
                    st.rerun()
            if is_direct_source and not _is_demo and "system_config" in st.session_state:
                # Occasional maintenance: re-fetch after data was corrected or back-filled at the source.
                with st.expander("Saved source data", expanded=False):
                    source_ui.render_saved_data_control(source_choice, st.session_state.get("system_config"))
            config_object = mapping_ui.render_configuration_interface(uploaded_files)
            _log("config_render_done")
            if config_object:
                st.session_state["system_config"] = config_object
                _save_active_profile(config_object)
                st.session_state.pop("force_system_setup", None)
                st.rerun()

        else:
            profile_now = st.session_state["system_config"].get("profile_name")
            # 1. GET DATA: the load controls in the sidebar (direct sources)
            if is_direct_source:
                with side_load:
                    source_payload = source_ui.render_source(source_choice, st.session_state.get("system_config"))
            _log("process_data_start")
            data = get_processed_data(
                source_payload if is_direct_source else uploaded_files,
                st.session_state["system_config"],
            )
            if data is None and is_direct_source and source_payload is None:
                # Nothing loaded in this session yet (or the period was changed without
                # pressing Load): keep showing what the user last looked at, or reopen
                # their last analysis from disk, instead of a blank page.
                previous = st.session_state.get("cached")
                expected_source = _dataset_source_id(source_choice)
                if (previous and previous.get("profile_name") == profile_now
                        and previous.get("source") == expected_source):
                    data = previous
                else:
                    attempted = st.session_state.get("restore_attempted")
                    if not isinstance(attempted, set):
                        attempted = set()
                    restore_key = (source_choice, profile_now)
                    if restore_key not in attempted:
                        attempted.add(restore_key)
                        st.session_state["restore_attempted"] = attempted
                        data = _restore_last_result(source_choice, profile_now)
            _log("process_data_done")

            if data is None and is_direct_source:
                with main_nav:
                    st.info(f"👈 Choose a period in the sidebar and press **Load latest data** to analyse "
                            f"your heat pump from {source_choice}.")
            # Shown data isn't from a Load in this session (reopened, or the period was
            # changed without loading): say so in the summary line above the dashboard.
            showing_previous = data is not None and is_direct_source and source_payload is None

            # 2. View toggle at the top of the main panel. Data Quality is an occasional
            # check, opened from the sidebar; while it is shown neither tab is selected.
            mode = None
            if data and "df" in data:
                views = {"Trends": "Long-Term Trends", "Run Inspector": "Run Inspector",
                         "Data Quality": "Data Quality Audit"}
                if "view_mode" not in st.session_state:
                    st.session_state["view_mode"] = (
                        None if st.session_state.get("last_view") == "Data Quality" else
                        st.session_state.get("last_view", "Trends"))
                with main_nav:
                    picked_view = st.segmented_control(
                        "View", ["Trends", "Run Inspector"], key="view_mode",
                        label_visibility="collapsed",
                    )
                # Clicking the selected tab again deselects it: stay on the last view.
                if picked_view:
                    st.session_state["last_view"] = picked_view
                mode = views[st.session_state.get("last_view", "Trends")]

            # 3. Sidebar: Global Stats (canonical), Troubleshooting
            with st.sidebar:
                if data and "df" in data:
                    # Canonical global stats from processing engine
                    stats = data.get("global_stats")
                    if not isinstance(stats, dict):
                        # Compatibility for a cache created by an older app process.
                        stats = processing.compute_global_stats(data["df"])
                    total_heat = stats["total_heat_kwh"]
                    total_elec = stats["total_elec_kwh"]
                    global_cop = stats["global_cop"]

                    runs_list = data.get("runs") or []
                    runs_detected = len(runs_list)
                    heating_runs = sum(1 for r in runs_list if r.get("run_type") == "Heating")
                    dhw_runs = sum(1 for r in runs_list if r.get("run_type") == "DHW")
                    cooling_runs = sum(1 for r in runs_list if r.get("run_type") == "Cooling")
                else:
                    if not is_direct_source:
                        st.info("Upload data and configure your system to begin analysis.")
                    total_heat = total_elec = global_cop = 0.0
                    runs_detected = 0
                    heating_runs = 0
                    dhw_runs = 0
                    cooling_runs = 0

                # --- Whole system: the period's totals, the same on every view (the Trends filter doesn't
                # change them). Label | value rows: compact enough to keep Setup / Data Quality in view.
                if data and mode:
                    st.markdown(
                        """
                        <style>
                        .therm-summary .title { font-weight: 600; font-size: 0.95rem; margin: 0.4rem 0 0.3rem 0; }
                        .therm-summary table { border: none; border-collapse: collapse; width: 100%;
                                               font-size: 0.85rem; margin: 0; }
                        .therm-summary tr, .therm-summary td { border: none !important; background: none !important; }
                        .therm-summary td { padding: 0.1rem 0; }
                        .therm-summary td.k { opacity: 0.7; width: 40%; }
                        .therm-summary .sub { font-size: 0.72rem; opacity: 0.7; margin-top: 0.15rem; }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    system_cop = view_trends.period_cop(data["daily"], whole_system=True) if data.get("daily") is not None else None
                    cop_text = f"{global_cop:.2f}" + (f" · system {system_cop:.2f}" if system_cop is not None else "")
                    rows = [
                        ("Heat", f"{total_heat:,.0f} kWh"),
                        ("Electricity", f"{total_elec:,.0f} kWh"),
                        ("COP", cop_text),
                        ("Runs", f"{runs_detected:,}"),
                    ]
                    run_split = f"{heating_runs:,} heating · {dhw_runs:,} hot water" + (
                        f" · {cooling_runs:,} cooling" if cooling_runs else "")
                    table = "".join(f"<tr><td class='k'>{k}</td><td>{v}</td></tr>" for k, v in rows)
                    st.markdown(
                        "<div class='therm-summary'><div class='title'>Whole system</div>"
                        f"<table>{table}</table><div class='sub'>{run_split}</div>"
                        f"<div class='sub'>COP basis (primary Power): "
                        f"{cop_provenance(st.session_state.get('system_config'), getattr(data.get('daily'), 'attrs', {}).get('heat_source')).get('display', 'Not specified')}. "
                        "This describes primary Power only, not optional Indoor Power.</div></div>",
                        unsafe_allow_html=True,
                    )

                # Setup and the occasional Data Quality check, under the stats.
                c_setup, c_quality = st.columns(2)
                if c_setup.button("⚙️ Setup", width="stretch",
                                  help="Sensors, rooms, zones, thresholds and electricity prices"):
                    st.session_state["force_system_setup"] = True
                    st.rerun()
                if data and "df" in data:
                    c_quality.button(
                        "Data Quality", width="stretch", on_click=_open_data_quality,
                        type="primary" if mode == "Data Quality Audit" else "secondary",
                        help="Sensor coverage, gaps and heartbeats for this period",
                    )


                # NOTE:
                # In Analysis Mode screens we intentionally do NOT show
                # the "Download Profile" button. Profile download remains
                # available as part of the System Setup / mapping UI.

                # --- Data Debugger at the bottom ---
                with st.expander("Troubleshooting (debug files)", expanded=False):
                    # Toggle for engine-level debug traces in processing.py
                    debug_flag = st.checkbox(
                        "Enable engine debug traces (JSON only)",
                        value=st.session_state.get("debug_engine", False),  # default OFF
                    )
                    st.session_state["debug_engine"] = debug_flag
                    engine_context.set_debug(
                        debug_flag,
                        st.session_state.setdefault("engine_debug_traces", []),
                    )

                    if data is not None and "df" in data:
                        st.caption(
                            "Debug ZIPs can be large. They are built only when requested, "
                            "so normal view changes and run navigation stay responsive."
                        )
                        prepare_debug = st.button(
                            "Prepare debug download bundles",
                            key="prepare_debug_download_bundles",
                        )
                        if prepare_debug:
                            try:
                                with st.spinner("Preparing debug bundles..."):
                                    payload = debug_exports.build_debug_download_bundles(
                                        data,
                                        config=st.session_state.get("system_config"),
                                        capabilities=st.session_state.get("capabilities", {}),
                                        dataset_source=st.session_state.get("dataset_source", "unknown"),
                                        include_engine_traces=debug_flag,
                                        trace_builder=_build_engine_debug_traces,
                                        inspector_summary=st.session_state.get("inspector_summary"),
                                        inspector_details=st.session_state.get("inspector_details"),
                                    )
                                st.session_state["debug_download_bundles"] = {
                                    "data_key": data.get("key"),
                                    "debug_engine": debug_flag,
                                    "payload": payload,
                                }
                            except Exception as exc:
                                st.warning(f"Could not prepare debug bundles: {exc}")

                        prepared = st.session_state.get("debug_download_bundles") or {}
                        if (
                            prepared.get("data_key") == data.get("key")
                            and prepared.get("debug_engine") == debug_flag
                        ):
                            payload = prepared.get("payload") or {}
                            ts_label = payload.get("timestamp_label", "debug")
                            st.download_button(
                                "⬇ Download merged CSV + debug JSON (ZIP)",
                                data=payload.get("merged_zip", b""),
                                file_name=f"THERM_debug_merged_and_json_{ts_label}.zip",
                                mime="application/zip",
                                key="download_zip_merged_json_lazy",
                                on_click="ignore",
                            )
                            all_zip = payload.get("all_zip")
                            st.download_button(
                                "⬇ Download all debug files (merged + raw + JSON) (ZIP)",
                                data=all_zip or b"",
                                file_name=f"THERM_debug_all_{ts_label}.zip",
                                mime="application/zip",
                                key="download_zip_all_lazy",
                                disabled=all_zip is None,
                                on_click="ignore",
                                help=(
                                    None if all_zip is not None
                                    else "Raw pre-physics dataframe not available for this run."
                                ),
                            )

                # About therm (analysis views)
                with st.sidebar.expander("About therm", expanded=False):
                    st.markdown("**therm (beta) - Heat Pump Performance Analysis**")

            # 3. Render Dashboard (main panel)
            if data and mode:
                caps = st.session_state.get("capabilities", {})
                has_flowrate = caps.get("has_flowrate", True)
                has_energy_channel = caps.get("has_energy_channel", True)

                if not has_energy_channel:
                    st.info(
                        "No Flow Rate or Heat output sensor mapped — energy output (Heat kWh), COP and SCOP are disabled. "
                        "The dashboard is running in Power & Temps only mode."
                    )

                # Context injection for AI
                st.session_state["ai_context_user"] = st.session_state["system_config"].get("ai_context", {})

                if mode == "Long-Term Trends":
                    view_trends.render_long_term_trends(
                        data["daily"], data["df"], data["runs"], st.session_state.get("system_config")
                    )
                elif mode == "Run Inspector":
                    view_runs.render_run_inspector(data["df"], data["runs"])
                elif mode == "Data Quality Audit":
                    source_ui.render_notes()  # mapped sensors missing or without history in the last load
                    hb_path = st.session_state.get("heartbeat_baseline_path")
                    view_quality.render_data_quality(
                        data["daily"],
                        data["df"],
                        data.get("unmapped_entities", []),
                        data["patterns"],
                        hb_path,
                        raw_events=data.get("raw_events"),
                        provenance_summary=data.get("provenance_summary"),
                        user_config=st.session_state.get("system_config"),
                    )

else:
    # Nudge initial info message down slightly
    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    st.info("Upload CSV files to begin.")
    st.sidebar.markdown("---")
