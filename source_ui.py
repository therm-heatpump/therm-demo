# source_ui.py
"""
Sidebar UI for therm's direct data sources.

- Home Assistant: recorder history over HA's REST API (sources/ha_api.py). In the
  add-on this uses the Supervisor token, so no setup is needed.
- InfluxDB: HA's InfluxDB integration database (sources/influx.py). Credentials
  come from THERM_INFLUX_* environment variables (set by the add-on options) or an
  [influx] table in .streamlit/secrets.toml.

Each render_*_source() draws the period and entity controls in the current
container and, once loaded, returns a payload for app.get_processed_data. Both
sources name entities `sensor.x` (InfluxDB from its entity_id and domain tags; only
series without a domain tag are bare `x`); resolve_mapping() adapts older profiles
with bare names, so one profile works with both.
Credentials are never stored in a profile or shown in the UI.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Optional

import streamlit as st

import hashlib

import memory_guard
import profile_store
import schema_defs
from sources.ha_api import HASettings, fetch_history, list_states
from sources.influx import InfluxSettings, fetch_period, list_entities

SOURCE_HA = "Home Assistant"
SOURCE_INFLUX = "InfluxDB"
SOURCE_CSV = "Upload CSV files"


def running_in_addon() -> bool:
    """True inside the HA add-on (the Supervisor provides the HA API token)."""
    import os

    return bool(os.environ.get("SUPERVISOR_TOKEN"))


def available_sources() -> list[str]:
    """Data-source choices: HA + InfluxDB in the add-on; CSV + InfluxDB (+ HA if configured) standalone."""
    if running_in_addon():
        return [SOURCE_HA, SOURCE_INFLUX]
    sources = [SOURCE_CSV, SOURCE_INFLUX]
    if ha_settings() is not None:
        sources.append(SOURCE_HA)
    return sources


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
def influx_settings() -> Optional[InfluxSettings]:
    """Env vars first (add-on), then st.secrets['influx'] (local runs), else None."""
    settings = InfluxSettings.from_env()
    if settings is not None:
        return settings
    try:
        sec = st.secrets.get("influx")
    except Exception:  # no secrets.toml
        sec = None
    if not sec or not sec.get("host"):
        return None
    return InfluxSettings(
        host=str(sec["host"]),
        port=int(sec.get("port", 8086)),
        database=str(sec.get("database", "homeassistant")),
        username=sec.get("username") or None,
        password=sec.get("password") or None,
        ssl=bool(sec.get("ssl", False)),
        version=int(sec.get("version", 1)),
    )


def ha_settings() -> Optional[HASettings]:
    """Supervisor proxy in the add-on, else THERM_HA_URL/THERM_HA_TOKEN, else the
    [ha] section of .streamlit/secrets.toml (url, token) for local runs, else None."""
    settings = HASettings.from_env()
    if settings is not None:
        return settings
    try:
        sec = st.secrets.get("ha")
    except Exception:  # no secrets.toml
        sec = None
    if not sec or not sec.get("url") or not sec.get("token"):
        return None
    url = str(sec["url"]).strip().rstrip("/")
    return HASettings(base_url=url if url.endswith("/api") else f"{url}/api", token=str(sec["token"]).strip())


def _identity(settings: InfluxSettings) -> str:
    """Connection identity for cache keys and display: never includes the password."""
    scheme = "https" if settings.ssl else "http"
    return f"{scheme}://{settings.host}:{settings.port}/{settings.database}"


def _ha_identity(settings: HASettings) -> str:
    return settings.base_url  # the token is never part of the identity


# ----------------------------------------------------------------------
# Cached calls (the settings objects are excluded from hashing: leading _)
# ----------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def _cached_entities(identity: str, _settings: InfluxSettings) -> list[str]:
    return list_entities(_settings)


def _influx_cache_root(identity: str, tz: str):
    """
    Month cache folder for one InfluxDB database and timezone (never the password).
    "influx-minute-tz" holds 1-minute rows with time-zone-aware timestamps. The
    earlier caches ("influx", "influx-minute": local times without a zone, which
    merged the repeated autumn hour) are not read any more and can be deleted.
    """
    key = hashlib.sha1(f"{identity}|{tz}".encode()).hexdigest()[:12]
    return profile_store.cache_dir("influx-minute-tz", key)


def _ha_cache_root(settings: HASettings, tz: str):
    """Day cache folder for one Home Assistant instance and timezone (never the token)."""
    return profile_store.cache_dir("ha-history", hashlib.sha256(f"{settings.base_url}|{tz}".encode()).hexdigest()[:12])


def _fetch_influx_cached(settings: InfluxSettings, identity: str, entities: tuple[str, ...],
                         start: date, end: date, tz: str, state_entities: set):
    """Month-by-month fetch (1-minute means + state changes) through the disk cache, with a progress bar."""
    bar = st.progress(0.0, text="Preparing InfluxDB fetch…")

    def progress(i: int, n: int, label: str) -> None:
        bar.progress(min(i / n, 1.0), text=f"{label} · month {i} of {n}")

    frames = fetch_period(
        settings, list(entities), start, end, timezone=tz, state_entities=state_entities,
        cache_root=_influx_cache_root(identity, tz), progress=progress,
    )
    bar.empty()
    return frames


@st.cache_data(ttl=600, show_spinner=False)
def _cached_ha_entities(identity: str, _settings: HASettings) -> list[str]:
    states = list_states(_settings)
    return sorted(states["entity_id"].dropna().astype(str).tolist())


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_instance_config(identity: str, _settings: HASettings) -> dict:
    from sources.ha_api import instance_config

    return instance_config(_settings)


def home_assistant_defaults() -> dict:
    """Home Assistant's own time zone, currency and location name ({} when HA can't be asked, e.g. an
    InfluxDB-only set-up outside the add-on)."""
    settings = ha_settings()
    return _cached_instance_config(_ha_identity(settings), settings) if settings is not None else {}


def _cached_ha_history(identity: str, entities: tuple[str, ...], start: date, end: date, tz: str,
                       _settings: HASettings):
    """HA history through the day cache: completed days from disk, today (and
    anything missing) from HA. Deliberately no whole-period memory cache, so
    every Load returns the latest readings."""
    from sources.ha_api import fetch_history_cached

    return fetch_history_cached(_settings, list(entities), start, end, timezone=tz,
                                cache_root=_ha_cache_root(_settings, tz))


# ----------------------------------------------------------------------
# Mapping helpers
# ----------------------------------------------------------------------
def mapped_entities(user_config: Optional[Dict[str, Any]]) -> list[str]:
    mapping = (user_config or {}).get("mapping", {}) or {}
    return sorted({str(v) for v in mapping.values() if v not in (None, "", "None")})


def resolve_mapping(mapping: Dict[str, Any], available: list[str]) -> Dict[str, Any]:
    """
    Adapt entity names to the active source: `sensor.x` ↔ `x`.

    A value already present in `available` is kept. Otherwise its object_id is
    matched: to the bare name (InfluxDB), or to a full `domain.object_id`
    (Home Assistant, preferring `sensor.`). Unmatched values are kept so the
    "not found" warning can name them.
    """
    avail = set(available)
    by_object: Dict[str, list[str]] = {}
    for e in available:
        by_object.setdefault(e.split(".", 1)[-1], []).append(e)

    out: Dict[str, Any] = {}
    for role, value in (mapping or {}).items():
        if value in (None, "", "None") or value in avail:
            out[role] = value
            continue
        obj = str(value).split(".", 1)[-1]
        candidates = by_object.get(obj, [])
        if candidates:
            out[role] = sorted(candidates, key=lambda e: (not e.startswith("sensor."), e))[0]
        else:
            out[role] = value
    return out


# ----------------------------------------------------------------------
# Entity lists for Setup
# ----------------------------------------------------------------------
def available_entities() -> Optional[list[str]]:
    """All entity_ids in InfluxDB, or None (reason shown in the current container)."""
    settings = influx_settings()
    if settings is None:
        st.info(
            "InfluxDB is not configured. In the add-on, set the InfluxDB options; "
            "when running locally, add an `[influx]` section to `.streamlit/secrets.toml`."
        )
        return None
    try:
        return _cached_entities(_identity(settings), settings)
    except Exception as e:  # message is already credential-free (sources.influx)
        st.error(f"Cannot reach InfluxDB: {e}")
        return None


def ha_available_entities() -> Optional[list[str]]:
    """All entity_ids in Home Assistant, or None (reason shown in the current container)."""
    settings = ha_settings()
    if settings is None:
        st.info(
            "Home Assistant is not reachable: run therm as the Home Assistant add-on, or set "
            "`THERM_HA_URL` and `THERM_HA_TOKEN` (a long-lived access token) when running locally."
        )
        return None
    try:
        return _cached_ha_entities(_ha_identity(settings), settings)
    except Exception as e:  # message is already token-free (sources.ha_api)
        st.error(f"Cannot read Home Assistant: {e}")
        return None


def setup_entities(source: str) -> Optional[list[str]]:
    return ha_available_entities() if source == SOURCE_HA else available_entities()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_influx_metadata(identity: str, _settings: InfluxSettings) -> list[dict]:
    from sources.influx import list_entity_metadata

    return list_entity_metadata(_settings)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_ha_metadata(identity: str, _settings: HASettings) -> list[dict]:
    from sources.ha_api import entity_platforms

    states = list_states(_settings)
    platforms = entity_platforms(_settings)
    rows = states.to_dict("records")
    for r in rows:
        r["platform"] = platforms.get(r["entity_id"])
    return rows


def setup_entity_metadata(source: str) -> list[dict]:
    """
    Entity metadata for automatic role matching (presets.auto_map): InfluxDB
    gives each entity's unit and domain; Home Assistant adds device class,
    friendly name and integration. [] when unavailable (matching then uses
    names only).
    """
    try:
        if source == SOURCE_HA:
            settings = ha_settings()
            return _cached_ha_metadata(_ha_identity(settings), settings) if settings else []
        settings = influx_settings()
        return _cached_influx_metadata(_identity(settings), settings) if settings else []
    except Exception:
        return []


# ----------------------------------------------------------------------
# Shared period / entity controls
# ----------------------------------------------------------------------
DEFAULT_PERIOD_DAYS = 7


def remembered_period(source: str) -> Optional[tuple]:
    """(start, last) the user last loaded from `source` (therm's state file), or None."""
    saved = (profile_store.get_state().get("last_period") or {}).get(source)
    try:
        start, last = date.fromisoformat(saved[0]), date.fromisoformat(saved[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (start, last) if start <= last else None


def remember_period(source: str, start: date, last: date) -> None:
    try:
        periods = dict(profile_store.get_state().get("last_period") or {})
        periods[source] = [start.isoformat(), last.isoformat()]
        profile_store.update_state(last_period=periods)
    except OSError:
        pass


def _default_period(source: str) -> tuple:
    """The last period loaded from this source, moved forward so it ends today if it
    ended on the day it was loaded (a rolling "last week" stays a rolling last week)."""
    today = date.today()
    saved = remembered_period(source)
    if saved:
        start, last = saved
        loaded_on = profile_store.get_state().get("last_period_loaded_on", {}).get(source)
        if loaded_on == last.isoformat() and last < today:  # it was "up to today" then
            shift = today - last
            start, last = start + shift, today
        return (start, min(last, today))
    return (today - timedelta(days=DEFAULT_PERIOD_DAYS), today)


def _request_controls(source: str, user_config: Optional[Dict[str, Any]], available: list[str],
                      period_help: str):
    """
    The load controls (sidebar): Period and Load. Only mapped sensors are fetched; problems with them are
    collected for the Data Quality view (render_notes). Returns
    (entities, start, end, last, tz, config, button_slot, notes_slot) or None.
    """
    base_cfg = user_config or {}
    if not mapped_entities(base_cfg):
        st.caption("Complete Setup first; therm then fetches only the sensors you mapped.")
        return None
    resolved = resolve_mapping(base_cfg.get("mapping", {}), available)
    config = {**base_cfg, "mapping": resolved}
    base = mapped_entities(config)

    today = date.today()
    # Stacked in the sidebar (a load bar across the top of the main panel
    # pushed the Daily Energy Balance chart below the fold).
    picked = st.date_input(
        "Period",
        value=_default_period(source),
        max_value=today,
        key=f"{source}_period",
        help=period_help,
        format="DD/MM/YYYY",
    )
    c_button, c_notes = st.container(), st.container()
    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        c_notes.caption("Pick a start and an end date.")
        return None
    start, last = picked
    end = last + timedelta(days=1)  # exclusive

    missing = [e for e in base if e not in set(available)]
    reset_notes()
    if missing:
        add_note("warning", f"Mapped but not found in {source}: " + ", ".join(f"`{m}`" for m in missing[:8])
                 + (" …" if len(missing) > 8 else ""))
        c_notes.caption(f"⚠️ {len(missing)} sensor{'s' if len(missing) != 1 else ''} not found: see Data Quality.")
    entities = tuple(sorted(set(base) - set(missing)))
    tz = base_cfg.get("timezone") or "Europe/Dublin"

    # InfluxDB periods over more than one calendar month are processed month by month
    # (app.get_processed_data → chunked.py), which needs far less memory.
    monthly = source == SOURCE_INFLUX and (start.year, start.month) != (last.year, last.month)
    guard = memory_guard.check((end - start).days * 1440, len(entities), monthly=monthly)
    if not guard["ok"]:
        st.error(
            f"This period needs about {guard['estimate'] / 1e9:.1f} GB of memory to process, but only "
            f"{guard['available'] / 1e9:.1f} GB is available on this Home Assistant host. Choose at most "
            f"about {guard['max_days']} days, or give the host more RAM. (Stopping here protects Home "
            "Assistant: running out of memory can affect other add-ons too.)"
        )
        return None
    return entities, start, end, last, tz, config, c_button, c_notes


def _fingerprint(frames: list, time_col: str) -> tuple:
    """Content identity of fetched data for processed_cache: row count, latest
    reading and a digest of every value, so new readings, corrected values or a
    back-fill with the same row count all give a new key."""
    import pandas as pd

    rows = sum(len(f) for f in frames)
    latest = max((str(f[time_col].max()) for f in frames if len(f)), default="")
    digest = hashlib.sha256()
    for f in frames:
        if len(f):
            cols = sorted(map(str, f.columns))
            digest.update("|".join(cols).encode())
            part = f[cols]
            obj = [c for c in cols if part[c].dtype == object]
            if obj:  # mixed/None states: hash their text form
                part = part.astype({c: str for c in obj})
            digest.update(pd.util.hash_pandas_object(part, index=False).to_numpy().tobytes())
    return rows, latest, digest.hexdigest()[:24]


def _loaded(source: str, request_key: tuple, entities: tuple, slot, start: date, last: date) -> Optional[int]:
    """None until Load is pressed for this request, then the load generation:
    each press is a fresh fetch (new readings for today / the current month),
    while ordinary reruns reuse the loaded data. The period is remembered for the
    next visit. (No tooltip: it stayed on screen over the controls.)"""
    state_key, gen_key = f"{source}_request", f"{source}_generation"
    if slot.button("Load latest data", type="primary", disabled=not entities, key=f"{source}_load",
                   width="stretch"):
        st.session_state[state_key] = request_key
        st.session_state[gen_key] = st.session_state.get(gen_key, 0) + 1
        remember_period(source, start, last)
        try:
            loaded_on = dict(profile_store.get_state().get("last_period_loaded_on") or {})
            loaded_on[source] = date.today().isoformat()
            profile_store.update_state(last_period_loaded_on=loaded_on)
        except OSError:
            pass
    if st.session_state.get(state_key) != request_key:
        return None
    return st.session_state.get(gen_key, 0)


_OUTAGE_STATES = {"unavailable", "unknown", "none", ""}


def _history_notes(frame, entities: tuple, notes) -> None:
    """Short note under Load about sensors without usable history; the details are in Setup.
    'Unavailable' (device offline, e.g. a flat battery) is told apart from 'not recorded'."""
    states = frame["state"].astype(str).str.strip().str.lower()
    usable = frame[~states.isin(_OUTAGE_STATES)]
    counts = usable["entity_id"].value_counts()
    seen = set(frame["entity_id"])
    unavailable = [e for e in entities if e in seen and counts.get(e, 0) == 0]
    sparse = [e for e in entities if e not in unavailable and counts.get(e, 0) <= 1]
    problems = len(unavailable) + len(sparse)
    if not problems:
        return
    notes.caption(f"⚠️ {problems} sensor{'s' if problems != 1 else ''} with no data in this period: "
                  "see Data Quality.")
    if unavailable:
        add_note("warning", "Unavailable in Home Assistant for the whole period (device offline or battery "
                 "flat?): " + ", ".join(f"`{e}`" for e in unavailable))
    if sparse:
        add_note("warning", "Almost no history returned for: " + ", ".join(f"`{e}`" for e in sparse)
                 + ". Check that these are recorded (recorder include/exclude) and that the period is "
                 "within the recorder's retention.")


def _clear_cache_control(source: str, root) -> None:
    """Escape hatch for back-filled or corrected source data: delete this source's saved months/days."""
    import shutil

    st.caption(
        "Completed months (InfluxDB) and days (Home Assistant) are saved so they are fetched only once. "
        "If data for a past period was corrected or back-filled at the source, clear the saved copy "
        "and load again."
    )
    if st.button(f"Clear saved {source} data", key=f"{source}_clear_cache"):
        shutil.rmtree(root, ignore_errors=True)
        for key in ("influx_frames", "ha_frames", f"{source}_request"):
            st.session_state.pop(key, None)
        st.success(f"Saved {source} data cleared. Press Load to fetch it again.")


# ----------------------------------------------------------------------
# Sensor notes (Data Quality) and the saved copy of source data (Setup)
# ----------------------------------------------------------------------
NOTES_KEY = "source_notes"


def reset_notes() -> None:
    st.session_state[NOTES_KEY] = []


def add_note(level: str, text: str) -> None:
    """A message about the loaded sensors, shown on the Data Quality view (level: warning / info)."""
    st.session_state.setdefault(NOTES_KEY, []).append((level, text))


def render_notes() -> None:
    """Data Quality: mapped sensors the source doesn't have, or without usable history, in the last load."""
    for level, text in st.session_state.get(NOTES_KEY) or []:
        getattr(st, level)(text)


def render_saved_data_control(source: str, user_config: Optional[Dict[str, Any]]) -> None:
    """Setup: clear the saved copy of the source's data (after a correction or back-fill at the source)."""
    tz = (user_config or {}).get("timezone") or "Europe/Dublin"
    if source == SOURCE_HA and (settings := ha_settings()) is not None:
        _clear_cache_control(SOURCE_HA, _ha_cache_root(settings, tz))
    elif source == SOURCE_INFLUX and (settings := influx_settings()) is not None:
        _clear_cache_control(SOURCE_INFLUX, _influx_cache_root(_identity(settings), tz))


# ----------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------
def render_influx_source(user_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    InfluxDB controls. Returns None until loaded, then
    {"kind": "influx", "numeric_dfs", "state_dfs", "config", "key", "label"}.
    """
    available = available_entities()
    if available is None:
        return None
    settings = influx_settings()
    identity = _identity(settings)

    req = _request_controls(SOURCE_INFLUX, user_config, available,
                            "Local dates, inclusive. Longer periods take longer to fetch and process.")
    if req is None:
        return None
    entities, start, end, last, tz, config, slot, notes = req
    request_key = (identity, entities, start.isoformat(), end.isoformat(), tz)
    generation = _loaded(SOURCE_INFLUX, request_key, entities, slot, start, last)
    if generation is None:
        return None

    memo = st.session_state.get("influx_frames")
    if memo and memo[0] == (request_key, generation):
        numeric_dfs, state_dfs, fingerprint = memo[1]
    else:
        try:
            with notes:  # progress bar in the load bar's status column
                numeric_dfs, state_dfs = _fetch_influx_cached(
                    settings, identity, entities, start, end, tz,
                    schema_defs.state_entities(config.get("mapping")),
                )
        except Exception as e:
            st.error(
                f"InfluxDB query failed: {e}. Months already fetched are saved; "
                "press Load again to continue from where it stopped."
            )
            return None
        # Hashing every value takes a moment on long periods: once per load, not per rerun.
        fingerprint = _fingerprint(numeric_dfs + state_dfs, "Time")
        # One loaded period per session; widget reruns reuse it without touching disk,
        # and the next Load press (a new generation) fetches the latest data.
        st.session_state["influx_frames"] = ((request_key, generation), (numeric_dfs, state_dfs, fingerprint))
    if not numeric_dfs and not state_dfs:
        st.warning("InfluxDB returned no data for these sensors and dates.")
        return None
    return {
        "kind": "influx",
        "numeric_dfs": numeric_dfs,
        "state_dfs": state_dfs,
        "config": config,
        "key": request_key,
        "fingerprint": fingerprint,
        "label": f"InfluxDB {start:%d %b %Y} – {last:%d %b %Y}",
        "source": SOURCE_INFLUX,
        "period": (start.isoformat(), last.isoformat()),
    }


def render_ha_source(user_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Home Assistant controls. Returns None until loaded, then
    {"kind": "ha", "frame", "config", "key", "label"} where frame is long-form
    history for ha_loader.process_ha_frame.
    """
    available = ha_available_entities()
    if available is None:
        return None
    settings = ha_settings()
    identity = _ha_identity(settings)

    req = _request_controls(
        SOURCE_HA, user_config, available,
        "Local dates, inclusive. Home Assistant keeps full history only for its recorder "
        "retention (10 days by default); therm keeps the days it has loaded, and InfluxDB "
        "covers longer periods.",
    )
    if req is None:
        return None
    entities, start, end, last, tz, config, slot, notes = req
    request_key = (identity, entities, start.isoformat(), end.isoformat(), tz)
    generation = _loaded(SOURCE_HA, request_key, entities, slot, start, last)
    if generation is None:
        return None

    memo = st.session_state.get("ha_frames")
    if memo and memo[0] == (request_key, generation):
        frame, fingerprint = memo[1]
    else:
        # In the load bar's status column: inserting it above the page left a faded
        # copy of the load bar below it while fetching.
        with notes, st.spinner(f"Fetching {len(entities)} sensors from Home Assistant…"):
            try:
                frame = _cached_ha_history(identity, entities, start, end, tz, settings)
            except Exception as e:
                st.error(f"Home Assistant history request failed: {e}")
                return None
        fingerprint = _fingerprint([frame], "last_changed") if frame is not None and not frame.empty else None
        # Reruns reuse this load; the next Load press fetches the latest data.
        st.session_state["ha_frames"] = ((request_key, generation), (frame, fingerprint))
    if frame is None or frame.empty:
        st.warning("Home Assistant returned no history for these sensors and dates.")
        return None
    _history_notes(frame, entities, notes)
    return {
        "kind": "ha",
        "frame": frame,
        "config": config,
        "key": request_key,
        "fingerprint": fingerprint,
        "source": SOURCE_HA,
        "period": (start.isoformat(), last.isoformat()),
        "label": f"Home Assistant {start:%d %b %Y} – {last:%d %b %Y}",
    }


def render_source(source: str, user_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return render_ha_source(user_config) if source == SOURCE_HA else render_influx_source(user_config)
