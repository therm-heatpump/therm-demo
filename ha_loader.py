# ha_loader.py
# Home Assistant → Engine-ready dataframe loader for THERM
#
# Key behaviours:
#   - Loads HA long-form CSV (entity_id, state, last_changed/last_updated/time)
#   - Infers dtype per entity (binary / numeric / string)
#   - Converts to wide, 1-minute resampled dataframe
#   - Interprets Samsung Modbus state/mode sensors into synthetic columns
#       * DHW_Mode_Label, DHW_Mode_Is_Active
#       * DHW_Status_Is_On
#       * Defrost_Is_Active
#       * Immersion_Is_On
#       * Valve_Is_DHW, Valve_Is_Heating, Valve_Position_Label
#   - NEW: Interpretation can use either:
#       * Raw Modbus entities (default), or
#       * Optional "HA mapped" entities (template / interpreted sensors),
#         controlled via user_config["state_mode_sources"].
#   - Applies user mapping for core engine roles (Power, FlowTemp, etc.)
#   - Runs engine gatekeepers, run detection, and daily stats.
#
# All raw columns from HA remain untouched. Synthetic columns are added
# after pivot/resample and BEFORE physics, and are not renamed by mapping.


import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Callable, Optional

from inspector import is_binary_sensor, safe_smart_parse
from config import resolve_timezone
from baselines import (
    PROVENANCE_HELD,
    add_energy_provenance,
    add_heartbeat_daily_counts,
    analyze_sensor_reporting_patterns,
    canonical_event_history,
    provenance_from_events,
    summarize_provenance,
)
import processing


# ----------------------------------------------------------------------
# CONSTANTS: default raw Modbus entity IDs for state/mode roles
# ----------------------------------------------------------------------

DEFAULT_RAW_ENTITIES = {
    "DHW_Mode": "sensor.heat_pump_hot_water_mode",
    "DHW_Status": "sensor.heat_pump_hot_water_status",
    "Defrost": "sensor.heat_pump_defrost_status",
    "Immersion": "sensor.heat_pump_immersion_heater_mode",
    "Valve": "sensor.heat_pump_3way_valve_position",
}


# ----------------------------------------------------------------------
# HELPERS: dtype inference and value conversion
# ----------------------------------------------------------------------

def _infer_dtype(series: pd.Series) -> str:
    """
    Infer dtype for a single sensor based on HA long-form 'state' series.
    Returns: "binary", "numeric", or "string".

    Improvements for Modbus / HA:
        - Classic HA binary detection via is_binary_sensor().
        - Numeric series that contain ONLY {0, 1} are treated as binary.
    """
    values = series.dropna().astype(str)

    # 1. Classic HA binary semantics (on/off, true/false, etc.)
    if is_binary_sensor(values):
        return "binary"

    # 2. Mostly numeric? (including Modbus integers, floats)
    parsed, mostly_numeric = safe_smart_parse(values)
    if mostly_numeric:
        try:
            num = pd.to_numeric(parsed, errors="coerce").dropna()
            uniq = set(num.unique().tolist())
            # Treat pure 0/1 numeric registers as binary
            if uniq and uniq.issubset({0, 1}):
                return "binary"
        except Exception:
            return "numeric"
        return "numeric"

    # 3. Fallback: treat as string
    return "string"


def _convert_value(val, dtype: str):
    """
    Convert HA 'state' values to proper Python values based on inferred dtype.
    """
    if dtype == "binary":
        s = str(val).strip().lower()
        return 1 if s in ("on", "true", "1", "yes") else 0

    elif dtype == "numeric":
        try:
            return float(val)
        except Exception:
            return np.nan

    else:
        # string / categorical
        return val


# ----------------------------------------------------------------------
# MODBUS INTERPRETATION (dual-source: raw or HA-mapped)
# ----------------------------------------------------------------------

def _pick_state_source(
    df: pd.DataFrame,
    user_config: Dict[str, Any],
    role_key: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Select which column to use for a given logical role (DHW_Mode, DHW_Status, etc.).

    Priority:
        1) If state_mode_sources[role_key]["source"] == "mapped" and mapped_entity_id
           exists in df.columns → use mapped.
        2) Else if raw_entity_id exists in df.columns → use raw.
        3) Else → (None, None) meaning "no data available for this role".

    user_config["state_mode_sources"] is optional. If absent or incomplete,
    this function defaults to raw Modbus entity IDs from DEFAULT_RAW_ENTITIES.
    """
    sources_cfg = user_config.get("state_mode_sources", {})
    cfg = sources_cfg.get(role_key, {})

    default_raw = DEFAULT_RAW_ENTITIES.get(role_key)
    source_mode = cfg.get("source", "raw")  # "raw" or "mapped"
    raw_id = cfg.get("raw_entity_id", default_raw)
    mapped_id = cfg.get("mapped_entity_id", None)

    # Prefer mapped if explicitly requested and present in df
    if source_mode == "mapped" and mapped_id and mapped_id in df.columns:
        return "mapped", mapped_id

    # Otherwise fall back to raw if present
    if raw_id and raw_id in df.columns:
        return "raw", raw_id

    # Nothing available
    return None, None


def enrich_modbus_interpretation(df: pd.DataFrame, user_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Insert interpreted columns for Samsung Modbus state/mode sensors (or their
    HA-mapped equivalents) INTO THE WIDE DATAFRAME, AFTER pivot/resample.

    Logical roles and their synthetic outputs:

        Role "DHW_Mode":
            - DHW_Mode_Label
            - DHW_Mode_Is_Active

        Role "DHW_Status":
            - DHW_Status_Is_On
            (DHW_Status_Raw is implicitly the underlying column if raw used)

        Role "Defrost":
            - Defrost_Is_Active

        Role "Immersion":
            - Immersion_Is_On

        Role "Valve":
            - Valve_Is_DHW
            - Valve_Is_Heating
            - Valve_Position_Label

    Behaviour:
        - If source == "raw": interpret numeric Modbus codes.
        - If source == "mapped": trust the mapped sensor as already-interpreted
          and derive booleans/labels with conservative logic.
        - Raw HA columns remain untouched.
    """

    df = df.copy()

    # ------------------------------------------------------------
    # 1. DHW Mode (logical role "DHW_Mode")
    # ------------------------------------------------------------
    mode_src, mode_col = _pick_state_source(df, user_config, "DHW_Mode")
    if mode_src and mode_col in df.columns:
        s = df[mode_col]

        if mode_src == "raw":
            # Raw Modbus register (73): 0 Eco, 1 Standard, 2 Power, 3 Force (on some units)
            mode_map = {
                0: "Eco",
                1: "Standard",
                2: "Power",
                3: "Force",
            }
            df["DHW_Mode_Label"] = s.map(mode_map).fillna("Unknown")
            df["DHW_Mode_Is_Active"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(v != 0)
            )
        else:
            # Mapped: assume s is already label/enum-like
            # We still provide a boolean "is active" guard.
            df["DHW_Mode_Label"] = s.astype("string")
            df["DHW_Mode_Is_Active"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(str(v).strip().lower() not in ("0", "off", "idle", "eco-off"))
            )

    # ------------------------------------------------------------
    # 2. DHW Status (logical role "DHW_Status")
    # ------------------------------------------------------------
    status_src, status_col = _pick_state_source(df, user_config, "DHW_Status")
    if status_src and status_col in df.columns:
        s = df[status_col]

        if status_src == "raw":
            # Raw Modbus register (72): 0, 1, 360 etc.
            # Treat > 0 as "on".
            df["DHW_Status_Is_On"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(v > 0)
            )
            # Raw values remain in the original column; if you want an
            # explicit alias for debugging, you can add it here:
            # df["DHW_Status_Raw"] = s
        else:
            # Mapped: expect boolean/int-like or "on"/"off"
            def _mapped_dhw_on(val):
                if pd.isna(val):
                    return 0
                sval = str(val).strip().lower()
                if sval in ("on", "true", "1", "yes", "active", "heating"):
                    return 1
                try:
                    return 1 if float(sval) > 0 else 0
                except Exception:
                    return 0

            df["DHW_Status_Is_On"] = s.apply(_mapped_dhw_on)

    # ------------------------------------------------------------
    # 3. Defrost Status (logical role "Defrost")
    # ------------------------------------------------------------
    defrost_src, defrost_col = _pick_state_source(df, user_config, "Defrost")
    if defrost_src and defrost_col in df.columns:
        s = df[defrost_col]

        if defrost_src == "raw":
            # Raw defrost status register (2): 0 idle, non-zero various active / transitions.
            df["Defrost_Is_Active"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(v != 0)
            )
        else:
            # Mapped: assume boolean-ish values (on/off, true/false, etc.)
            def _mapped_defrost(val):
                if pd.isna(val):
                    return 0
                sval = str(val).strip().lower()
                if sval in ("on", "true", "1", "yes", "defrost", "active"):
                    return 1
                try:
                    return 1 if float(sval) > 0 else 0
                except Exception:
                    return 0

            df["Defrost_Is_Active"] = s.apply(_mapped_defrost)

    # ------------------------------------------------------------
    # 4. Immersion Heater Mode (logical role "Immersion")
    # ------------------------------------------------------------
    imm_src, imm_col = _pick_state_source(df, user_config, "Immersion")
    if imm_src and imm_col in df.columns:
        s = df[imm_col]

        if imm_src == "raw":
            # Raw Modbus (85): 0 off, 1 on.
            df["Immersion_Is_On"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(v == 1)
            )
        else:
            # Mapped: boolean-ish
            def _mapped_imm(val):
                if pd.isna(val):
                    return 0
                sval = str(val).strip().lower()
                if sval in ("on", "true", "1", "yes", "heating"):
                    return 1
                try:
                    return 1 if float(sval) > 0 else 0
                except Exception:
                    return 0

            df["Immersion_Is_On"] = s.apply(_mapped_imm)

    # ------------------------------------------------------------
    # 5. 3-way Valve Position (logical role "Valve")
    # ------------------------------------------------------------
    valve_src, valve_col = _pick_state_source(df, user_config, "Valve")
    if valve_src and valve_col in df.columns:
        s = df[valve_col]

        if valve_src == "raw":
            # Raw Modbus (89): 0 = heating, 1 = DHW.
            df["Valve_Is_DHW"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(v == 1)
            )
            df["Valve_Is_Heating"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(v == 0)
            )

            def _raw_valve_label(v):
                if pd.isna(v):
                    return None
                if v == 0:
                    return "Heating"
                if v == 1:
                    return "DHW"
                return f"Unknown({int(v)})"

            df["Valve_Position_Label"] = s.apply(_raw_valve_label)

        else:
            # Mapped: we try both numeric and string possibilities.
            def _mapped_valve_flags(val):
                if pd.isna(val):
                    return 0, 0, None
                sval = str(val).strip().lower()
                # Common label-style values
                if sval in ("heating", "space", "rad", "rads", "space_heat"):
                    return 0, 1, "Heating"
                if sval in ("dhw", "hot water", "tank", "cylinder"):
                    return 1, 0, "DHW"
                # Numeric-ish values
                try:
                    fv = float(sval)
                    if fv == 0:
                        return 0, 1, "Heating"
                    if fv == 1:
                        return 1, 0, "DHW"
                    return 0, 0, f"Unknown({fv:g})"
                except Exception:
                    return 0, 0, f"Unknown({sval})"

            mapped = s.apply(_mapped_valve_flags)
            df["Valve_Is_DHW"] = mapped.apply(lambda x: x[0])
            df["Valve_Is_Heating"] = mapped.apply(lambda x: x[1])
            df["Valve_Position_Label"] = mapped.apply(lambda x: x[2])

    return df


# ----------------------------------------------------------------------
# SAMPLING QUALITY DETECTION
# ----------------------------------------------------------------------

def _infer_sampling_meta(
    grouped: pd.DataFrame,
    ts_col: str,
    user_config: Dict[str, Any],
    full_res_from: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Infer whether the HA feed is sparse/hourly so we can avoid per-run logic.

    Heuristic:
      - Look at the mapped Power entity in the pre-resample grouped data.
      - Ignore rows before the detected full-resolution retention boundary.
      - A day qualifies for run analysis when it has at least 300 Power events,
        at least two gaps of five minutes or less, or at least two events away
        from exact-hour boundaries. The last two conditions admit quiet
        change-driven sensors and partial days without admitting exact-hour
        long-term statistics.
      - Split qualifying days into local-calendar contiguous blocks and select
        the most recent block for run analysis.
    """
    mapping = user_config.get("mapping", {}) if isinstance(user_config, dict) else {}
    power_entity = mapping.get("Power")

    meta = {
        "power_entity": power_entity,
        "median_sample_seconds": None,
        "coverage_ratio": None,
        "low_res_days_ratio": None,
        "low_res_day_count": None,
        "total_day_count": None,
        "high_res_day_count": None,
        "high_res_threshold_per_day": 300,
        "high_res_short_gap_seconds": 5 * 60,
        "high_res_min_short_gaps": 2,
        "high_res_min_off_hour_events": 2,
        "high_res_block_count": 0,
        "high_res_blocks": [],
        "high_res_selection": "most_recent_contiguous",
        "high_res_run_window_start": None,
        "high_res_run_window_end": None,
        "high_res_run_window_rows": None,
        "is_sparse_power_sampling": False,
    }

    if not power_entity or grouped.empty:
        return meta

    power_rows = grouped[grouped["entity_id"] == power_entity]
    if power_rows.empty:
        return meta

    ts = pd.to_datetime(power_rows[ts_col], errors="coerce").dropna().sort_values()
    if full_res_from is not None and not ts.empty:
        boundary = pd.Timestamp(full_res_from)
        series_tz = ts.dt.tz
        if series_tz is not None:
            boundary = (
                boundary.tz_localize(series_tz)
                if boundary.tzinfo is None
                else boundary.tz_convert(series_tz)
            )
        elif boundary.tzinfo is not None:
            boundary = boundary.tz_localize(None)
        ts = ts[ts >= boundary]

    if len(ts) < 2:
        meta["total_day_count"] = int(ts.dt.floor("D").nunique()) if not ts.empty else 0
        meta["high_res_day_count"] = 0
        meta["low_res_day_count"] = meta["total_day_count"]
        meta["low_res_days_ratio"] = 1.0 if meta["total_day_count"] else None
        meta["is_sparse_power_sampling"] = True
        return meta

    deltas = ts.diff().dropna().dt.total_seconds()
    if deltas.empty:
        return meta

    median_secs = float(deltas.median())
    span_secs = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    expected_minutes = max(1.0, span_secs / 60.0)
    coverage_ratio = float(len(ts) / expected_minutes)

    day_labels = ts.dt.floor("D")
    per_day_counts = day_labels.value_counts().sort_index()
    high_res_threshold = meta["high_res_threshold_per_day"]
    short_gap_seconds = meta["high_res_short_gap_seconds"]
    min_short_gaps = meta["high_res_min_short_gaps"]
    min_off_hour_events = meta["high_res_min_off_hour_events"]

    # HA history is change-driven, so event count alone is not a cadence. A
    # short burst proves that the source can report at run-analysis resolution,
    # even when an otherwise quiet day has far fewer than 300 events.
    short_gaps_by_day: Dict[pd.Timestamp, int] = {}
    off_hour_events_by_day: Dict[pd.Timestamp, int] = {}
    for day, day_ts in ts.groupby(day_labels):
        day_ts = day_ts.sort_values()
        gaps = day_ts.diff().dropna().dt.total_seconds()
        short_gaps_by_day[day] = int(((gaps > 0) & (gaps <= short_gap_seconds)).sum())
        # Inspect wall-clock components rather than flooring to the hour;
        # floor("h") is ambiguous during the repeated autumn DST hour.
        off_hour = (
            (day_ts.dt.minute != 0)
            | (day_ts.dt.second != 0)
            | (day_ts.dt.microsecond != 0)
            | (day_ts.dt.nanosecond != 0)
        )
        off_hour_events_by_day[day] = int(off_hour.sum())

    high_res_days = [
        day for day, count in per_day_counts.items()
        if (
            count >= high_res_threshold
            or short_gaps_by_day.get(day, 0) >= min_short_gaps
            or off_hour_events_by_day.get(day, 0) >= min_off_hour_events
        )
    ]
    low_res_days = len(per_day_counts) - len(high_res_days)
    total_days = len(per_day_counts)
    low_res_ratio = float(low_res_days / total_days) if total_days else None
    meta["high_res_day_count"] = len(high_res_days)

    # Split on missing/non-qualifying local calendar days. Calendar dates are
    # used instead of elapsed 24-hour intervals so DST days remain contiguous.
    blocks: List[List[pd.Timestamp]] = []
    for day in sorted(high_res_days):
        if not blocks or (day.date() - blocks[-1][-1].date()).days != 1:
            blocks.append([day])
        else:
            blocks[-1].append(day)

    meta["high_res_block_count"] = len(blocks)
    meta["high_res_blocks"] = [
        {
            "start": block[0].isoformat(),
            "end": (block[-1] + pd.DateOffset(days=1) - pd.Timedelta(minutes=1)).isoformat(),
            "day_count": len(block),
        }
        for block in blocks
    ]

    # HA retention keeps the most recent history at full resolution. When an
    # outage splits valid data, use the newest block rather than silently
    # stopping at the first chronological gap.
    run_start_day = None
    run_end_day = None
    if blocks:
        selected = blocks[-1]
        run_start_day = selected[0]
        run_end_day = selected[-1]

    # No day demonstrated run-analysis resolution.
    is_sparse = run_start_day is None

    meta["median_sample_seconds"] = median_secs
    meta["coverage_ratio"] = coverage_ratio
    meta["low_res_days_ratio"] = low_res_ratio
    meta["low_res_day_count"] = int(low_res_days)
    meta["total_day_count"] = int(total_days)
    if run_start_day is not None:
        meta["high_res_run_window_start"] = run_start_day.isoformat()
        # Include the full final local day, including a 23 h / 25 h DST day.
        run_end_ts = run_end_day + pd.DateOffset(days=1) - pd.Timedelta(minutes=1)
        meta["high_res_run_window_end"] = run_end_ts.isoformat()
    meta["is_sparse_power_sampling"] = is_sparse

    return meta


# ----------------------------------------------------------------------
# RETENTION BOUNDARY (hourly long-term statistics at the start of an export)
# ----------------------------------------------------------------------

# Roles whose state is needed to classify / attribute runs. Run detection waits
# until these have reported (at most HA_STATE_WAIT_HOURS after power is full-res).
_CLASSIFICATION_ROLES = [
    "ValveMode", "DHW_Active", "DHW_Mode", "Zone_1", "Zone_2", "Zone_3", "Zone_4",
    "Defrost", "Immersion_Mode",
]
HA_STATE_WAIT_HOURS = 24


def _relabel_hourly_statistics(long: pd.DataFrame, ts_col: str):
    """
    HA keeps full history only for its retention period (e.g. 10 days). Older
    numeric history survives only as hourly long-term statistics, which appear
    in an export as rows exactly on the hour at the START of each entity's
    series. Each is the mean of the hour ENDING at its timestamp (verified
    against Grafana), so they are moved back one hour to label the hour they
    describe. They are kept (daily energy stays right) but runs must not be
    detected from them.

    Detection (per entity, timestamps in UTC): a leading run of >= 2 rows that
    fall exactly on the hour and are spaced exactly 1 h apart.

    Returns (long, first_full_res {entity: Timestamp}, n_relabelled).
    """
    ts = long[ts_col]
    exact = (ts == ts.dt.floor("h"))
    shift = pd.Series(False, index=long.index)
    first_full_res: Dict[str, pd.Timestamp] = {}
    for eid, grp in long.groupby("entity_id", sort=False):
        g = grp.sort_values(ts_col)
        # HA keeps long-term statistics only for numeric measurements. Skip
        # states/labels and binary 0/1 registers so a state that happens to
        # change exactly on the hour is never shifted.
        num = pd.to_numeric(g["state"], errors="coerce")
        numeric_values = set(num.dropna().unique())
        is_binary_register = bool(numeric_values) and numeric_values.issubset({0.0, 1.0})
        if num.notna().mean() < 0.9 or is_binary_register:
            first_full_res[eid] = g[ts_col].iloc[0]
            continue
        ex = exact.loc[g.index].to_numpy()
        n = 0
        while n < len(g) and ex[n]:
            n += 1
        if n >= 2:
            gaps = g[ts_col].iloc[:n].diff().dt.total_seconds().iloc[1:]
            if (gaps.sub(3600).abs() <= 1).all():
                shift.loc[g.index[:n]] = True
                first_full_res[eid] = g[ts_col].iloc[n] if n < len(g) else g[ts_col].iloc[-1] + pd.Timedelta(hours=1)
                continue
        first_full_res[eid] = g[ts_col].iloc[0]
    if shift.any():
        long = long.copy()
        long.loc[shift, ts_col] = long.loc[shift, ts_col] - pd.Timedelta(hours=1)
    return long, first_full_res, int(shift.sum())


def _complete_from(long, ts_col, mapping, first_full_res):
    """
    Earliest time run detection can be trusted: power at full resolution AND the
    mapped classification states (valve, DHW status/mode, zones, defrost,
    immersion) have reported. States with no row within HA_STATE_WAIT_HOURS of
    that are treated as unavailable rather than delaying the window further.
    """
    power_ent = (mapping or {}).get("Power")
    if not power_ent or power_ent not in first_full_res:
        return None
    t0 = first_full_res[power_ent]
    ents = {mapping[r] for r in _CLASSIFICATION_ROLES if mapping.get(r)} | set(DEFAULT_RAW_ENTITIES.values())
    firsts = long[long["entity_id"].isin(ents)].groupby("entity_id")[ts_col].min()
    firsts = firsts[firsts <= t0 + pd.Timedelta(hours=HA_STATE_WAIT_HOURS)]
    return max([t0] + list(firsts))


# ----------------------------------------------------------------------
# RESAMPLING (change-driven history)
# ----------------------------------------------------------------------

# Home Assistant records a row only when a value CHANGES (numeric sensors too),
# so silence means "unchanged" and can last days for states (a zone left on, a
# valve on "Heating") or hours for numerics (flow rate sitting at 0 overnight).
# Outages are explicit: HA writes 'unavailable' / 'unknown' rows.
HA_NUMERIC_MAX_HOLD_MIN = 24 * 60   # safety cap if the recorder stopped silently
HA_OUTAGE_STATES = {"unavailable", "unknown", "none", "nan", ""}


def _resample_change_driven(
    wide: pd.DataFrame,
    long: pd.DataFrame,
    ts_col: str,
    dtype_map: Dict[str, str],
) -> pd.DataFrame:
    """
    1-minute grid where every value holds until its next recorded change.

    - binary / string entities: held indefinitely (states can be silent for days)
    - numeric entities: held up to HA_NUMERIC_MAX_HOLD_MIN
    - any entity: blank from an 'unavailable'/'unknown' row until its next
      real reading (explicit outage)
    """
    df = wide.resample("1min").last()

    state_cols = [c for c in df.columns if dtype_map.get(c) in ("binary", "string")]
    num_cols = [c for c in df.columns if c not in set(state_cols)]
    if state_cols:
        df[state_cols] = df[state_cols].ffill()
    if num_cols:
        df[num_cols] = df[num_cols].ffill(limit=HA_NUMERIC_MAX_HOLD_MIN)

    # Explicit outages: 1 = real reading, 0 = unavailable/unknown; hold the
    # latest status between rows and blank the minutes flagged 0.
    status = long[[ts_col, "entity_id", "state"]].copy()
    status["ok"] = (
        ~status["state"].astype(str).str.strip().str.lower().isin(HA_OUTAGE_STATES)
    ).astype(float)
    for eid, grp in status.groupby("entity_id", observed=True):
        if eid not in df.columns or grp["ok"].all():
            continue
        ok = (
            grp.set_index(ts_col)["ok"].sort_index()
            .groupby(level=0).last()          # several rows in one timestamp
            .resample("1min").last()
            .ffill()
            .reindex(df.index)
            .ffill()
        )
        df.loc[ok == 0, eid] = np.nan
    return df


# ----------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------

def process_ha_files(
    files: List[Any],
    user_config: Dict[str, Any],
    progress_cb: Optional[Callable[[str, float], None]] = None,
    heartbeat_baseline: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load Home Assistant long-form CSV(s), convert to wide 1-minute data,
    apply Modbus/HA-mapped interpretation, apply user mapping, run physics,
    detect runs, and compute daily stats.

    Returns dict with:
        df          → engine dataframe (post-physics)
        raw_history → wide dataframe after interpretation, before physics
        raw_events  → canonical long-form source events for heartbeats
        provenance  → uint8 observed/held/missing mask for source sensors
        runs        → list of runs
        daily       → daily energy table
        patterns    → detected reporting patterns
        baselines   → active reusable heartbeat baseline, if supplied
    """

    # ------------------------------------------------------------------
    # 1. LOAD CSVs
    # ------------------------------------------------------------------
    dfs = []
    t_start = time.time()
    total = len(files)

    for i, f in enumerate(files):
        if progress_cb:
            progress_cb(f"Reading HA CSV {i+1}/{total}", (i + 1) / total)
        t_read = time.time()
        f.seek(0)
        df = pd.read_csv(f)
        read_secs = time.time() - t_read
        try:
            import sys
            sys.stdout.write(f"[ha_loader] read_csv secs={read_secs:.3f} rows={len(df)}\n")
        except Exception:
            pass
        dfs.append(df)

    if not dfs:
        return None

    return process_ha_frame(
        pd.concat(dfs, ignore_index=True),
        user_config,
        progress_cb=progress_cb,
        heartbeat_baseline=heartbeat_baseline,
        t_start=t_start,
    )


def process_ha_frame(
    long: pd.DataFrame,
    user_config: Dict[str, Any],
    progress_cb: Optional[Callable[[str, float], None]] = None,
    heartbeat_baseline: Optional[Dict[str, Any]] = None,
    t_start: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Source-agnostic core of process_ha_files: takes HA history in long form
    (`entity_id`, `state`, and one of `last_changed` / `last_updated` / `time`,
    in UTC, as an HA history CSV export or the HA history API provides) and
    returns the same dict as process_ha_files.
    """
    if t_start is None:
        t_start = time.time()

    # ------------------------------------------------------------------
    # 2. VERIFY REQUIRED COLUMNS
    # ------------------------------------------------------------------
    # HA files usually have: entity_id, state, last_changed
    time_cols = [c for c in ("last_changed", "last_updated", "time") if c in long.columns]
    if "entity_id" not in long.columns or "state" not in long.columns or not time_cols:
        return None

    ts_col = time_cols[0]
    # Stage messages for the progress panel (one line for the whole wait).
    def _step(label: str, frac: float) -> None:
        if progress_cb:
            progress_cb(label, frac)

    _step("Reading sensor history", 0.1)
    # format="ISO8601": the HA history API mixes "…:00+00:00" (start-of-window
    # states) with "…:05.123456+00:00"; pandas' inferred format then turned every
    # later row into NaT and dropped it (one row per entity survived).
    long[ts_col] = pd.to_datetime(long[ts_col], errors="coerce", utc=True, format="ISO8601")
    long = long.dropna(subset=[ts_col])

    # Retention boundary: hourly statistics rows at the start (#22). Done in UTC,
    # where "exactly on the hour" is unambiguous.
    long, first_full_res, n_stat_rows = _relabel_hourly_statistics(long, ts_col)

    # Local time (#21): HA exports UTC. Convert to the installation's zone so
    # tariffs, day boundaries and run times are local wall-clock. The index stays
    # tz-aware, so clock-change days (23 h / 25 h) are handled correctly.
    tz_name = resolve_timezone(user_config.get("timezone") if isinstance(user_config, dict) else None)
    long[ts_col] = long[ts_col].dt.tz_convert(tz_name)
    first_full_res = {k: v.tz_convert(tz_name) for k, v in first_full_res.items()}
    complete_from = _complete_from(
        long, ts_col, user_config.get("mapping", {}) if isinstance(user_config, dict) else {}, first_full_res
    )

    # ------------------------------------------------------------------
    # 3. INFER DTYPES PER ENTITY
    # ------------------------------------------------------------------
    _step("Interpreting sensor values", 0.25)
    dtype_map: Dict[str, str] = {}
    t_dtype = time.time()
    for eid, grp in long.groupby("entity_id"):
        dtype_map[eid] = _infer_dtype(grp["state"])
    dtype_secs = time.time() - t_dtype

    # ------------------------------------------------------------------
    # 4. CONVERT VALUES
    # ------------------------------------------------------------------
    t_convert = time.time()
    long["value"] = [
        _convert_value(v, dtype_map.get(eid, "string"))
        for v, eid in zip(long["state"], long["entity_id"])
    ]
    convert_secs = time.time() - t_convert

    mapping = user_config.get("mapping", {}) if isinstance(user_config, dict) else {}
    source_kinds = {
        str(eid): (
            "state" if dtype in {"binary", "string"} else "numeric"
        )
        for eid, dtype in dtype_map.items()
    }
    raw_events = canonical_event_history(
        long,
        ts_col,
        mapping=mapping,
        value_col="value",
        state_col="state",
        source_kind=source_kinds,
    )
    mapped_sources = {
        str(v) for v in mapping.values() if v not in (None, "", "None")
    }
    # Raw Samsung registers and explicitly configured state-mode sources are
    # consumed by the interpretation layer even when they are not mapped to a
    # public THERM sensor role. Do not mislabel those inputs as dropped data.
    consumed_sources = set(mapped_sources)
    consumed_sources.update(str(v) for v in DEFAULT_RAW_ENTITIES.values())
    for source_spec in (user_config.get("state_mode_sources", {}) or {}).values():
        if isinstance(source_spec, dict):
            for key in ("raw_entity_id", "mapped_entity_id"):
                value = source_spec.get(key)
                if value not in (None, "", "None"):
                    consumed_sources.add(str(value))
    source_entities = set(raw_events["source_entity_id"].astype(str))
    unmapped_entities = sorted(source_entities - consumed_sources)

    # ------------------------------------------------------------------
    # 5. GROUP DUPLICATES  ONE VALUE PER (timestamp, entity_id)
    # ------------------------------------------------------------------
    # Optimised: split numeric vs string to avoid slow per-group apply().
    t_group = time.time()
    numeric_entities = {eid for eid, dt in dtype_map.items() if dt != 'string'}
    long['entity_id'] = long['entity_id'].astype('category')

    grouped_frames = []

    if numeric_entities:
        num_mask = long['entity_id'].isin(numeric_entities)
        num_df = long.loc[num_mask, [ts_col, 'entity_id', 'value']]
        grouped_num = (
            num_df.groupby([ts_col, 'entity_id'], sort=False, observed=True)['value']
            .mean()
        )
        grouped_frames.append(grouped_num)

    str_mask = ~long['entity_id'].isin(numeric_entities)
    if str_mask.any():
        str_df = long.loc[str_mask, [ts_col, 'entity_id', 'value']]
        grouped_str = (
            str_df.groupby([ts_col, 'entity_id'], sort=False, observed=True)['value']
            .last()
        )
        grouped_frames.append(grouped_str)

    if grouped_frames:
        grouped = pd.concat(grouped_frames).reset_index()
    else:
        grouped = pd.DataFrame(columns=[ts_col, 'entity_id', 'value'])
    group_secs = time.time() - t_group
    # 6. PIVOT TO WIDE
    # ------------------------------------------------------------------
    t_pivot = time.time()
    wide = grouped.pivot(index=ts_col, columns="entity_id", values="value")
    wide.index = pd.to_datetime(wide.index)
    pivot_secs = time.time() - t_pivot

    # ------------------------------------------------------------------
    # 6b. SAMPLING QUALITY CHECK (pre-resample)
    # ------------------------------------------------------------------
    power_entity = (
        user_config.get("mapping", {}).get("Power")
        if isinstance(user_config, dict)
        else None
    )
    sampling_meta = _infer_sampling_meta(
        grouped,
        ts_col,
        user_config,
        full_res_from=first_full_res.get(power_entity),
    )
    sampling_meta["timezone"] = tz_name
    sampling_meta["hourly_statistic_rows"] = n_stat_rows
    sampling_meta["complete_from"] = complete_from.isoformat() if complete_from is not None else None
    # Runs start no earlier than the point where the data is complete
    if complete_from is not None and sampling_meta.get("high_res_run_window_start"):
        ws = pd.Timestamp(sampling_meta["high_res_run_window_start"])
        if ws.tzinfo is None:
            ws = ws.tz_localize(tz_name)
        if complete_from > ws:
            sampling_meta["high_res_run_window_start"] = complete_from.isoformat()

    # ------------------------------------------------------------------
    # 7. RESAMPLE TO 1-MINUTE
    # ------------------------------------------------------------------
    _step("Building the 1-minute timeline", 0.45)
    t_resample = time.time()
    df_wide = _resample_change_driven(wide, long, ts_col, dtype_map)
    resample_secs = time.time() - t_resample

    # ------------------------------------------------------------------
    # 8. APPLY MODBUS / HA-MAPPED INTERPRETATION
    # ------------------------------------------------------------------
    df_wide = enrich_modbus_interpretation(df_wide, user_config)

    # ------------------------------------------------------------------
    # 9. APPLY USER SENSOR MAPPING (core engine roles)
    # ------------------------------------------------------------------
    # user_config["mapping"] maps THERM keys → entity_id strings
    mapping = user_config.get("mapping", {})

    # Reverse: entity_id → THERM key
    rename_dict: Dict[str, str] = {}
    for therm_key, entity_id in mapping.items():
        if entity_id in df_wide.columns:
            rename_dict[entity_id] = therm_key

    df = df_wide.rename(columns=rename_dict)

    # ------------------------------------------------------------------
    # 10. COMPUTE DeltaT IF MAPPED
    # ------------------------------------------------------------------
    if "FlowTemp" in df.columns and "ReturnTemp" in df.columns:
        df["DeltaT"] = df["FlowTemp"] - df["ReturnTemp"]

    # Ensure numeric types for engine core numeric columns
    numeric_cols = ["Power", "FlowTemp", "ReturnTemp", "FlowRate", "Freq", "DeltaT"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    provenance = provenance_from_events(df, raw_events, filled_code=PROVENANCE_HELD)
    raw_timestamps = {
        str(sensor): pd.DatetimeIndex(events["last_changed"].drop_duplicates())
        for sensor, events in raw_events[raw_events["is_available"]].groupby(
            "entity_id", observed=True
        )
    }
    patterns = analyze_sensor_reporting_patterns(
        df,
        baselines=heartbeat_baseline,
        raw_timestamps=raw_timestamps,
    )

    # ------------------------------------------------------------------
    # 11. RUN PHYSICS ENGINE
    # ------------------------------------------------------------------
    _step("Calculating heat output and COP", 0.65)
    t_gate = time.time()
    df_engine = processing.apply_gatekeepers(df, user_config)
    gate_secs = time.time() - t_gate
    if df_engine is None or df_engine.empty:
        return None
    # Minutes before the data is complete (hourly statistics / states not yet
    # reported): kept for daily energy, excluded from run analysis.
    if complete_from is not None:
        df_engine["Before_Complete"] = df_engine.index < complete_from

    # ------------------------------------------------------------------
    # 12. RUN DETECTOR + DAILY STATS
    # ------------------------------------------------------------------
    _step("Finding heating and hot-water runs", 0.8)
    t_runs = time.time()
    run_detection_disabled = False
    runs = []

    if sampling_meta.get("is_sparse_power_sampling"):
        run_detection_disabled = True
        try:
            import sys
            med = sampling_meta.get("median_sample_seconds")
            cov = sampling_meta.get("coverage_ratio")
            low = sampling_meta.get("low_res_day_count")
            tot = sampling_meta.get("total_day_count")
            med_str = f"{med:.0f}s" if med is not None else "unknown"
            cov_str = f"{cov:.3f}" if cov is not None else "unknown"
            low_str = f"{low}/{tot}" if low is not None and tot is not None else "unknown"
            sys.stdout.write(
                "[ha_loader] skipping run detection: sparse/hourly sampling "
                f"(median={med_str} coverage={cov_str} low_res_days={low_str})\n"
            )
        except Exception:
            pass
    else:
        run_df = df_engine
        # If we have a high-res retention window, limit run detection to that period
        start_str = sampling_meta.get("high_res_run_window_start")
        end_str = sampling_meta.get("high_res_run_window_end")
        if start_str and end_str:
            start_ts = pd.to_datetime(start_str)
            end_ts = pd.to_datetime(end_str)
            try:
                tz = run_df.index.tz
                if tz is not None:
                    # Bring both ends into the index's zone: across a clock
                    # change their stored UTC offsets differ (+01:00 vs +00:00),
                    # and pandas can't slice between mixed fixed offsets.
                    start_ts = start_ts.tz_localize(tz) if start_ts.tzinfo is None else start_ts.tz_convert(tz)
                    end_ts = end_ts.tz_localize(tz) if end_ts.tzinfo is None else end_ts.tz_convert(tz)
            except Exception:
                pass
            run_df = run_df.loc[start_ts:end_ts]
            sampling_meta["high_res_run_window_rows"] = int(len(run_df))
        runs = processing.detect_runs(run_df, user_config) if not run_df.empty else []

    _step("Daily totals and data quality", 0.9)
    t_daily = time.time()
    daily = processing.get_daily_stats(df_engine)
    daily = processing.add_diagnostic_daily_metrics(
        daily, df_engine, runs, user_config,
        run_window_start=sampling_meta.get("high_res_run_window_start"),
        run_window_end=sampling_meta.get("high_res_run_window_end"),
    )
    daily = add_heartbeat_daily_counts(daily, raw_events)
    daily = add_energy_provenance(daily, df_engine, provenance, patterns)
    daily_secs = time.time() - t_daily
    runs_secs = t_daily - t_runs

    total_secs = time.time() - t_start
    try:
        import sys
        sys.stdout.write(
            "[ha_loader] timing "
            f"dtype={dtype_secs:.3f}s convert={convert_secs:.3f}s "
            f"group={group_secs:.3f}s pivot={pivot_secs:.3f}s resample={resample_secs:.3f}s "
            f"gate={gate_secs:.3f}s runs={runs_secs:.3f}s daily={daily_secs:.3f}s total={total_secs:.3f}s\n"
        )
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 13. OUTPUT STRUCTURE (same as Grafana loader)
    # ------------------------------------------------------------------
    return {
        "df": df_engine,
        "raw_history": df,   # wide dataframe after interpretation, before physics
        "raw_events": raw_events,
        "provenance": provenance,
        "provenance_summary": summarize_provenance(provenance),
        "unmapped_entities": unmapped_entities,
        "runs": runs,
        "daily": daily,
        "sampling_meta": sampling_meta,
        "run_detection_disabled": run_detection_disabled,
        "patterns": patterns,
        "baselines": heartbeat_baseline or {},
    }
