# data_loader.py
"""
Core CSV loader for THERM.

- Handles both numeric ("value") and state ("state") CSVs.
- Normalises time to a 'Time' index.
- Pivots to wide format (index=Time, columns=entity_id).
- Resamples to 1-minute cadence.
- Applies user-config mapping.
- Provides hooks for unified Modbus interpretation across HA + Grafana paths.

v2.9: Smart heartbeat-driven forward-fill
- Analyzes actual reporting patterns from uploaded data
- Per-sensor forward-fill limits based on learned behavior
- Binary state sensors (zones, valves): on-change reporting with 30+ min gaps
- Periodic sensors (temps, power): conservative 10-20 min limits
- Sparse sensors (weather): up to 120 min for hourly reporting
- Falls back to v2.8.1 two-tier approach on errors
"""

import logging
from typing import Any, Callable, Dict, Optional

import pandas as pd

_LOGGER = logging.getLogger("therm")


def _emit_message(
    on_message: Optional[Callable[[str, str], None]],
    level: str,
    msg: str,
) -> None:
    """Emit message to callback if provided, else log."""
    if on_message is not None:
        on_message(level, msg)
    else:
        if level == "error":
            _LOGGER.error(msg)
        elif level == "warning":
            _LOGGER.warning(msg)
        else:
            _LOGGER.info(msg)

# Smart forward-fill support
from baselines import (
    PROVENANCE_HELD,
    PROVENANCE_INTERPOLATED,
    PROVENANCE_MISSING,
    PROVENANCE_OBSERVED,
    analyze_sensor_reporting_patterns,
    canonical_event_history,
    smart_forward_fill,
    summarize_provenance,
)


# ----------------------------------------------------------------------
# TIME NORMALISATION
# ----------------------------------------------------------------------

# Accept a broader set of time column names, including Home Assistant history
# exports. This list is intentionally aligned with the main branch + HA exports.
_TIME_CANDIDATES = [
    "Time",
    "time",
    "timestamp",
    "Timestamp",
    "date",
    "datetime",
    "DateTime",
    "Date",
    # Home Assistant history export timestamp
    "last_changed",
]


def _normalise_time_column(
    temp: pd.DataFrame,
    filename: str,
    on_message: Optional[Callable[[str, str], None]] = None,
) -> Optional[pd.DataFrame]:
    """
    Ensure the dataframe has a valid 'Time' column in datetime format.

    - Looks for a set of common timestamp column names.
    - Parses to datetime with day-first semantics.
    - Drops rows with invalid timestamps.
    - Renames the chosen column to 'Time'.

    Returns:
        Cleaned dataframe, or None if no usable time column is found.
    """
    time_col: Optional[str] = None

    for cand in _TIME_CANDIDATES:
        if cand in temp.columns:
            time_col = cand
            break

    if time_col is None:
        _emit_message(
            on_message,
            "error",
            f"{filename} has no recognised time column "
            f"(expected one of: {_TIME_CANDIDATES}) – skipping this file.",
        )
        return None

    temp = temp.copy()
    temp[time_col] = pd.to_datetime(
        temp[time_col],
        dayfirst=True,
        errors="coerce",
    )
    temp = temp.dropna(subset=[time_col])

    if temp.empty:
        _emit_message(
            on_message,
            "error",
            f"{filename} has no valid timestamps after parsing – skipping this file.",
        )
        return None

    if time_col != "Time":
        temp = temp.rename(columns={time_col: "Time"})

    return temp


# ----------------------------------------------------------------------
# SOURCE DETECTION (HA vs Grafana/Influx)
# ----------------------------------------------------------------------

def detect_file_source(file_obj) -> str:
    """
    Heuristically detect whether a CSV is a Home Assistant history export
    or a Grafana/Influx export.

    Returns: "ha", "grafana", or "unknown"
    """
    try:
        pos = file_obj.tell()
    except Exception:
        pos = None

    try:
        df_head = pd.read_csv(file_obj, nrows=20)
        cols = set(df_head.columns.str.lower())

        has_entity = "entity_id" in cols
        has_state = "state" in cols
        has_value = "value" in cols
        has_last_changed = any(c in cols for c in ["last_changed", "last_updated"])
        has_time = any(c in cols for c in [c.lower() for c in _TIME_CANDIDATES])

        # Home Assistant history exports usually have state + last_changed + entity_id
        if has_entity and has_state and has_last_changed:
            return "ha"

        # Grafana/Influx exports typically have value + Time/Date + entity_id
        if has_entity and has_value and has_time:
            return "grafana"

        # If we see state without last_changed (some HA variants), still lean HA
        if has_entity and has_state and not has_value:
            return "ha"

    except Exception:
        return "unknown"
    finally:
        if pos is not None:
            try:
                file_obj.seek(pos)
            except Exception:
                pass

    return "unknown"


# ----------------------------------------------------------------------
# MODBUS INTERPRETATION HOOK (SKELETON)
# ----------------------------------------------------------------------

def apply_modbus_interpretation(
    df: pd.DataFrame,
    mapping: Optional[Dict[str, Any]] = None,
    source_hint: Optional[str] = None,
) -> pd.DataFrame:
    """
    Unified hook for interpreting raw Modbus registers into synthetic columns.

    This is intentionally a NO-OP skeleton for now – it is safe to call in
    existing workflows without changing behaviour.

    Intended responsibilities (to implement later, mirroring ha_loader.py):
        - Use raw Modbus integer registers (0/1, small integer enums, etc.).
        - Derive synthetic boolean / label columns such as:
            * DHW_Mode_Label, DHW_Mode_Is_Active
            * DHW_Status_Is_On
            * Defrost_Is_Active
            * Immersion_Is_On
            * Valve_Is_DHW, Valve_Is_Heating, Valve_Position_Label
        - Work for both:
            * Home Assistant history CSV exports (raw Modbus entities)
            * Grafana/Influx exports (same entities via Influx bridge)
        - Optionally use `mapping` and/or `source_hint` to:
            * Disambiguate where multiple registers are present
            * Support future dual-source architectures
              ("raw Modbus" primary, "HA template" fallback)

    Args:
        df: Wide dataframe indexed by Time, columns = entity_id or mapped names.
        mapping: Optional user mapping dictionary from config, if needed.
        source_hint: Optional free-text hint ("ha_csv", "grafana_csv", etc.).

    Returns:
        DataFrame with the same index/columns for now. Future versions will
        add synthetic columns but will avoid destructive changes.
    """
    # NOTE: This is currently a stub. Implementation will live here or will
    # call a separate shared module (e.g., modbus_interpreter.py) so that
    # HA and Grafana paths share identical decoding rules.
    return df


# ----------------------------------------------------------------------
# NUMERIC COERCION
# ----------------------------------------------------------------------

def _coerce_numeric_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """ Coerce sensor-like columns to numeric while preserving explicit text modes.
    Mirrors the main-branch invariant:
    - FlowRate, Power, temperatures, etc. should be numeric.
    - Only a small, explicit set of mode/label columns stay as text.
    """
    # Extend this set as you add more textual mode/label columns.
    text_cols = {
        # Core textual modes
        "ValveMode",
        "DHW_Mode",
        "DHW_Status",
        # Future examples (once synthetic Modbus columns are added):
        # "Valve_Position_Label",
        # "DHW_Mode_Label",
    }

    # Tokens we consider as binary-like states (lowercased & stripped)
    binary_token_map = {
        "on": 1,
        "off": 0,
        "true": 1,
        "false": 0,
        "running": 1,
        "not running": 0,
        "yes": 1,
        "no": 0,
        "open": 1,
        "closed": 0,
        "1": 1,
        "0": 0,
    }

    for col in df.columns:
        if col in text_cols:
            # Explicitly textual columns stay as-is
            continue

        series = df[col]

        # If it's already numeric, leave it alone
        if pd.api.types.is_numeric_dtype(series):
            continue

        # Try to detect binary-like columns (on/off, 0/1, running/not running, etc.)
        non_null = series.dropna()
        if not non_null.empty:
            norm = (
                non_null.astype(str)
                .str.strip()
                .str.lower()
            )
            unique_vals = set(norm.unique())

            # Only treat as binary if small cardinality and all values are in our token map
            if 0 < len(unique_vals) <= 6 and unique_vals.issubset(binary_token_map.keys()):
                df[col] = (
                    series.astype(str)
                    .str.strip()
                    .str.lower()
                    .map(binary_token_map)
                    .astype("float64")
                )
                continue

        # Fallback: numeric coercion as before
        try:
            df[col] = pd.to_numeric(series, errors="coerce")
        except Exception:
            # If conversion fails for some weird column, leave it as-is.
            # Downstream code should ignore non-numeric channels.
            pass

    return df


# ----------------------------------------------------------------------
# PUBLIC LOADER API
# ----------------------------------------------------------------------

def load_and_clean_data(
    files,
    user_config: Optional[Dict[str, Any]],
    progress_cb: Optional[Callable[[str, float], None]] = None,
    heartbeat_baseline: Optional[Dict[str, Any]] = None,
    on_message: Optional[Callable[[str, str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Robust data loader that handles both Numeric (Float) and State (Text) CSV files.

    Steps:
        1) Read each uploaded CSV and normalise the time column to 'Time'.
        2) Split into numeric ('value') vs state ('state') tables.
        3) Pivot each to wide format: index=Time, columns=entity_id.
        4) Resample to 1-minute resolution:
            - numeric: mean + short interpolation
            - state: forward-fill (state persists until changed)
        5) Merge numeric + state with an outer join on Time.
        6) Apply mapping (simple column rename from config).
        7) Apply unified Modbus interpretation hook (currently NO-OP).
        8) Coerce sensor-like columns to numeric, preserve explicit text columns.
        9) Fill small gaps with sensor-type-aware forward-fill limits and return the dataset.

    Returns:
        dict with keys:
            - "df": processed wide dataframe
            - "raw_history": wide pre-physics dataframe
            - "raw_events": canonical long-form source events for heartbeats
            - "provenance": uint8 observed/interpolated/held/missing mask
            - "unmapped_entities": raw source entities not present in mapping
            - "baselines": active reusable heartbeat baseline, if supplied
            - "patterns": learned reporting patterns
        or None if no usable data was found.
    """
    if not files:
        return None

    numeric_dfs, state_dfs = _read_grafana_csvs(files, progress_cb, on_message=on_message)
    return load_and_clean_frames(
        numeric_dfs,
        state_dfs,
        user_config,
        progress_cb=progress_cb,
        heartbeat_baseline=heartbeat_baseline,
        on_message=on_message,
    )


def _read_grafana_csvs(
    files,
    progress_cb: Optional[Callable[[str, float], None]] = None,
    on_message: Optional[Callable[[str, str], None]] = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Read Grafana/Influx CSV exports and split them into numeric and state
    tables, each with a parsed 'Time' column plus 'entity_id' and
    'value' / 'state'. Files with no usable time column are skipped.
    """
    numeric_dfs: list[pd.DataFrame] = []
    state_dfs: list[pd.DataFrame] = []

    # 1. Read and classify files
    for i, f in enumerate(files):
        try:
            # Read CSV
            if hasattr(f, "seek"):
                f.seek(0)
            temp = pd.read_csv(f)

            # Normalise / parse time column
            temp = _normalise_time_column(
                temp,
                getattr(f, "name", "uploaded file"),
                on_message=on_message,
            )
            if temp is None:
                # Skip files with no usable time column
                continue

            # Classify based on columns
            if "state" in temp.columns:
                state_dfs.append(temp)
            elif "value" in temp.columns:
                numeric_dfs.append(temp)

            if progress_cb:
                progress_cb(
                    f"Read {getattr(f, 'name', 'uploaded file')}",
                    (i / max(len(files), 1)) * 0.2,
                )
        except Exception as e:
            _emit_message(
                on_message,
                "error",
                f"Error reading {getattr(f, 'name', 'uploaded file')}: {e}",
            )

    return numeric_dfs, state_dfs


_EVENT_CATEGORY_COLUMNS = ("entity_id", "source_entity_id", "source_kind")


def _concat_event_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate canonical event frames keeping their categorical id columns.

    pd.concat turns categoricals with different category sets into object
    columns (7 M Python-object rows over ten months of InfluxDB data, ~1–2 GB
    at peak). Aligning the category sets first keeps the result categorical.
    """
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    for col in _EVENT_CATEGORY_COLUMNS:
        if all(col in p and isinstance(p[col].dtype, pd.CategoricalDtype) for p in parts):
            union = pd.Index(sorted(set().union(*(set(p[col].cat.categories) for p in parts))))
            for p in parts:
                p[col] = p[col].cat.set_categories(union)
    return pd.concat(parts, ignore_index=True)


def _pivot_numeric_minute_means(df_num: pd.DataFrame) -> pd.DataFrame:
    """
    Wide 1-minute means, one column per entity.

    Equivalent to
        df_num.groupby(["Time", "entity_id"])["value"].mean().unstack().resample("1min").mean()
    (duplicate timestamps averaged first, then minute means over the union of
    minutes from the first to the last reading of any entity), but computed one
    entity at a time. On ten months of InfluxDB data (7 M readings) the combined
    two-key groupby + unstack peaked at ~2.7 GB and was OOM-killed in the add-on.
    """
    columns: Dict[str, pd.Series] = {}
    for eid, sub in df_num.groupby("entity_id", sort=True, observed=True):
        columns[eid] = sub.groupby("Time")["value"].mean().resample("1min").mean()
    if not columns:
        return pd.DataFrame()

    start = min(s.index.min() for s in columns.values())
    end = max(s.index.max() for s in columns.values())
    index = pd.date_range(start, end, freq="1min", name="Time")
    wide = pd.DataFrame({eid: s.reindex(index) for eid, s in columns.items()}, index=index)
    wide.columns.name = "entity_id"
    return wide


def _raw_report_times(numeric_dfs: list, state_dfs: list,
                      user_config: Optional[Dict[str, Any]]) -> Dict[str, pd.DatetimeIndex]:
    """Each sensor's original report times (before resampling), keyed by mapped role."""
    try:
        rev = {}
        if user_config and isinstance(user_config.get("mapping"), dict):
            rev = {v: k for k, v in user_config["mapping"].items()}
        parts: Dict[str, list] = {}
        for src in list(numeric_dfs) + list(state_dfs):
            if src is None or src.empty:
                continue
            for eid, times in src.groupby("entity_id", observed=True)["Time"]:
                parts.setdefault(rev.get(eid, eid), []).append(times)
        return {k: pd.DatetimeIndex(pd.concat(v).drop_duplicates()) for k, v in parts.items()}
    except Exception:
        return {}  # fall back to learning from the resampled frame


def request_reporting_patterns(numeric_dfs: list, state_dfs: list, user_config: Optional[Dict[str, Any]],
                               heartbeat_baseline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    The reporting patterns (how long each sensor may be silent before its reading is no
    longer held) that a single pass over these frames would learn. Needs only the report
    times, not the minute frame, so a month-by-month analysis can compute them for the
    whole request and pass them to every month (load_and_clean_frames(reporting_patterns=)).
    """
    raw = _raw_report_times(numeric_dfs, state_dfs, user_config)
    if not raw:
        return {}
    shape = pd.DataFrame(index=[0], columns=sorted(raw))  # column names only; times come from `raw`
    return analyze_sensor_reporting_patterns(shape, baselines=heartbeat_baseline, raw_timestamps=raw)


def load_and_clean_frames(
    numeric_dfs: list[pd.DataFrame],
    state_dfs: list[pd.DataFrame],
    user_config: Optional[Dict[str, Any]],
    progress_cb: Optional[Callable[[str, float], None]] = None,
    heartbeat_baseline: Optional[Dict[str, Any]] = None,
    on_message: Optional[Callable[[str, str], None]] = None,
    index_span: Optional[tuple] = None,
    reporting_patterns: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Source-agnostic core of load_and_clean_data (steps 2-9).

    reporting_patterns: optional per-sensor patterns (request_reporting_patterns) to
    use instead of learning them from these frames. chunked.py passes the patterns of
    the whole request, so every month holds silent sensors exactly as a single pass does.

    index_span: optional (first, last) minute the numeric timeline must cover even
    without readings there. chunked.py passes a month window's edges (within the
    whole period's data) so an outage at a window edge keeps its minutes, as in a
    single pass over the whole period.

    Takes long-form tables in the Grafana/Influx shape: 'Time', 'entity_id'
    and 'value' (numeric tables) or 'state' (state tables, optionally with
    'value'). Any source that can produce these frames (CSV exports, a direct
    InfluxDB query) shares the same resampling, mapping and gap-filling.
    Returns the same dict as load_and_clean_data, or None if no usable data.
    """
    mapping = user_config.get("mapping", {}) if isinstance(user_config, dict) else {}

    # 2. Process Numeric Data (Resample: Mean)
    df_numeric_wide = pd.DataFrame()
    provenance_numeric = pd.DataFrame()
    raw_event_parts: list[pd.DataFrame] = []
    if numeric_dfs:
        if progress_cb:
            progress_cb("Processing numeric data...", 0.3)

        df_num = pd.concat(numeric_dfs, ignore_index=True)
        raw_event_parts.append(
            canonical_event_history(
                df_num,
                "Time",
                mapping=mapping,
                value_col="value",
                source_kind="numeric",
            )
        )

        # Pivot: Index=Time, Columns=entity_id, Values=value (1-minute means)
        numeric_observed = _pivot_numeric_minute_means(df_num)
        if index_span is not None and not numeric_observed.empty:
            first = min(numeric_observed.index[0], index_span[0])
            last = max(numeric_observed.index[-1], index_span[1])
            numeric_observed = numeric_observed.reindex(
                pd.date_range(first, last, freq="1min", name=numeric_observed.index.name))

        # Resample to 1 minute, interpolating missing values
        df_numeric_wide = numeric_observed.interpolate(limit=30)
        provenance_numeric = pd.DataFrame(
            PROVENANCE_MISSING,
            index=df_numeric_wide.index,
            columns=df_numeric_wide.columns,
            dtype="uint8",
        )
        provenance_numeric[numeric_observed.notna()] = PROVENANCE_OBSERVED
        provenance_numeric[df_numeric_wide.notna() & numeric_observed.isna()] = PROVENANCE_INTERPOLATED

    # 3. Process State Data (Resample: FFill)
    df_state_wide = pd.DataFrame()
    provenance_state = pd.DataFrame()
    if state_dfs:
        if progress_cb:
            progress_cb("Processing state data...", 0.5)

        df_state = pd.concat(state_dfs, ignore_index=True)

        # Grafana state exports carry both 'value' and 'state' columns. Most
        # entities fill 'state', but some (e.g. heat_pump_defrost_status) put
        # their reading in 'value' and leave 'state' blank. Use 'value' there,
        # otherwise those entities arrive entirely empty.
        if "value" in df_state.columns:
            df_state["state"] = df_state["state"].where(
                df_state["state"].notna(), df_state["value"]
            )
        raw_event_parts.append(
            canonical_event_history(
                df_state,
                "Time",
                mapping=mapping,
                value_col="state",
                state_col="state",
                source_kind="state",
            )
        )

        # Pivot: Index=Time, Columns=entity_id, Values=state
        # For state, we take the 'last' known state if duplicates exist
        state_observed = (
            df_state.groupby(["Time", "entity_id"])["state"]
            .last()
            .unstack()
            .resample("1min")
            .last()
        )

        # Resample to 1 minute, keeping the last event within each minute
        # (important when state changes land at hh:mm:ss, not exactly hh:mm)
        # Then forward-fill so the state persists until the next change.
        # pandas' future behaviour (no silent downcast of the object column), which the app
        # enables globally (app.py) - explicit here so tools and scripts give the same
        # result without pandas' FutureWarning.
        with pd.option_context("future.no_silent_downcasting", True):
            df_state_wide = state_observed.ffill()
        provenance_state = pd.DataFrame(
            PROVENANCE_MISSING,
            index=df_state_wide.index,
            columns=df_state_wide.columns,
            dtype="uint8",
        )
        provenance_state[state_observed.notna()] = PROVENANCE_OBSERVED
        provenance_state[df_state_wide.notna() & state_observed.isna()] = PROVENANCE_HELD

    # 4. Merge
    if progress_cb:
        progress_cb("Merging datasets...", 0.6)

    if df_numeric_wide.empty and df_state_wide.empty:
        return None
    elif df_numeric_wide.empty:
        combined_df = df_state_wide
    elif df_state_wide.empty:
        combined_df = df_numeric_wide
    else:
        # Outer join to align timestamps
        combined_df = df_numeric_wide.join(df_state_wide, how="outer")

    provenance = provenance_numeric.join(provenance_state, how="outer").fillna(
        PROVENANCE_MISSING
    ).astype("uint8")

    raw_events = _concat_event_parts(raw_event_parts)
    mapped_sources = {
        str(v) for v in mapping.values() if v not in (None, "", "None")
    }
    # unique() first: converting all ~7 M event rows to str just to build a set of
    # ~30 names was the largest single spike on long InfluxDB periods.
    source_entities = {
        str(v) for v in pd.unique(raw_events.get("source_entity_id", pd.Series(dtype=str)).dropna())
    }
    unmapped_entities = sorted(source_entities - mapped_sources)

    # 5. Apply Mapping (simple column rename)
    if progress_cb:
        progress_cb("Applying sensor mapping...", 0.8)

    # Default: treat state columns as they come from df_state_wide
    state_source_cols = set(df_state_wide.columns) if not df_state_wide.empty else set()

    if user_config and "mapping" in user_config:
        forward_map = mapping or {}
        if isinstance(forward_map, dict) and forward_map:
            reverse_map = {v: k for k, v in forward_map.items()}
            combined_df = combined_df.rename(columns=reverse_map)
            provenance = provenance.rename(columns=reverse_map)
            
            # Update state_source_cols to their mapped names
            state_source_cols = {reverse_map.get(c, c) for c in state_source_cols}

    # 5b. Apply unified Modbus interpretation hook (currently NO-OP)
    # NOTE: When implementing, we may want to pass an explicit source_hint
    # such as "grafana_csv" vs "ha_csv" depending on upstream caller.
    combined_df = apply_modbus_interpretation(
        combined_df,
        mapping=user_config.get("mapping") if user_config else None,
        source_hint=None,
    )

    # ===================================================================
    # 6. Smart heartbeat-driven forward-fill
    # ===================================================================
    # Uses pattern analysis to determine per-sensor forward-fill limits based
    # on actual reporting behavior. This handles:
    # - Zone/room signals with varying heartbeats
    # - Binary state sensors (zones, valves) that report on-change
    # - Periodic sensors (temps, power) with conservative limits
    # - Sparse sensors (weather) with longer allowable gaps
    #
    # Falls back to the v2.8.1 two-tier approach if pattern analysis fails.
    # ===================================================================

    # Original report times per sensor (before resampling), keyed by the
    # mapped column name, so the heartbeat analysis can learn each sensor's
    # true cadence (1-min core sensors vs ~30-min on-change room sensors).
    raw_timestamps = {} if reporting_patterns else _raw_report_times(numeric_dfs, state_dfs, user_config)

    combined_df = combined_df.sort_index()
    provenance = provenance.reindex(index=combined_df.index, columns=combined_df.columns, fill_value=PROVENANCE_MISSING)
    combined_df = _coerce_numeric_sensors(combined_df)
    before_smart_fill = combined_df.notna()

    # Try smart forward-fill with pattern analysis
    patterns = None
    try:
        if reporting_patterns:
            # The whole request's patterns (chunked.py); any column they lack is learned here.
            patterns = {c: reporting_patterns[c] for c in combined_df.columns if c in reporting_patterns}
            missing = [c for c in combined_df.columns if c not in patterns]
            if missing:
                patterns.update(analyze_sensor_reporting_patterns(
                    combined_df[missing], baselines=heartbeat_baseline,
                    raw_timestamps=_raw_report_times(numeric_dfs, state_dfs, user_config)))
        else:
            # Analyze reporting patterns from the current dataset
            patterns = analyze_sensor_reporting_patterns(
                combined_df,
                baselines=heartbeat_baseline,
                raw_timestamps=raw_timestamps,
            )

        # Apply smart forward-fill based on learned patterns
        combined_df = smart_forward_fill(combined_df, patterns)

    except Exception as e:
        # Fallback to v2.8.1 two-tier approach on any error
        _emit_message(
            on_message,
            "warning",
            f"Smart forward-fill unavailable, using fallback: {e}",
        )
        patterns = None

        if not df_state_wide.empty:
            state_cols = [c for c in combined_df.columns if c in state_source_cols]
            numeric_cols = [c for c in combined_df.columns if c not in state_source_cols]

            # State sensors: binary sensors can stay "on" for 3+ hours
            if state_cols:
                combined_df[state_cols] = combined_df[state_cols].ffill(limit=240)

            # Numeric sensors: conservative limit to avoid inventing data
            if numeric_cols:
                combined_df[numeric_cols] = combined_df[numeric_cols].ffill(limit=60)
        else:
            # Fallback: if no state CSV present, use original uniform limit
            combined_df = combined_df.ffill(limit=60)

    # Coercion/fallback can blank values; provenance must describe the returned
    # dataframe rather than a pre-coercion intermediate.
    provenance = provenance.reindex(index=combined_df.index, columns=combined_df.columns, fill_value=PROVENANCE_MISSING)
    newly_held = combined_df.notna() & ~before_smart_fill
    provenance[newly_held] = PROVENANCE_HELD
    provenance[combined_df.isna()] = PROVENANCE_MISSING
    provenance = provenance.astype("uint8")
    provenance.attrs["codes"] = {
        PROVENANCE_MISSING: "missing",
        PROVENANCE_OBSERVED: "observed",
        PROVENANCE_INTERPOLATED: "interpolated",
        PROVENANCE_HELD: "held",
    }

    return {
        "df": combined_df,
        "raw_history": combined_df.copy(),
        "raw_events": raw_events,
        "provenance": provenance,
        "provenance_summary": summarize_provenance(provenance),
        "unmapped_entities": unmapped_entities,
        "baselines": heartbeat_baseline or {},
        "patterns": patterns,
    }
