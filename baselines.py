# baselines.py
import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime, timezone
from utils import _log_warn
from config import (
    BASELINE_JSON_PATH,
    SENSOR_ROLES,
    CALC_VERSION,
    HEARTBEAT_MIN_REPORTING_DAY_COVERAGE,
)


PROVENANCE_MISSING = 0
PROVENANCE_OBSERVED = 1
PROVENANCE_INTERPOLATED = 2
PROVENANCE_HELD = 3
PROVENANCE_LABELS = {
    PROVENANCE_MISSING: "missing",
    PROVENANCE_OBSERVED: "observed",
    PROVENANCE_INTERPOLATED: "interpolated",
    PROVENANCE_HELD: "held",
}


def canonical_event_history(
    frame,
    timestamp_col,
    mapping=None,
    value_col=None,
    state_col=None,
    source_kind="unknown",
):
    """Return the minimal long-form event contract used by heartbeats.

    ``entity_id`` is the mapped THERM sensor name where a mapping exists and
    otherwise remains the raw source entity. ``source_entity_id`` always keeps
    the original identifier so the Unmapped Data view remains traceable.
    """
    columns = [
        "entity_id",
        "source_entity_id",
        "last_changed",
        "value",
        "source_kind",
        "is_mapped",
        "is_available",
    ]
    if frame is None or frame.empty or timestamp_col not in frame or "entity_id" not in frame:
        return pd.DataFrame(columns=columns)

    reverse_map = {
        str(source): str(role)
        for role, source in (mapping or {}).items()
        if source not in (None, "", "None")
    }
    ids = frame["entity_id"]
    if isinstance(ids.dtype, pd.CategoricalDtype):
        # Categorical ids (InfluxDB frames): map the few categories, not millions of rows.
        source_ids = ids.cat.rename_categories(ids.cat.categories.astype(str))
    else:
        source_ids = ids.astype(str)
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
    values = frame[value_col] if value_col and value_col in frame else pd.Series(np.nan, index=frame.index)

    available = timestamps.notna() & values.notna()
    if state_col and state_col in frame:
        states = frame[state_col].astype(str).str.strip().str.lower()
        available &= ~states.isin({"unavailable", "unknown", "none", "nan", ""})

    categorical = isinstance(source_ids.dtype, pd.CategoricalDtype)
    if isinstance(source_kind, dict):
        kinds = (source_ids.map(lambda s: source_kind.get(s, "unknown")) if categorical
                 else source_ids.map(source_kind).fillna("unknown"))
    else:
        kinds = pd.Series(str(source_kind), index=frame.index)
    # Categorical: the lambda runs once per category. Strings: vectorised dict lookup.
    mapped = (source_ids.map(lambda s: reverse_map.get(s, s)) if categorical
              else source_ids.map(reverse_map).fillna(source_ids))

    out = pd.DataFrame({
        "entity_id": mapped,
        "source_entity_id": source_ids,
        "last_changed": timestamps,
        "value": values,
        "source_kind": kinds,
        "is_mapped": source_ids.isin(reverse_map),
        "is_available": available,
    })
    out = out.dropna(subset=["last_changed"])
    out["entity_id"] = out["entity_id"].astype("category")
    out["source_entity_id"] = out["source_entity_id"].astype("category")
    out["source_kind"] = out["source_kind"].astype("category")
    return out.reset_index(drop=True)


def _minute_floor(values: pd.Series) -> pd.Series:
    """Floor timestamps without failing in the repeated autumn DST hour."""
    values = pd.to_datetime(values, errors="coerce")
    if values.dt.tz is None:
        return values.dt.floor("min")
    tz = values.dt.tz
    return values.dt.tz_convert("UTC").dt.floor("min").dt.tz_convert(tz)


def provenance_from_events(df, raw_events, filled_code=PROVENANCE_HELD):
    """Build a compact uint8 provenance mask aligned to source sensor columns."""
    if df is None or df.empty or raw_events is None or raw_events.empty:
        return pd.DataFrame(index=getattr(df, "index", None), dtype="uint8")

    source_cols = [
        str(c) for c in raw_events["entity_id"].astype(str).unique()
        if str(c) in df.columns
    ]
    provenance = pd.DataFrame(
        PROVENANCE_MISSING,
        index=df.index,
        columns=source_cols,
        dtype="uint8",
    )
    available = raw_events
    if "is_available" in available:
        available = available[available["is_available"].fillna(False)]

    for sensor, events in available.groupby("entity_id", observed=True):
        sensor = str(sensor)
        if sensor not in provenance.columns:
            continue
        minutes = pd.DatetimeIndex(_minute_floor(events["last_changed"]).dropna().unique())
        observed_index = provenance.index.intersection(minutes)
        provenance.loc[observed_index, sensor] = PROVENANCE_OBSERVED

    for sensor in provenance.columns:
        present = df[sensor].notna()
        provenance.loc[present & provenance[sensor].eq(PROVENANCE_MISSING), sensor] = filled_code
        provenance.loc[~present, sensor] = PROVENANCE_MISSING

    provenance.attrs["codes"] = dict(PROVENANCE_LABELS)
    return provenance


def summarize_provenance(provenance):
    """Return JSON-friendly observed/interpolated/held/missing counts per sensor."""
    if provenance is None or provenance.empty:
        return {}
    total = len(provenance)
    summary = {}
    for sensor in provenance.columns:
        counts = provenance[sensor].value_counts().to_dict()
        observed = int(counts.get(PROVENANCE_OBSERVED, 0))
        interpolated = int(counts.get(PROVENANCE_INTERPOLATED, 0))
        held = int(counts.get(PROVENANCE_HELD, 0))
        missing = int(counts.get(PROVENANCE_MISSING, 0))
        synthetic = interpolated + held
        summary[str(sensor)] = {
            "total_minutes": int(total),
            "observed_minutes": observed,
            "interpolated_minutes": interpolated,
            "held_minutes": held,
            "synthetic_minutes": synthetic,
            "missing_minutes": missing,
            "observed_pct": round((observed / total) * 100, 2) if total else 0.0,
            "synthetic_pct": round((synthetic / total) * 100, 2) if total else 0.0,
        }
    return summary


def add_energy_provenance(daily_df, engine_df, provenance, patterns=None):
    """Add strict provenance and cadence-aware daily energy companion splits.

    The strict split treats only a report in that exact minute as observed.
    The separate cadence split treats interpolation/holding inside a source's
    learned normal gap as within-cadence and longer bridging as extended-gap.

    The split follows the source channel that drives each calculation. Power
    energy uses the ``Power`` provenance. Heat uses a native ``Heat`` channel
    when present, otherwise the hydraulic inputs used to derive heat. This does
    not alter any existing energy total; it adds an auditable companion split.

    Immersion and cooling are deliberately excluded so the two electricity
    columns reconcile to Electricity_Heating_kWh + Electricity_DHW_kWh.
    """
    if daily_df is None or engine_df is None:
        return daily_df
    daily = daily_df.copy()
    if engine_df.empty or provenance is None or provenance.empty:
        return daily

    aligned = provenance.reindex(engine_df.index).fillna(PROVENANCE_MISSING)

    def _series_sum(columns):
        present = [column for column in columns if column in engine_df.columns]
        if not present:
            return None
        values = engine_df[present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return values.sum(axis=1)

    gap_limits = {
        sensor: details.get("gap_threshold_sec")
        for sensor, details in (patterns or {}).items()
        if isinstance(details, dict)
    }
    index = engine_df.index

    def _column_measured(column):
        codes = aligned[column]
        observed = codes.eq(PROVENANCE_OBSERVED)
        limit = gap_limits.get(column)
        if limit is None or not np.isfinite(limit) or limit <= 0:
            return observed
        last_report = pd.Series(index.where(observed), index=index).ffill()
        age_sec = (pd.Series(index, index=index) - last_report).dt.total_seconds()
        filled = codes.isin([PROVENANCE_INTERPOLATED, PROVENANCE_HELD])
        return observed | (filled & age_sec.le(limit))

    def _input_within_cadence(columns):
        present = [column for column in columns if column in aligned.columns]
        if len(present) != len(columns):
            return pd.Series(False, index=engine_df.index)
        measured = pd.Series(True, index=engine_df.index)
        for column in present:
            measured &= _column_measured(column)
        return measured

    def _input_strictly_observed(columns):
        present = [column for column in columns if column in aligned.columns]
        if len(present) != len(columns):
            return pd.Series(False, index=engine_df.index)
        return aligned[present].eq(PROVENANCE_OBSERVED).all(axis=1)

    def _daily_split(values, observed, observed_name, estimated_name):
        if values is None:
            return
        observed_values = values.where(observed, 0.0)
        estimated_values = values.where(~observed, 0.0)
        daily[observed_name] = observed_values.resample("D").sum().reindex(daily.index, fill_value=0.0) / 60000.0
        daily[estimated_name] = estimated_values.resample("D").sum().reindex(daily.index, fill_value=0.0) / 60000.0

    power_values = _series_sum(["Power_Heating", "Power_DHW"])
    _daily_split(
        power_values,
        _input_within_cadence(["Power"]),
        "WithinCadence_Input_Electricity_kWh",
        "ExtendedGap_Input_Electricity_kWh",
    )
    _daily_split(
        power_values,
        _input_strictly_observed(["Power"]),
        "Observed_Input_Electricity_kWh",
        "Estimated_Input_Electricity_kWh",
    )

    heat_values = _series_sum(["Heat_Heating", "Heat_DHW"])
    heat_source = engine_df.attrs.get("heat_source")
    if heat_source == "native" or (
        heat_source is None and "Heat" in aligned.columns
    ):
        heat_inputs = ["Heat"]
    elif "DeltaT" in aligned.columns:
        heat_inputs = ["FlowRate", "DeltaT"]
    else:
        heat_inputs = ["FlowRate", "FlowTemp", "ReturnTemp"]
    _daily_split(
        heat_values,
        _input_within_cadence(heat_inputs),
        "WithinCadence_Input_Heat_kWh",
        "ExtendedGap_Input_Heat_kWh",
    )
    _daily_split(
        heat_values,
        _input_strictly_observed(heat_inputs),
        "Observed_Input_Heat_kWh",
        "Estimated_Input_Heat_kWh",
    )
    return daily

_DAY_NS = 86_400 * 10**9


def _report_arrays(history_df):
    """
    The report history as flat arrays, for valid timestamps of available
    reports: entity codes and names, UTC nanoseconds (for gaps), the calendar
    day in the timestamps' own time zone (for daily counts, like .dt.date),
    is_mapped, and source-entity codes and names. A 10-month analysis has ~9M
    rows; pandas groupbys on it took minutes and GBs, these arrays do not.
    """
    cols = history_df.columns
    times = pd.to_datetime(history_df["last_changed"], errors="coerce")
    mask = times.notna().to_numpy()
    if "is_available" in cols:
        mask &= history_df["is_available"].fillna(False).astype(bool).to_numpy()
    rows = np.flatnonzero(mask)

    stamps = times.array  # DatetimeArray, no copy
    utc_ns = stamps.asi8[rows]
    if stamps.tz is not None:
        day = (stamps.tz_localize(None).asi8[rows] // _DAY_NS).astype(np.int32)
    else:
        day = (utc_ns // _DAY_NS).astype(np.int32)
    del times, stamps
    ents = history_df["entity_id"]
    if isinstance(ents.dtype, pd.CategoricalDtype):
        # All categories, as groupby(observed=False) listed them.
        names = list(ents.cat.categories)
        codes = ents.cat.codes.to_numpy()[rows]
    else:
        cat = pd.Categorical(ents.iloc[rows])
        names, codes = list(cat.categories), cat.codes
    out = {
        "names": names,
        "codes": codes.astype(np.int32),
        "utc": utc_ns,
        "day": day,
        "mapped": (history_df["is_mapped"].fillna(False).astype(bool).to_numpy()[rows]
                   if "is_mapped" in cols else np.ones(len(rows), dtype=bool)),
        "src_names": None,
        "src_codes": None,
    }
    if "source_entity_id" in cols:
        src = history_df["source_entity_id"]
        src_cat = src if isinstance(src.dtype, pd.CategoricalDtype) else src.astype("category")
        out["src_names"] = list(src_cat.cat.categories)
        out["src_codes"] = src_cat.cat.codes.to_numpy()[rows].astype(np.int32)
    return out


def _subset(arr, keep):
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in arr.items()}


def _active_day_numbers(active_days):
    if active_days is None:
        return None
    idx = pd.Index(active_days)
    if idx.dtype.kind in "iu":
        return np.unique(idx.to_numpy())
    stamps = pd.DatetimeIndex(pd.to_datetime(idx))
    if stamps.tz is not None:
        stamps = stamps.tz_localize(None)
    return np.unique(stamps.asi8 // _DAY_NS)


def _baselines_from_arrays(arr, sensor_roles, min_good_days, active, min_reporting_day_coverage):
    baselines = {}
    if not arr["mapped"].all():
        arr = _subset(arr, arr["mapped"])
    if len(arr["codes"]) == 0:
        return baselines
    if active is None:
        active = np.unique(arr["day"])
    n_active = len(active)

    source_ids = {}
    if arr["src_codes"] is not None:
        n_src = max(len(arr["src_names"]), 1)
        # (sensor, source) pairs, one sensor at a time to keep memory flat
        for code in np.unique(arr["codes"]):
            srcs = np.unique(arr["src_codes"][arr["codes"] == code])
            source_ids[int(code)] = {str(arr["src_names"][s]) for s in srcs if 0 <= s < n_src}

    # Overlapping exports can repeat the same event. A heartbeat measures
    # reporting cadence, so count unique source timestamps per mapped role.
    order = np.lexsort((arr["utc"], arr["codes"]))
    c = arr["codes"][order]
    u = arr["utc"][order]
    first = np.ones(len(c), dtype=bool)
    first[1:] = (c[1:] != c[:-1]) | (u[1:] != u[:-1])
    d = arr["day"][order][first]
    del order
    c, u = c[first], u[first]
    del first
    # Days never go backwards within a sensor's time-sorted reports, so each
    # (sensor, day) is one run.
    brk = np.ones(len(c), dtype=bool)
    brk[1:] = (c[1:] != c[:-1]) | (d[1:] != d[:-1])
    starts = np.flatnonzero(brk)
    run_counts = np.diff(np.append(starts, len(c)))
    run_code, run_day = c[starts], d[starts]

    for code, entity_id in enumerate(arr["names"]):
        role = sensor_roles.get(entity_id, 'unknown')
        if role == 'rare_event':
            baselines[entity_id] = {'role': role, 'has_baseline': False}
            continue
        lo, hi = np.searchsorted(c, code, 'left'), np.searchsorted(c, code, 'right')
        if hi == lo:
            continue
        rlo, rhi = np.searchsorted(run_code, code, 'left'), np.searchsorted(run_code, code, 'right')
        daily_stats = pd.Series(run_counts[rlo:rhi])
        reporting_day_coverage = (
            float(np.isin(run_day[rlo:rhi], active).sum()) / n_active if n_active else 0.0
        )
        # Reference = 90th percentile of daily report counts, not the single
        # busiest day. Change-driven sensors (e.g. Power reporting sub-minute)
        # have occasional burst days; measuring against the max left too few
        # "good" days and silently dropped the sensor from the heartbeat.
        best_minutes = float(daily_stats.quantile(0.9, interpolation='lower'))
        if not np.isfinite(best_minutes) or best_minutes <= 0: continue

        # --- LOGIC UPDATE: Sparse Sensor Handling ---
        if role == 'weather_sparse':
            # For hourly sensors (OWM), even 12 samples (50%) is "good"
            good_min = max(1, best_minutes * 0.5)
        else:
            # For standard sensors, we expect 90% consistency
            good_min = best_minutes * 0.9

        good_days_mask = daily_stats >= good_min
        intermittent_role = role in {'core_periodic', 'room_temp', 'weather_on_change'}
        if intermittent_role and reporting_day_coverage < min_reporting_day_coverage:
            baselines[entity_id] = {
                'role': role,
                'has_baseline': False,
                'reason': 'intermittent',
                'reporting_days': int(len(daily_stats)),
                'active_days': int(n_active),
                'reporting_day_coverage': round(reporting_day_coverage, 4),
            }
            continue
        if good_days_mask.sum() < min_good_days:
            # Not enough consistent days to form a baseline
            baselines[entity_id] = {'role': role, 'has_baseline': False}
            continue

        # Expected reports/day = median over ALL active days: a typical day,
        # not a busy one. The good-days test above only decides whether the
        # sensor is consistent enough to have a baseline at all. (For a
        # fixed-cadence sensor both give the cadence; for change-driven
        # sensors the busy-day rate made an ordinary period look ~25% short.)
        expected_minutes = max(1, int(round(float(daily_stats.median()))))

        baselines[entity_id] = {
            'role': role,
            # New name states the actual unit. Keep expected_minutes as a
            # compatibility alias for existing exported heartbeat files.
            'expected_reports_per_day': expected_minutes,
            'expected_minutes': expected_minutes,
            'has_baseline': True,
            'good_days': int(good_days_mask.sum()),
            'history_days': int(len(daily_stats)),
            'reporting_days': int(len(daily_stats)),
            'active_days': int(n_active),
            'reporting_day_coverage': round(reporting_day_coverage, 4),
        }
        # Normal silences between reports over the long history. A short
        # analysis inherits these so a quiet month doesn't learn gap limits
        # that are too tight (see analyze_sensor_reporting_patterns).
        gaps = pd.Series(np.diff(u[lo:hi]) * 1e-9)
        gaps = gaps[gaps > 0]
        if len(gaps):
            baselines[entity_id]['gap_q95_sec'] = round(float(gaps.quantile(0.95)), 1)
            baselines[entity_id]['gap_q99_sec'] = round(float(gaps.quantile(0.99)), 1)
        if arr["src_codes"] is not None:
            baselines[entity_id]['source_entity_ids'] = sorted(source_ids.get(code, set()))

    return baselines


def build_sensor_baselines(
    history_df,
    sensor_roles,
    min_good_days=3,
    active_days=None,
    min_reporting_day_coverage=HEARTBEAT_MIN_REPORTING_DAY_COVERAGE,
):
    """
    Scans history to determine the 'normal' reporting frequency (samples/day)
    for each sensor.
    """
    if history_df is None or history_df.empty:
        return {}
    if 'entity_id' not in history_df.columns or 'last_changed' not in history_df.columns:
        return {}
    return _baselines_from_arrays(
        _report_arrays(history_df), sensor_roles, min_good_days,
        _active_day_numbers(active_days), min_reporting_day_coverage,
    )

def analyze_sensor_reporting_patterns(df, baselines=None, raw_timestamps=None):
    """
    Analyzes the current dataset to determine reporting intervals (gap thresholds)
    for filling data.

    raw_timestamps: optional {column: DatetimeIndex} of each sensor's ORIGINAL
    report times (before 1-minute resampling). Pass these whenever available:
    after resampling and interpolation every sensor looks like a 1-minute sensor,
    so its true cadence (e.g. a room sensor reporting every ~30 min on change)
    can't be learned from the resampled frame.
    """
    patterns = {}
    if df.empty: return patterns
    raw_timestamps = raw_timestamps or {}

    for entity_id in df.columns:
        base = baselines.get(entity_id, {}) if baselines else {}
        role = base.get('role', SENSOR_ROLES.get(entity_id, 'unknown'))
        baseline_ready = base.get('has_baseline', True)
        expected_reports = (
            base.get('expected_reports_per_day', base.get('expected_minutes', 0))
            if baseline_ready else 0
        )
        baseline_interval = (
            (24 * 60 * 60) / expected_reports
            if expected_reports and expected_reports > 0
            else 60.0
        )

        # Prefer the sensor's original report times; fall back to non-NaN
        # timestamps of the (resampled) frame.
        if entity_id in raw_timestamps:
            g = pd.DatetimeIndex(raw_timestamps[entity_id]).sort_values()
        else:
            g = df[entity_id].dropna().index
        if len(g) < 2:
            # A short analysis can legitimately contain only one (or no)
            # report from a slow sensor. In that case the long-period
            # heartbeat must supply the cadence instead of being discarded.
            median_interval = baseline_interval
            q95_gap = baseline_interval
            q99_gap = baseline_interval
        else:
            diffs = g.to_series().diff().dt.total_seconds().dropna()
            if len(diffs) == 0:
                median_interval = baseline_interval
                q95_gap = baseline_interval
                q99_gap = baseline_interval
            else:
                median_interval = diffs.median()
                q95_gap = diffs.quantile(0.95)
                q99_gap = diffs.quantile(0.99)

        # A reusable heartbeat carries the long-period normal silences. Take
        # the larger of short vs long: a quiet short period can't tighten the
        # gap limit below what is normal for this sensor over the long run.
        if baseline_ready and base.get('gap_q95_sec'):
            q95_gap = max(q95_gap, float(base['gap_q95_sec']))
        if baseline_ready and base.get('gap_q99_sec'):
            q99_gap = max(q99_gap, float(base['gap_q99_sec']))

        # Smart fusion of observed vs baseline interval
        normal_interval = (
            baseline_interval
            if len(g) < 2 and expected_reports
            else np.mean([median_interval, baseline_interval])
        )
        # Use the larger of observed Q95 or implied baseline gap to prevent over-segmentation
        baseline_gap_95 = baseline_interval * 2.0 
        gap_95 = max(q95_gap, baseline_gap_95)

        # --- LOGIC UPDATE: Sparse Role Handling ---
        if role == 'weather_sparse':
            report_type = 'sparse'
            # Allow huge gaps (up to 2 hours) for OWM
            gap_threshold = max(normal_interval * 2.0, gap_95, 7200.0)
        elif role in ('room_temp', 'setpoint', 'weather_on_change'):
            # Room sensors (e.g. Zigbee) report on CHANGE: a steady room can be
            # silent for hours (typically a median gap of ~30 min, up to ~6 h).
            # Weather stations behave the same way through HA/InfluxDB (Ecowitt
            # humidity: median 2 min, max ~340 min; solar sits at 0 overnight).
            # Hold the last reading across the sensor's normal silences.
            report_type = 'on_change_numeric'
            gap_threshold = max(normal_interval * 6.0, q99_gap, 3600.0)
            if role == 'weather_on_change':
                # InfluxDB carries no 'unavailable' rows, so silence cannot be
                # told from an outage; a steady spell of up to 3 h is normal.
                # (HA exports mark outages explicitly and blank them directly.)
                gap_threshold = max(gap_threshold, 3 * 3600.0)
        elif role in ('core_periodic', 'unknown'):
            report_type = 'periodic'
            gap_threshold = max(normal_interval * 6.0, gap_95, 600.0) 
        elif role == 'binary_state':
            report_type = 'on_change'
            gap_threshold = max(normal_interval * 4.0, gap_95, 1800.0)
        else:
            report_type = 'rare_event'
            gap_threshold = 3600.0

        patterns[entity_id] = {
            'normal_interval_sec': normal_interval,
            'report_type': report_type,
            'gap_threshold_sec': gap_threshold,
            'sample_count': int(len(g)),
            'role': role,
            'baseline_expected_reports_per_day': expected_reports or None,
            # Compatibility for downstream consumers of the former name.
            'baseline_expected_minutes': expected_reports or None,
        }
    return patterns


def add_heartbeat_daily_counts(daily_df, raw_events):
    """Add raw report counts used when an external heartbeat is active."""
    if daily_df is None:
        return daily_df
    result = daily_df.copy()
    if len(result.index) == 0 or raw_events is None or raw_events.empty:
        return result

    events = raw_events.copy()
    events['last_changed'] = pd.to_datetime(events['last_changed'], errors='coerce')
    events = events.dropna(subset=['last_changed'])
    if 'is_available' in events.columns:
        events = events[events['is_available'].fillna(False)]
    # Heartbeats are keyed by mapped role; unmapped sources belong on the
    # Unmapped Data tab, not as HB_<raw entity>_Count columns.
    if 'is_mapped' in events.columns:
        events = events[events['is_mapped'].fillna(False)]
    if events.empty:
        return result

    events['day'] = _minute_floor(events['last_changed']).dt.floor('D')
    events = events.drop_duplicates(subset=['entity_id', 'last_changed'])
    counts = events.groupby(['day', 'entity_id'], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(result.index, fill_value=0)
    for sensor in counts.columns:
        result[f"HB_{sensor}_Count"] = counts[sensor].astype(int)
    return result

def smart_forward_fill(df_resampled, patterns):
    """
    Fills NaN gaps in the resampled DataFrame based on the calculated gap thresholds.
    """
    if df_resampled.empty: return df_resampled
    df_filled = df_resampled.copy()
    
    for col in df_filled.columns:
        pat = patterns.get(col)
        
        # Determine filling limit
        if pat is None:
            limit_minutes = 5
        else:
            # Convert gap threshold (seconds) to minutes for ffill limit
            gap_sec = pat.get('gap_threshold_sec', 300)
            limit_minutes = int(np.ceil(gap_sec / 60.0))

            # --- LOGIC UPDATE: Smart limits based on sensor reporting patterns ---
            if pat.get('report_type') == 'sparse':
                # OWM sensors: cap at 2 hours (120 mins)
                limit_minutes = min(limit_minutes, 120)
            elif pat.get('report_type') == 'on_change_numeric':
                # Room temperatures: hold up to 6 hours; longer is a real outage
                limit_minutes = min(limit_minutes, 360)
            elif pat.get('report_type') == 'periodic':
                # Core sensors: cap at 20 mins to avoid inventing data
                limit_minutes = min(limit_minutes, 20)
            elif pat.get('report_type') in ('on_change', 'rare_event'):
                # Binary state / rare event sensors: Trust the learned pattern
                # These report when state CHANGES, not periodically
                # Defrost might stay "off" for months, zones "off" for 20+ hours
                # Use the calculated gap threshold from actual data (95th percentile)
                # No arbitrary cap - the pattern analysis already provides safe limits
                pass  # Use limit_minutes as calculated from gap_threshold_sec
        
        # Apply forward fill with limit
        df_filled[col] = df_filled[col].ffill(limit=limit_minutes)

        # For binary state sensors: backfill leading NaNs with 0 (default "off" state)
        # These sensors report on state CHANGE - absence means default state
        if pat and pat.get('report_type') in ('on_change', 'rare_event'):
            # Only fill leading NaNs (before first real value)
            first_valid_idx = df_filled[col].first_valid_index()
            if first_valid_idx is not None:
                df_filled.loc[:first_valid_idx, col] = df_filled[col].loc[:first_valid_idx].fillna(0)

    return df_filled

def load_heartbeat_baseline(source, current_month=None):
    """Load a heartbeat from a path, uploaded file, bytes, or JSON string.

    Returns ``(baselines, source_label, metadata)``. Legacy plain dictionaries,
    wrapped files, and seasonal lists remain supported.
    """
    if source is None:
        return {}, None, {}

    source_label = getattr(source, "name", None)
    try:
        if hasattr(source, "getvalue"):
            raw = source.getvalue()
            raw = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            data = json.loads(raw)
        elif hasattr(source, "read"):
            raw = source.read()
            raw = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            data = json.loads(raw)
        elif isinstance(source, bytes):
            data = json.loads(source.decode("utf-8"))
        elif isinstance(source, (str, os.PathLike)) and os.path.exists(source):
            source_label = str(source)
            with open(source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = json.loads(str(source))

        # 1. Simple dict format (Legacy)
        if isinstance(data, dict) and 'meta' not in data and 'baselines' not in data:
            return data, source_label or "uploaded heartbeat", {}

        # 2. Wrapped format (Standard)
        if isinstance(data, dict) and 'baselines' in data and isinstance(data['baselines'], dict):
            return data['baselines'], source_label or "uploaded heartbeat", data.get('meta', {})

        # 3. Seasonal List format (Advanced)
        if isinstance(data, list):
            # Find entry for current month
            if current_month is None: current_month = datetime.now().month
            
            best_entry = None
            for entry in data:
                if 'months' in entry and current_month in entry['months']:
                    best_entry = entry
                    break
            
            # Fallback to 'default' tag if no month match
            if not best_entry:
                for entry in data:
                    if entry.get('tag') == 'default':
                        best_entry = entry
                        break
            
            if best_entry:
                label = source_label or "uploaded seasonal heartbeat"
                return (
                    best_entry.get('baselines', {}),
                    f"{label} [{best_entry.get('tag')}]",
                    best_entry.get('meta', {}),
                )

        return {}, source_label, {}

    except Exception as e:
        _log_warn(f"Error loading heartbeat baseline: {e}")
        return {}, None, {}


def load_saved_heartbeat_baseline(json_path, current_month=None):
    """Backward-compatible path loader returning the historical two-tuple."""
    baselines, label, _meta = load_heartbeat_baseline(json_path, current_month)
    return baselines, label

def build_offline_aware_seasonal_baseline(history_df, sensor_roles):
    """
    Wrapper to build baselines that handles offline days intelligently.
    """
    if history_df is None or history_df.empty:
        return {}
    if 'entity_id' not in history_df.columns or 'last_changed' not in history_df.columns:
        return {}
    arr = _report_arrays(history_df)
    if len(arr["codes"]) == 0:
        return {}

    # Simple check for "system active" days to filter out complete outages
    days, counts = np.unique(arr["day"], return_counts=True)
    busy_threshold = np.median(counts) * 0.1
    active_days = days[counts > busy_threshold]

    # Only mapped reports on active days (one copy; the original is released)
    arr = _subset(arr, np.isin(arr["day"], active_days) & arr["mapped"])
    return _baselines_from_arrays(
        arr, sensor_roles, 3, active_days, HEARTBEAT_MIN_REPORTING_DAY_COVERAGE,
    )

def heartbeat_baseline_payload(
    baselines,
    tag,
    days_in_history=0,
    profile_name=None,
    period_start=None,
    period_end=None,
):
    """Build the portable heartbeat JSON wrapper used for download/reload."""
    return {
        "meta": {
            "schema_version": 2,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "calc_version": CALC_VERSION,
            "days_analyzed": int(days_in_history),
            "tag": tag or "manual_generation",
            "profile_name": profile_name,
            "period_start": str(period_start) if period_start is not None else None,
            "period_end": str(period_end) if period_end is not None else None,
        },
        "baselines": baselines
    }


def save_heartbeat_baseline_to_json(
    baselines,
    tag,
    days_in_history=0,
    profile_name=None,
    period_start=None,
    period_end=None,
):
    """Save a portable heartbeat baseline to the configured local JSON path."""
    output = heartbeat_baseline_payload(
        baselines,
        tag,
        days_in_history,
        profile_name,
        period_start,
        period_end,
    )
    with open(BASELINE_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    return BASELINE_JSON_PATH
