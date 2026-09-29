# view_quality.py

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

from config import (
    UNSCORED_DQ_MODES,
    SENSOR_EXPECTATION_MODE,
    SENSOR_GROUPS,
    SENSOR_ROLES,
    BASELINE_JSON_PATH,
)
from utils import availability_pct, strip_entity_prefix
import baselines


def render_data_quality(
    daily_df: pd.DataFrame,
    df: pd.DataFrame,
    unmapped_entities: list,
    patterns: dict | None,
    heartbeat_path: str | None,
    raw_events: pd.DataFrame | None = None,
    provenance_summary: dict | None = None,
    user_config: dict | None = None,
) -> None:
    """
    Data Quality Studio:
    - Overview scorecard (DQ_Tier + category availability)
    - Category drill-down
    - Master sensor matrix
    - Heartbeat baselines
    - Unmapped data
    """
    st.title("Data Quality")

    if daily_df is None or daily_df.empty:
        st.warning("No data loaded.")
        return

    # ------------------------------------------------------------------
    # Partial Day Logic
    # ------------------------------------------------------------------
    # Denominator is based on actual system-on minutes per day rather than
    # blindly hardcoding 1440 mins. This allows 100% uptime on partial days.
    # Recorded_Minutes = minutes with a Power reading (from get_daily_stats);
    # Power_count is the same figure, kept as a fallback for older daily tables.
    system_on_minutes = daily_df.apply(
        lambda r: max(
            r.get("Recorded_Minutes", 0),
            r.get("Power_count", 0),
            1,
        ),
        axis=1,
    )

    def count_column_for(sensor_name: str) -> str | None:
        """Minutes-with-data column used for availability scoring.

        Raw report counts (HB_<sensor>_Count) are NOT scored: almost every
        source reports on change, so fewer reports on a quiet day is not
        missing data. A heartbeat affects scoring through the gap limits it
        supplies to gap-filling (a silence longer than the long-run normal
        leaves minutes empty). Report rates are compared on the Heartbeats tab.
        """
        windowed_col = f"DQ_{sensor_name}_Count"
        if windowed_col in daily_df.columns:
            return windowed_col
        ordinary_col = f"{sensor_name}_count"
        if ordinary_col in daily_df.columns:
            return ordinary_col
        return None

    def expected_window_series(sensor_name: str, system_on_minutes_series: pd.Series):
        """
        Compute the expected minutes-with-data per day for a sensor from
        SENSOR_EXPECTATION_MODE (system/heating/dhw/system_slow/event_only).

        (A former "Scenario A" divided heartbeat reports/day by 1440 and used
        it as expected MINUTES. The units didn't match - a forward-filled
        sparse sensor always scored 100% - so it was removed in v1.29.0.)
        """
        mode = SENSOR_EXPECTATION_MODE.get(sensor_name, "system")

        # Counts come from the 1-minute engine grid (after resample and
        # gap-filling), so every expectation is in minutes. heating_active and
        # dhw_active sensors are counted only inside their window
        # (DQ_<sensor>_Count), matching these denominators.
        if mode == "heating_active":
            base = daily_df.get("Active_Mins", system_on_minutes_series)
        elif mode == "dhw_active":
            base = daily_df.get("DHW_Mins", system_on_minutes_series)
        else:
            # "system" and "system_slow": sparse (hourly) sensors are forward-
            # filled onto the minute grid, so they are also judged per minute.
            base = system_on_minutes_series

        return base.replace(0, np.nan)

    def format_dq_df(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        df_out.index.name = "Date"
        try:
            df_out.index = df_out.index.strftime("%d-%m-%Y")
        except Exception:
            # If not a datetime index, leave as-is
            pass
        return df_out

    # --- Helper: resolve mapped / display names (shared by all tabs) ---
    config_obj = st.session_state.get("system_config", {}) or {}
    if isinstance(config_obj, dict):
        mapping = config_obj.get("mapping", {}) or {}
    else:
        mapping = {}

    mapped_keys = set(mapping.keys())

    # Canonical UI labels for standardised core data points
    STANDARD_LABELS = {
        "Power": "Outdoor Power",
        "Indoor_Power": "Indoor Power",
        "Heat": "Heat Output",
        "FlowTemp": "Flow Temp",
        "ReturnTemp": "Return Temp",
        "ValveMode": "3 Way Valve",
        "OutdoorTemp": "Outdoor Temp",
        "DHW_Temp": "Hot Water Temp",
        "Freq": "Compressor Freq",
    }

    # Derived / internal-only names that should NEVER appear in All Sensors
    DERIVED_NAMES = {
        "DeltaT",
        "COP_Real",
        "COP_Graph",
        "Active_Zones_Count",
        "hour",
        "is_night_rate",
        "Current_Rate",
        "Zone_Config",
        "Immersion_Active",
        "Immersion_Power",
        "Cost_Inc",
        "Electricity_Heating_Wmin",
        "Electricity_DHW_Wmin",
        "Heat_Heating_Wmin",
        "Heat_DHW_Wmin",
        "Immersion_Wh",
    }

    def sensor_is_allowed(name: str) -> bool:
        """
        Only include:
        - Sensors that the user explicitly mapped in the profile
        - AND are not in the derived/internal blacklist.
        """
        if not name:
            return False
        if name in DERIVED_NAMES:
            return False
        return name in mapped_keys

    def display_label_for(internal_name: str) -> str:
        """
        What the user sees on the front end.

        - Zones & Rooms: show the mapped sensor name (entity_id) with prefixes stripped.
        - Standard core sensors: show FRIENDLY labels (Outdoor Power, Flow Temp, etc.).
        - Everything else: mapped entity_id (prefix-stripped), or internal name if unmapped.
        """
        if internal_name.startswith("Zone_") or internal_name.startswith("Room_"):
            return strip_entity_prefix(str(mapping.get(internal_name, internal_name)))

        if internal_name in STANDARD_LABELS:
            return STANDARD_LABELS[internal_name]

        return strip_entity_prefix(str(mapping.get(internal_name, internal_name)))


    def tooltip_label_for(internal_name: str) -> str:
        """
        Tooltip content: show the mapped sensor (entity_id),
        and optionally the internal name for debugging context.
        """
        entity = mapping.get(internal_name, internal_name)
        if entity == internal_name:
            return str(entity)
        return f"{entity} (internal: {internal_name})"


    dq_tab1, dq_tab2, dq_tab3, dq_tab4, dq_tab5 = st.tabs(
        ["Overview", "Category Drill-Down", "All Sensors", "Heartbeats", "⚠️ Unmapped Data"]
    )

    # ------------------------------------------------------------------
    # TAB 1: Overview
    # ------------------------------------------------------------------
    with dq_tab1:
        st.markdown("### System Health Scorecard")

        dq_avg = float(daily_df.get("DQ_Score", 0).mean())
        tiers = daily_df.get("DQ_Tier", pd.Series("", index=daily_df.index)).astype(str)
        # Today is still in progress: its coverage is naturally low. Label it as such
        # instead of "Bronze", and leave it out of the tier counts.
        try:
            today = datetime.now(ZoneInfo((user_config or {}).get("timezone", "Europe/Dublin"))).date()
        except (KeyError, TypeError, ValueError):
            today = date.today()
        is_today = pd.Series([str(d)[:10] == today.isoformat() for d in daily_df.index],
                             index=daily_df.index)
        tiers = tiers.where(~is_today, "Today (in progress)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Average Power Coverage",
            f"{dq_avg:.1f}%",
            help="Mean % of each day's 1440 minutes with a Power reading (DQ_Score).",
        )
        c2.metric("Gold Days", int(tiers.str.contains("Gold").sum()))
        c3.metric("Silver Days", int(tiers.str.contains("Silver").sum()))
        c4.metric("Bronze Days", int(tiers.str.contains("Bronze").sum()))
        st.caption(
            "Gold: Power coverage and heat-input coverage are both at least 90%. "
            "Heat-input coverage follows the engine source: the Heat meter, or Flow Temp, "
            "Return Temp and Flow Rate when heat is derived. Silver: Power is at least "
            "90% but heat input is incomplete or unavailable. Bronze: Power coverage "
            "is below 90%, including partial first/last days."
        )

        overview_df = daily_df[["DQ_Tier"]].copy()
        overview_df["DQ_Tier"] = tiers
        group_cols: list[str] = []

        for group_name, sensors in SENSOR_GROUPS.items():
            if "Events" in group_name or "Event" in group_name:
                continue

            # Identify sensors that actually have count columns
            valid_sensors = [
                s
                for s in sensors
                if count_column_for(s) is not None
            ]
            if not valid_sensors:
                continue

            group_pcts = []
            for s in valid_sensors:
                # On-change / event sensors aren't scored (silence = unchanged)
                if SENSOR_EXPECTATION_MODE.get(s, "system") in UNSCORED_DQ_MODES:
                    continue
                col = count_column_for(s)
                pct = availability_pct(
                    daily_df[col],
                    expected_window_series(s, system_on_minutes),
                )
                group_pcts.append(pct)

            if group_pcts:
                overview_df[group_name] = (
                    pd.concat(group_pcts, axis=1).mean(axis=1).round(0)
                )
                group_cols.append(group_name)

        overview_disp = format_dq_df(overview_df[["DQ_Tier"] + group_cols])

        st.dataframe(
            overview_disp.style.background_gradient(
                subset=group_cols,
                cmap="RdYlGn",
                vmin=0,
                vmax=100,
            ).format("{:.0f}", subset=group_cols),
            width="stretch",
        )

    # ------------------------------------------------------------------
    # TAB 2: Category Drill-Down
    # ------------------------------------------------------------------
    with dq_tab2:
        st.markdown("### Category Inspector")

        cat = st.selectbox("Select System Category", list(SENSOR_GROUPS.keys()))
        selected = SENSOR_GROUPS.get(cat, [])

        cat_df = pd.DataFrame(index=daily_df.index)
        valid_cols: list[str] = []

        for sensor in selected:
            col_name = count_column_for(sensor)
            if col_name is None:
                continue

            mode = SENSOR_EXPECTATION_MODE.get(sensor, "system")

            if mode in UNSCORED_DQ_MODES or "defrost" in sensor.lower():
                # On-change / event sensors: state changes per day, not scored
                cat_df[sensor] = daily_df[col_name].fillna(0).astype(int)
            else:
                exp = expected_window_series(sensor, system_on_minutes)
                cat_df[sensor] = availability_pct(
                    daily_df[col_name], exp
                ).round(0)

            valid_cols.append(sensor)

        cat_disp = format_dq_df(cat_df)

        if not cat_disp.empty:
            normal_cols = [
                c
                for c in valid_cols
                if SENSOR_EXPECTATION_MODE.get(c, "system") not in UNSCORED_DQ_MODES
            ]
            styler = cat_disp.style.format("{:.0f}", na_rep="-")
            if normal_cols:
                styler = styler.background_gradient(
                    subset=normal_cols,
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=100,
                )
            st.dataframe(styler, width="stretch")
            st.caption(
                "Coloured cells: % of expected reporting/availability with data. Grey cells: state "
                "changes that day for sensors that only report on change (valve, "
                "pumps, zones, modes) or rare events (defrost). These aren't scored, "
                "because silence means the state didn't change, not that data is missing."
            )
        else:
            st.info("No data for this category.")

    # ------------------------------------------------------------------
    # TAB 3: Master Sensor Matrix (with categorisation from main branch)
    # ------------------------------------------------------------------
    with dq_tab3:
        st.markdown("### Master Sensor Matrix")

        # 1. Build Data (counts -> availability % or raw event counts)
        count_cols = {
            sensor: count_column_for(sensor)
            for sensor in mapped_keys
            if count_column_for(sensor) is not None
        }
        if count_cols:
            flat_data: dict[str, pd.Series] = {}

            for clean_name, c in count_cols.items():
                # Skip internal diagnostic counters, like short-cycle trackers
                if "short_cycle" in clean_name.lower():
                    continue

                # Only keep "real" sensors that the user mapped
                if not sensor_is_allowed(clean_name):
                    continue

                mode = SENSOR_EXPECTATION_MODE.get(clean_name, "system")

                # Event-style sensors: raw count
                if mode in UNSCORED_DQ_MODES:
                    flat_data[clean_name] = daily_df[c].fillna(0).astype(int)
                else:
                    flat_data[clean_name] = availability_pct(
                        daily_df[c],
                        expected_window_series(clean_name, system_on_minutes),
                    ).round(0)

            df_flat = pd.DataFrame(flat_data, index=daily_df.index)

            if df_flat.empty:
                st.info("No mapped sensor count columns found in daily data.")
            else:
                # 2. Re-construct Ordered Columns (Events moved to end)
                column_meta = []   # list of dicts: {category, internal, display, tooltip}
                valid_data_cols: list[str] = []
                events_cat: str | None = None
                events_list: list[str] = []
                zones_cat_name: str | None = None

                # A. Normal Groups (except Zones + Events)
                for cat_name, sensors in SENSOR_GROUPS.items():
                    if "Event" in cat_name or "Events" in cat_name:
                        events_cat = cat_name
                        events_list = sensors
                        continue

                    # Option B: treat "Zones" as a generic group, not fixed names
                    if "Zones" in cat_name or " Zone" in cat_name:
                        zones_cat_name = cat_name
                        continue

                    found_sensors = [s for s in sensors if s in df_flat.columns]
                    for s in found_sensors:
                        column_meta.append(
                            {
                                "category": cat_name,
                                "internal": s,
                                "display": display_label_for(s),
                                "tooltip": tooltip_label_for(s),
                            }
                        )
                        valid_data_cols.append(s)

                # B. Zones (generic: any Zone_* present in df_flat)
                if zones_cat_name:
                    zone_cols = sorted(
                        [
                            c for c in df_flat.columns
                            if c.startswith("Zone_") and c not in valid_data_cols
                        ]
                    )
                    for z in zone_cols:
                        column_meta.append(
                            {
                                "category": zones_cat_name,
                                "internal": z,
                                "display": display_label_for(z),
                                "tooltip": tooltip_label_for(z),
                            }
                        )
                        valid_data_cols.append(z)

                # C. Rooms (any Room_* not yet placed)
                room_cols = sorted(
                    [
                        c for c in df_flat.columns
                        if c.startswith("Room_") and c not in valid_data_cols
                    ]
                )
                for r in room_cols:
                    column_meta.append(
                        {
                            "category": "️ Rooms",
                            "internal": r,
                                # Show friendly label or entity ID, not "Room_1"/"1"
                            "display": display_label_for(r),
                            "tooltip": tooltip_label_for(r),
                        }
                    )
                    valid_data_cols.append(r)

                # D. Others (still mapped, but not in any category above)
                remaining = sorted(
                    [
                        c for c in df_flat.columns
                        if c not in valid_data_cols
                        and (not events_list or c not in events_list)
                    ]
                )
                for rem in remaining:
                    column_meta.append(
                        {
                            "category": "Other",
                            "internal": rem,
                            "display": display_label_for(rem),
                            "tooltip": tooltip_label_for(rem),
                        }
                    )
                    valid_data_cols.append(rem)

                # E. Events (appended last)
                if events_cat:
                    found_events = [s for s in events_list if s in df_flat.columns]
                    for s in found_events:
                        column_meta.append(
                            {
                                "category": events_cat,
                                "internal": s,
                                "display": display_label_for(s),
                                "tooltip": tooltip_label_for(s),
                            }
                        )
                        valid_data_cols.append(s)

                # 3. Build Final DataFrame
                df_final = df_flat[valid_data_cols].copy()
                df_final = format_dq_df(df_final)

                # Build MultiIndex columns from category + display label
                multi_cols = pd.MultiIndex.from_tuples(
                    [(m["category"], m["display"]) for m in column_meta]
                )
                df_final.columns = multi_cols

                # 4. Apply Styles (normal vs event-style)
                event_cols = []
                normal_cols = []
                internal_names = [m["internal"] for m in column_meta]

                for col_tuple, internal_name in zip(df_final.columns, internal_names):
                    mode = SENSOR_EXPECTATION_MODE.get(internal_name, "system")
                    if mode in UNSCORED_DQ_MODES or "defrost" in internal_name.lower():
                        event_cols.append(col_tuple)
                    else:
                        normal_cols.append(col_tuple)

            # Only render when the matrix was built (df_flat non-empty)
            if not df_flat.empty:
                styler = df_final.style.format("{:.0f}", na_rep="-")
                if normal_cols:
                    styler = styler.background_gradient(
                        subset=normal_cols, cmap="RdYlGn", vmin=0, vmax=100,
                    )
                if event_cols:
                    # Grey background for event-style sensors
                    styler = styler.map(
                        lambda x: "background-color: #e0e0e0; color: #555555",
                        subset=event_cols,
                    )

                # NOTE:
                # We intentionally do NOT use Styler tooltips here because Streamlit
                # will escape the HTML and show it in the cells instead of rendering
                # proper hover-tooltips. Column headers already show the correct
                # standard labels (or entity_ids for Rooms/Zones).

                st.dataframe(styler, width="stretch")
                st.caption(
                    "Coloured cells: % of expected reporting/availability with data. Grey cells: state "
                    "changes that day for sensors that only report on change (valve, "
                    "pumps, zones, modes) or rare events (defrost). These aren't scored, "
                    "because silence means the state didn't change, not that data is missing."
                )

        else:
            st.info("No count-based columns found in daily data.")





    # ------------------------------------------------------------------
    # TAB 4: Heartbeats
    # ------------------------------------------------------------------
    with dq_tab4:
        st.markdown("### ❤️ Sensor Heartbeats")
        import source_ui

        addon = source_ui.running_in_addon()
        if addon:
            st.caption(
                "How often each sensor normally reports and how long it is normally quiet. therm builds "
                "this automatically from any analysis of 28 days or more, saves it with your profile, "
                "and uses it for shorter periods, so short periods are judged against the long-run normal."
            )
        else:
            st.caption(
                "Generate this from a long analysis (ideally 3–12 months), download it, "
                "then load it from the sidebar before processing a shorter period. The "
                "short analysis is scored against the long-period raw reporting cadence."
            )

        # Try to load existing baseline into memory if not already
        baseline = st.session_state.get("heartbeat_baseline")

        if heartbeat_path and not baseline:
            try:
                baseline, loaded_from = baselines.load_saved_heartbeat_baseline(
                    heartbeat_path
                )
                st.session_state["heartbeat_baseline"] = baseline
                st.session_state["heartbeat_baseline_source"] = loaded_from
                st.info(f"Loaded existing baseline from: {loaded_from}")
            except Exception as e:
                st.warning(f"Could not load baseline file: {e}")
                baseline = None

        # --- Build / Rebuild button ---
        # Add-on: automatic (heartbeat_store); no manual build or file download.
        build_clicked = (not addon) and st.button("Generate / Rebuild Heartbeat Baseline")

        if build_clicked:
            history_df = raw_events
            if history_df is None or history_df.empty:
                history_df = st.session_state.get("raw_history_df")

            if history_df is None or history_df.empty:
                st.error(
                    "Raw long-form report history is unavailable, so a valid heartbeat "
                    "cannot be generated from this run."
                )
            else:
                with st.spinner("Building baselines..."):
                    if "is_mapped" in history_df.columns:
                        history_df = history_df[history_df["is_mapped"].fillna(False)]
                    new_bl = baselines.build_offline_aware_seasonal_baseline(
                        history_df, SENSOR_ROLES
                    )
                    st.session_state["heartbeat_baseline"] = new_bl
                    st.session_state["heartbeat_baseline_source"] = "generated from current analysis"
                    st.session_state["heartbeat_profile"] = (
                        st.session_state.get("system_config") or {}).get("profile_name")
                    dates = pd.to_datetime(history_df["last_changed"], errors="coerce").dropna()
                    days_analyzed = int(dates.dt.date.nunique()) if not dates.empty else 0
                    st.session_state["heartbeat_baseline_meta"] = {
                        "days_analyzed": days_analyzed,
                        "period_start": dates.min().isoformat() if not dates.empty else None,
                        "period_end": dates.max().isoformat() if not dates.empty else None,
                    }
                    baseline = new_bl
                if baseline:
                    # Reprocess immediately: the heartbeat is part of the data
                    # cache identity and must affect the current DQ tables, not
                    # only the next unrelated UI interaction.
                    st.rerun()
                else:
                    st.warning(
                        "No sensor had enough consistent history to form a baseline. "
                        "At least three comparable active days are required."
                    )

        if baseline:
            source = st.session_state.get("heartbeat_baseline_source", "current session")
            meta = st.session_state.get("heartbeat_baseline_meta", {}) or {}
            st.info(
                f"Active heartbeat: **{len(baseline)} sensors** from **{source}**"
                + (
                    f" ({meta.get('days_analyzed')} days analysed)"
                    if meta.get("days_analyzed") is not None
                    else ""
                )
            )
            # Show summary table
            rows = []
            for sensor_name, meta in baseline.items():
                mapped_entity = mapping.get(sensor_name)
                rows.append(
                    {
                        "Sensor": display_label_for(sensor_name),
                        "Mapped Entity": (
                            str(mapped_entity)
                            if mapped_entity else "—"
                        ),
                        "Role": meta.get("role", SENSOR_ROLES.get(sensor_name, "")),
                        "Expected Reports / Day": meta.get(
                            "expected_reports_per_day",
                            meta.get("expected_minutes", None),
                        ),
                        "Baseline Ready": bool(meta.get("has_baseline")),
                        "Reporting Day Coverage %": (
                            round(100 * float(meta.get("reporting_day_coverage")), 1)
                            if meta.get("reporting_day_coverage") is not None else None
                        ),
                        "Reason": meta.get("reason", ""),
                        "Mode": SENSOR_EXPECTATION_MODE.get(sensor_name, "system"),
                    }
                )
            hb_df = pd.DataFrame(rows).sort_values("Sensor")
            st.dataframe(hb_df, width="stretch")

            # Allow user to download to a location of their choice
            import json as _json

            raw_dates = pd.Series(dtype="datetime64[ns]")
            if raw_events is not None and not raw_events.empty:
                raw_dates = pd.to_datetime(raw_events["last_changed"], errors="coerce").dropna()
            days_analyzed = int(raw_dates.dt.date.nunique()) if not raw_dates.empty else int(
                (st.session_state.get("heartbeat_baseline_meta", {}) or {}).get("days_analyzed", 0)
            )
            payload = baselines.heartbeat_baseline_payload(
                baseline,
                tag="long_analysis",
                days_in_history=days_analyzed,
                profile_name=(st.session_state.get("system_config", {}) or {}).get("profile_name"),
                period_start=raw_dates.min().isoformat() if not raw_dates.empty else None,
                period_end=raw_dates.max().isoformat() if not raw_dates.empty else None,
            )
            if not addon:  # the add-on saves it with the profile automatically
                st.download_button(
                    label="💾 Download Heartbeat Baseline (JSON)",
                    data=_json.dumps(payload, indent=2),
                    file_name="therm_heartbeat_baseline.json",
                    mime="application/json",
                )

            # This period vs the heartbeat: informational, not scored.
            compare_rows = []
            for sensor_name, meta in baseline.items():
                expected = meta.get("expected_reports_per_day", meta.get("expected_minutes"))
                hb_col = f"HB_{sensor_name}_Count"
                if not expected or hb_col not in daily_df.columns:
                    continue
                this_period = float(daily_df[hb_col].mean())
                compare_rows.append({
                    "Sensor": display_label_for(sensor_name),
                    "Heartbeat Reports / Day": expected,
                    "This Period Reports / Day": round(this_period, 1),
                    "Ratio %": round(100.0 * this_period / expected, 0),
                    "Days With No Reports": int((daily_df[hb_col] == 0).sum()),
                })
            if compare_rows:
                st.markdown("### This Period vs Heartbeat")
                st.dataframe(
                    pd.DataFrame(compare_rows).sort_values("Ratio %"),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Most sensors report on change, so a low ratio in a quiet or "
                    "summer period is normal. Days with no reports for a sensor "
                    "that normally reports daily are worth checking. Availability "
                    "scores use the heartbeat through its gap limits: a silence "
                    "longer than the long-period normal leaves minutes empty."
                )
        else:
            st.info(
                "No heartbeat yet. therm builds one automatically the first time you analyse 28 days or more."
                if addon else
                "No heartbeat baseline loaded yet. Click the button above to generate one."
            )

        # Optional: show the pattern analysis table if available
        if patterns:
            st.markdown("### Detected Sensor Patterns")
            pat_data = []
            for sensor, details in patterns.items():
                if sensor not in mapped_keys:
                    continue
                pat_data.append(
                    {
                        "Sensor": display_label_for(sensor),
                        "Type": details["report_type"],
                        "Interval (s)": round(details["normal_interval_sec"], 1),
                        "Gap Limit (s)": round(details["gap_threshold_sec"], 1),
                    }
                )
            st.dataframe(pd.DataFrame(pat_data), width="stretch")

        if provenance_summary:
            st.markdown("### Observed vs Filled Data")
            provenance_rows = []
            for sensor, details in provenance_summary.items():
                if sensor not in mapped_keys:
                    continue
                provenance_rows.append({
                    "Sensor": display_label_for(sensor),
                    "Observed min": details.get("observed_minutes", 0),
                    "Interpolated min": details.get("interpolated_minutes", 0),
                    "Held min": details.get("held_minutes", 0),
                    "Missing min": details.get("missing_minutes", 0),
                    "Observed %": details.get("observed_pct", 0.0),
                    "Filled %": details.get("synthetic_pct", 0.0),
                })
            if provenance_rows:
                st.dataframe(
                    pd.DataFrame(provenance_rows).sort_values("Sensor"),
                    width="stretch",
                )
                st.caption(
                    "Observed minutes contain a source report in that minute. "
                    "Interpolated and held minutes carry a value between reports. "
                    "For on-change sensors (states, zones, valve) a held value is "
                    "the true state, so a high Held % is expected, not a fault."
                )

        energy_columns = {
            "WithinCadence_Input_Electricity_kWh": "Cadence — electricity within normal reporting gap",
            "ExtendedGap_Input_Electricity_kWh": "Cadence — electricity across an extended gap",
            "WithinCadence_Input_Heat_kWh": "Cadence — heat within normal reporting gaps",
            "ExtendedGap_Input_Heat_kWh": "Cadence — heat across an extended gap",
            "Observed_Input_Electricity_kWh": "Strict — electricity from an observed Power minute",
            "Estimated_Input_Electricity_kWh": "Strict — electricity from an interpolated/held input",
            "Observed_Input_Heat_kWh": "Strict — heat from observed source-input minutes",
            "Estimated_Input_Heat_kWh": "Strict — heat from interpolated/held source inputs",
        }
        available_energy = {
            column: label for column, label in energy_columns.items()
            if column in daily_df.columns
        }
        if available_energy:
            st.markdown("### Energy affected by data gaps")
            energy_rows = [
                {"Energy basis": label, "kWh": round(float(daily_df[column].sum()), 3)}
                for column, label in available_energy.items()
            ]
            st.dataframe(pd.DataFrame(energy_rows), width="stretch", hide_index=True)
            st.caption(
                "These companion totals do not change the existing calculations. "
                "The cadence split distinguishes normal sample-and-hold behaviour "
                "from extended gaps. The strict split calls only a source report in "
                "that exact minute observed; interpolation and holds are estimated. "
                "For change-driven sensors, a high strict estimated share can be "
                "normal. Immersion and cooling are outside these splits."
            )


    # ------------------------------------------------------------------
    # TAB 5: Unmapped Data
    # ------------------------------------------------------------------
    with dq_tab5:
        st.markdown("### Unmapped / Dropped Entities")

        if not unmapped_entities:
            st.info("No unmapped entities found in the source files.")
        else:
            st.write(
                "The following entities were present in the upload but not "
                "mapped to any internal sensor role:"
            )
            st.dataframe(
                pd.DataFrame(
                    sorted(set(unmapped_entities)), columns=["Entity ID"]
                ),
                width="stretch",
            )
