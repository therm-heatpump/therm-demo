# view_runs.py - Fixed zone and room naming



import streamlit as st

import charts

import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import json

import numpy as np

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo



from config import (
    CALC_VERSION,
    THRESHOLDS,
    AI_RUN_METRIC_DEFINITIONS,
    build_ai_system_context,
    cop_provenance,
    physics_assumptions,
)

from utils import safe_div, strip_entity_prefix

import processing







def _build_system_context(user_config: dict | None, include_heating_note: bool) -> str:
    """Compose AI context from the default prompt plus user-provided freetext."""
    parts: list[str] = []

    base = build_ai_system_context("SINGLE_RUN_INSPECTOR").strip()
    if base:
        parts.append(base)

    if isinstance(user_config, dict):
        ai_ctx = user_config.get("ai_context") or {}
        for key in ("hp_model", "property_context", "operational_goals"):
            val = ai_ctx.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        tariff = user_config.get("tariff_structure")
        currency = user_config.get("currency", "€")
        if isinstance(tariff, dict):
            day = tariff.get("day_rate")
            night = tariff.get("night_rate", day)
            parts.append(f"Tariff: day/night rates {currency}{day} / {currency}{night}.")
        elif isinstance(tariff, list) and tariff:
            rules = tariff[0].get("rules", [])
            if rules:
                summary = "; ".join(
                    f"{r.get('name','')}: {r.get('start','')}-{r.get('end','')} @ {currency}{r.get('rate','')}"
                    for r in rules
                )
                parts.append(f"Tariff bands: {summary}")
    if include_heating_note:
        parts.append(
            "Heating during DHW detected: zone pumps active during DHW can cause return mixing and low COP; attribute DHW efficiency penalties accordingly."
        )
    return "\n".join(parts) if parts else "No additional system context supplied."

def _get_friendly_name(internal_key: str, user_config: dict) -> str:

    """

    Get the friendly display name for a zone or room.

    

    For Zone_1, Zone_2, etc. or Room_1, Room_2, etc.:

    - Returns the mapped entity_id from user config

    - Strips "binary_sensor." and "sensor." prefixes for cleaner display

    - Falls back to the internal key if not mapped

    

    Args:

        internal_key: Internal name like "Zone_1" or "Room_3"

        user_config: User's system configuration dict

        

    Returns:

        Friendly name string (cleaned entity_id or internal key)

    """

    if not isinstance(user_config, dict):

        return str(internal_key)

    

    mapping = user_config.get("mapping", {})

    if not isinstance(mapping, dict):

        return str(internal_key)



    # Get the mapped entity_id and strip HA/Grafana prefixes for display

    entity_id = str(mapping.get(internal_key, internal_key))

    return strip_entity_prefix(entity_id)





def _get_rooms_per_zone_config(user_config: dict) -> dict:

    """

    Get the user's rooms_per_zone configuration.

    

    Returns:

        dict mapping zone keys (e.g., "Zone_1") to lists of room keys (e.g., ["Room_3"])

    """

    if not isinstance(user_config, dict):

        return {}

    

    return user_config.get("rooms_per_zone", {})


def build_run_ai_payload(selected_run: dict, run_data: pd.DataFrame, user_config=None) -> dict:
    """Build the SINGLE_RUN_INSPECTOR payload without Streamlit state."""
    cfg = user_config if isinstance(user_config, dict) else {}
    run_date_str = str(pd.Timestamp(selected_run["start"]).date())
    active_tag = "baseline_v1"
    active_note = "Initial commissioning."
    for entry in sorted(cfg.get("config_history", []), key=lambda x: str(x.get("start", ""))):
        if run_date_str >= str(entry.get("start", "")):
            active_tag = entry.get("config_tag", active_tag)
            active_note = entry.get("change_note", "")
        else:
            break

    tariff_name = "Unknown"
    tariff = cfg.get("tariff_structure")
    if isinstance(tariff, list) and tariff:
        tariff_name = tariff[0].get("name", "Custom")
    elif isinstance(tariff, dict):
        tariff_name = (
            "Flat Rate"
            if tariff.get("day_rate") == tariff.get("night_rate", tariff.get("day_rate"))
            else "Day/Night"
        )

    total_rows = len(run_data)
    freq = pd.to_numeric(run_data.get("Freq", pd.Series(index=run_data.index, dtype=float)), errors="coerce")
    pct_low_hz = float((freq < 25).sum() / total_rows * 100) if total_rows else 0.0
    pct_high_hz = float((freq > 45).sum() / total_rows * 100) if total_rows else 0.0
    avg_hz = float(freq.mean()) if freq.notna().any() else 0.0
    flow = pd.to_numeric(run_data.get("FlowTemp", pd.Series(index=run_data.index, dtype=float)), errors="coerce")

    target_flow = None
    target_flow_source = None
    flow_limited_mins = None
    flow_limit_input_coverage = None
    if selected_run.get("run_type") == "DHW":
        target_flow = 50.0
        target_flow_source = "fixed DHW assumption (50 degC)"
    else:
        target_series, target_flow_source = processing.target_flow_series(run_data, cfg)
        if target_series is not None and target_series.notna().any():
            target_flow = float(target_series.mean())
        counted, valid, eligible, _ = processing.flow_limit_analysis(run_data, cfg)
        eligible_mins = int(eligible.sum())
        if eligible_mins:
            flow_limit_input_coverage = float(valid.sum() / eligible_mins)
        if counted is not None and (
            eligible_mins == 0 or flow_limit_input_coverage >= 0.90
        ):
            flow_limited_mins = int(counted.sum())

    run_cop = selected_run.get("run_cop")
    payload = {
        "meta": {
            "report_type": "SINGLE_RUN_INSPECTOR",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "calc_version": CALC_VERSION,
            "run_id": selected_run.get("id"),
            "timestamp_start": str(selected_run.get("start")),
            "timestamp_end": str(selected_run.get("end")),
            "duration_minutes": selected_run.get("duration_mins"),
            "run_type": selected_run.get("run_type"),
            "active_zones": selected_run.get("dominant_zones"),
        },
        "system_context": _build_system_context(
            cfg, bool(selected_run.get("heating_during_dhw_detected"))
        ),
        "metric_definitions": AI_RUN_METRIC_DEFINITIONS,
        "config_history": cfg.get("config_history", []),
        "configuration_state": {
            "active_profile_tag": active_tag,
            "change_note_if_new": active_note,
            "tariff_profile": tariff_name,
        },
        "physics_assumptions": physics_assumptions(cfg, run_data.attrs.get("heat_source")),
        "cop_provenance": cop_provenance(cfg, run_data.attrs.get("heat_source")),
        "economics": {
            "run_cost_euro": round(float(run_data.get("Cost_Inc", pd.Series(dtype=float)).sum()), 3),
            "kwh_electricity": round(float(selected_run.get("electricity_kwh", 0) or 0), 2),
            "kwh_heat": round(float(selected_run.get("heat_kwh", 0) or 0), 2),
            "effective_cop": round(float(run_cop or 0), 2),
            "run_cop": round(float(run_cop or 0), 2),
            "cost_per_kwh_heat_euro": (
                round(float(selected_run["cost_per_kwh_heat"]), 3)
                if selected_run.get("cost_per_kwh_heat") is not None else None
            ),
            "immersion_kwh_estimated": selected_run.get("immersion_kwh", 0),
            "immersion_was_active": bool(selected_run.get("immersion_detected", False)),
            "immersion_active_minutes": selected_run.get("immersion_mins", 0),
        },
        "heat_accounting": {
            "kwh_heat_gross": round(float(selected_run.get("heat_gross_kwh", 0) or 0), 3),
            "defrost_count": int(selected_run.get("defrost_count", 0) or 0),
            "defrost_minutes": int(selected_run.get("defrost_mins", 0) or 0),
            "defrost_heat_loss_kwh": round(float(selected_run.get("defrost_heat_loss_kwh", 0) or 0), 3),
            "transient_heat_loss_kwh": round(float(selected_run.get("transient_heat_loss_kwh", 0) or 0), 3),
        },
        "diagnostics_physics": {
            "avg_flow_rate_lpm": round(float(selected_run.get("avg_flow_rate", 0) or 0), 1),
            "avg_delta_t": round(float(selected_run.get("avg_dt", 0) or 0), 2),
            "avg_flow_temp_c": round(float(flow.mean()), 1) if flow.notna().any() else None,
            "max_flow_temp_c": round(float(flow.max()), 1) if flow.notna().any() else None,
            "avg_return_temp_c": selected_run.get("avg_return_temp"),
            "min_return_temp_c": selected_run.get("min_return_temp"),
            "return_temp_range_c": selected_run.get("return_temp_range"),
            "target_flow_temp_avg": round(target_flow, 1) if target_flow is not None else None,
            "target_flow_source": target_flow_source,
            "flow_limited_mins": flow_limited_mins,
            "flow_limit_input_coverage": flow_limit_input_coverage,
            "compressor_stats": {
                "avg_hz": round(avg_hz, 1),
                "min_hz": selected_run.get("min_freq"),
                "max_hz": selected_run.get("max_freq"),
                "std_dev_hz": selected_run.get("freq_std_dev"),
                "pct_time_low_modulation (<25Hz)": round(pct_low_hz, 1),
                "pct_time_high_modulation (>45Hz)": round(pct_high_hz, 1),
            },
        },
        "control_modes": {
            "dhw_mode_value": selected_run.get("dhw_mode"),
            "quiet_mode_active": selected_run.get("quiet_mode_active", False),
        },
        "run_characteristics": {
            "is_short_cycle": selected_run.get("is_short_cycle", False),
            "short_cycle_threshold_minutes": processing.resolve_thresholds(cfg).get("short_cycle_min", 20),
        },
        "environmental_conditions": {
            "outdoor_temp_avg": (
                round(float(pd.to_numeric(run_data["OutdoorTemp"], errors="coerce").mean()), 1)
                if "OutdoorTemp" in run_data and run_data["OutdoorTemp"].notna().any() else None
            ),
            "outdoor_temp_min": selected_run.get("outdoor_temp_min"),
            "outdoor_temp_max": selected_run.get("outdoor_temp_max"),
            "outdoor_temp_change_c": selected_run.get("outdoor_temp_change"),
            "outdoor_humidity_avg": (
                round(float(pd.to_numeric(run_data["Outdoor_Humidity"], errors="coerce").mean()), 1)
                if "Outdoor_Humidity" in run_data and run_data["Outdoor_Humidity"].notna().any() else None
            ),
            "wind_speed_avg": (
                round(float(pd.to_numeric(run_data["Wind_Speed"], errors="coerce").mean()), 1)
                if "Wind_Speed" in run_data and run_data["Wind_Speed"].notna().any() else None
            ),
        },
    }
    if selected_run.get("run_type") in ("Heating", "Cooling"):
        payload["room_response_deltas"] = selected_run.get("room_deltas", {})
    if selected_run.get("run_type") == "Cooling":
        payload["cooling"] = {
            "heat_removed_kwh": round(float(selected_run.get("heat_kwh", 0) or 0), 3),
            "eer": selected_run.get("eer"),
        }
    if selected_run.get("run_type") == "DHW":
        payload["dhw_temperature_profile"] = {
            "start_c": selected_run.get("dhw_temp_start"),
            "end_c": selected_run.get("dhw_temp_end"),
            "rise_c": selected_run.get("dhw_rise"),
            "stratification_range_c": selected_run.get("dhw_stratification_range"),
        }
        payload["hydraulic_integrity"] = {
            "heating_during_dhw_detected": selected_run.get("heating_during_dhw_detected"),
            "heating_during_dhw_pct": selected_run.get("heating_during_dhw_pct", 0.0),
        }
    return payload





_FINDER_METRICS = {
    # label: (column, higher is better)
    "COP": ("COP", True),
    "Cost per kWh of heat": ("Cost / kWh heat", False),
    "Cost per kWh of heat at today's prices": ("Cost / kWh heat (today's prices)", False),
}
_FINDER_BANDS = {
    "All runs": None,
    "Best 10%": ("best", 0.10), "Best 25%": ("best", 0.25), "Best 50%": ("best", 0.50),
    "Worst 50%": ("worst", 0.50), "Worst 25%": ("worst", 0.25), "Worst 10%": ("worst", 0.10),
}
_SHORT_RUN_MINS = 10


def _runs_table(df, runs_list, tariff_structure, as_of=None) -> pd.DataFrame:
    """One row per run (newest first, `#` = position in runs_list) with COP, the cost
    of its heat at the prices of the day, and at today's prices (comparable across
    price changes: processing.run_costs_at_current_prices)."""
    import processing

    if hasattr(df, "run_costs_at_current_prices"):  # month-by-month analysis (chunked.MonthFrames)
        at_today = df.run_costs_at_current_prices(runs_list, tariff_structure, as_of=as_of)
    else:
        at_today = processing.run_costs_at_current_prices(df, runs_list, tariff_structure, as_of=as_of)
    rows = []
    for i, (r, cost_today) in enumerate(zip(runs_list, at_today)):
        heat = float(r.get("heat_kwh") or 0.0)
        start = r["start"]
        rows.append({
            "#": i,
            "Start": start.tz_localize(None) if getattr(start, "tzinfo", None) else start,
            "Type": {"DHW": "Hot water"}.get(r["run_type"], r["run_type"]),
            "Minutes": int(r.get("duration_mins") or 0),
            "Heat (kWh)": round(heat, 2),
            "COP": r.get("run_cop"),
            "Cost / kWh heat": r.get("cost_per_kwh_heat"),
            "Cost / kWh heat (today's prices)": (cost_today / heat) if cost_today is not None and heat > 0.05 else None,
            "Outdoor (°C)": r.get("avg_outdoor"),
            "Flow (°C)": r.get("avg_flow_temp"),
        })
    return pd.DataFrame(rows)


def _render_run_finder(df, runs_list, user_config) -> None:
    """
    'Find runs to investigate': every run with its COP and cost of heat, sortable, with
    best / worst 10-25-50 % filters by COP or by cost per kWh of heat (also at today's
    prices, so runs from different price periods compare by behaviour). Selecting a
    row opens that run below.

    Collapsed by default, and the table is built only while it is open: pricing every run at
    today's prices reads every saved month of a long InfluxDB analysis.
    """
    finder = st.expander("🔎 Find runs to investigate", expanded=False, key="finder_open", on_change="rerun")
    if not finder.open:
        return
    tariff = (user_config or {}).get("tariff_structure")
    currency = (user_config or {}).get("currency", "€")
    try:
        today = datetime.now(ZoneInfo((user_config or {}).get("timezone", "Europe/Dublin"))).date()
    except (KeyError, TypeError, ValueError):
        today = date.today()
    # ``runs_list`` is sorted into a new list on every fragment rerun, so its id is
    # not a stable cache key. The small run signature remains stable while filters
    # and row selection change, but invalidates if any displayed metric changes.
    run_signature = json.dumps([
        {
            "start": r.get("start"), "end": r.get("end"),
            "heat": r.get("heat_kwh"), "cop": r.get("run_cop"),
            "cost": r.get("cost_per_kwh_heat"),
        }
        for r in runs_list
    ], sort_keys=True, default=str)
    cache_key = (id(df), run_signature, json.dumps(tariff, sort_keys=True, default=str), today.isoformat())
    cached = st.session_state.get("_runs_table_cache")
    if not cached or cached[0] != cache_key:
        cached = (cache_key, _runs_table(df, runs_list, tariff, as_of=today))
        st.session_state["_runs_table_cache"] = cached
    table = cached[1]

    with finder:
        c_kind, c_metric, c_band, c_short = st.columns([2, 2.2, 1.3, 1.3], vertical_alignment="bottom")
        with c_kind:
            kind = st.segmented_control("Runs", ["All", "Heating", "Hot water"], default="All",
                                        key="finder_kind") or "All"
        with c_metric:
            metric = st.selectbox("Rank by", list(_FINDER_METRICS), key="finder_metric",
                                  help="Cost per kWh of heat combines efficiency and when the run happened. "
                                       "'At today's prices' prices every run with your current tariff, so "
                                       "runs from before and after a price change can be compared.")
        with c_band:
            band = st.selectbox("Show", list(_FINDER_BANDS), key="finder_band")
        with c_short:
            skip_short = st.checkbox("Skip short runs", value=True, key="finder_skip_short",
                                     help=f"Leave out runs under {_SHORT_RUN_MINS} minutes: very short runs give "
                                          "extreme COP and cost figures.")

        view = table
        if kind != "All":
            view = view[view["Type"] == kind]
        if skip_short:
            view = view[view["Minutes"] >= _SHORT_RUN_MINS]
        column, higher_better = _FINDER_METRICS[metric]
        ranked = view.dropna(subset=[column]).sort_values(column, ascending=not higher_better, kind="stable")
        chosen = _FINDER_BANDS[band]
        if chosen:
            which, share = chosen
            n = max(1, int(np.ceil(len(ranked) * share))) if len(ranked) else 0
            ranked = ranked.head(n) if which == "best" else ranked.tail(n).iloc[::-1]
        if ranked.empty:
            st.info("No runs match these filters.")
            return
        st.caption(
            f"{len(ranked)} run{'s' if len(ranked) != 1 else ''} · median COP {ranked['COP'].median():.2f}"
            + (f" · median cost {currency}{ranked['Cost / kWh heat'].median():.3f} per kWh of heat"
               if ranked["Cost / kWh heat"].notna().any() else "")
            + ". Select a row to open that run below."
        )
        money = f"{currency}%.3f"
        event = st.dataframe(
            ranked,
            hide_index=True,
            height=min(38 + 35 * len(ranked), 300),
            on_select="rerun",
            selection_mode="single-row",
            key="finder_table",
            column_order=[c for c in ranked.columns if c != "#"],
            column_config={
                "Start": st.column_config.DatetimeColumn("Start", format="DD/MM/YYYY HH:mm"),
                "COP": st.column_config.NumberColumn("COP", format="%.2f"),
                "Cost / kWh heat": st.column_config.NumberColumn("Cost / kWh heat", format=money),
                "Cost / kWh heat (today's prices)": st.column_config.NumberColumn(
                    "Cost / kWh heat (today's prices)", format=money),
                "Outdoor (°C)": st.column_config.NumberColumn(format="%.1f"),
                "Flow (°C)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        rows = event.selection.rows if event and hasattr(event, "selection") else []
        picked = int(ranked.iloc[rows[0]]["#"]) if rows else None
        if picked is not None and picked != st.session_state.get("_finder_last_pick"):
            st.session_state["_finder_last_pick"] = picked
            st.session_state["run_selector_idx"] = picked


@st.fragment
def render_run_inspector(df, runs_list):

    """

    Run Inspector:

    - Sidebar global stats

    - Run picker with Previous/Next navigation

    - Selectable detail sections:

        ⚡ Efficiency

        Hydraulics (incl. Heating during DHW)
        Rooms

        AI Data (per-run JSON payload)

    """

    st.title(" Run Inspector")

    

    # Get user config from session state for friendly names

    user_config = st.session_state.get("system_config", {})

    mapping = user_config.get("mapping", {}) if isinstance(user_config, dict) else {}

    has_zones_mapped = any(k.startswith("Zone_") for k in mapping)



    if not runs_list:

        st.info("No runs detected.")

        return



    runs_list = sorted(runs_list, key=lambda x: x["start"], reverse=True)

    # Run list with COP / cost and best-worst filters; picking a row selects the run.
    _render_run_finder(df, runs_list, user_config)



    # --- Run Selection Logic ---

    run_options = {}

    for r in runs_list:

        start_str = r["start"].strftime("%d/%m/%Y %H:%M")



        # Only zone/pump signals can show heating during DHW (processing.detect_runs).
        heating_during_dhw = bool(r.get("heating_during_dhw_detected"))

        if r["run_type"] == "DHW":
            # Use a water drop to represent DHW runs
            icon = "💧"
        elif r["run_type"] == "Cooling":
            icon = "❄️"
        else:
            icon = "🔥"



        zone_raw = r.get("active_zones", r.get("dominant_zones", "None"))



        # For DHW runs, hide zone info unless heating during DHW was detected.

        show_zone = (

            has_zones_mapped

            and (

                r["run_type"] != "DHW"

                or r.get("heating_during_dhw_detected")

            )

        )

        if show_zone:
            zone_label = zone_raw if zone_raw and str(zone_raw).lower() != "none" else "No Zone Data"
        else:
            zone_label = ""

        # Avoid redundant "(DHW)" suffix when the run itself is DHW
        if heating_during_dhw and r["run_type"] == "DHW":
            if zone_label.strip().lower() == "dhw":
                zone_label = ""

        label = f"{start_str} | {r['duration_mins']}m | {icon} {r['run_type']}"
        if heating_during_dhw and r["run_type"] == "DHW":
            label = f"{label} (Heating Active❗)"
        if zone_label:
            label = f"{label} ({zone_label})"

        run_options[label] = r



    option_labels = list(run_options.keys())

    if "run_selector_idx" not in st.session_state:

        st.session_state["run_selector_idx"] = 0



    st.session_state["run_selector_idx"] = min(

        st.session_state["run_selector_idx"], len(option_labels) - 1

    )



    nav_prev, nav_select, nav_next = st.columns([1, 4, 1])



    with nav_prev:

        if st.button(

            "← Previous", disabled=st.session_state["run_selector_idx"] <= 0

        ):

            st.session_state["run_selector_idx"] = max(

                0, st.session_state["run_selector_idx"] - 1

            )



    with nav_next:

        if st.button(

            "Next →",

            disabled=st.session_state["run_selector_idx"] >= len(option_labels) - 1,

        ):

            st.session_state["run_selector_idx"] = min(

                len(option_labels) - 1, st.session_state["run_selector_idx"] + 1

            )



    with nav_select:

        selected_label = st.selectbox(

            "Select Run",

            options=option_labels,

            index=st.session_state["run_selector_idx"],

        )

        st.session_state["run_selector_idx"] = option_labels.index(selected_label)



    selected_run = run_options[selected_label]

    # A month-by-month analysis (chunked.MonthFrames) loads only this run's month(s).
    run_data = (df.slice(selected_run["start"], selected_run["end"]) if hasattr(df, "slice")
                else df.loc[selected_run["start"] : selected_run["end"]])
    if getattr(df, "missing_months", None):
        st.warning("Some months of this analysis are no longer saved (only the most recent months are kept). "
                   "Press **Load latest data** to process them again.")



    # --- Top-level stats ---

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Duration", f"{selected_run['duration_mins']}m")

    # Cooling runs report EER (heat removed / electricity) instead of COP
    c2.metric(
        "EER" if selected_run["run_type"] == "Cooling" else "COP",
        f"{selected_run['run_cop']:.2f}",
    )
    provenance = cop_provenance(user_config, run_data.attrs.get("heat_source"))
    efficiency_label = "EER basis" if selected_run["run_type"] == "Cooling" else "COP basis"
    st.caption(
        f"{efficiency_label} (primary Power): {provenance.get('display', 'Not specified')}. "
        "This does not describe optional Indoor Power."
    )

    c3.metric("Avg ΔT", f"{selected_run['avg_dt']:.1f} °C")

    avg_flow_lpm = selected_run.get("avg_flow_rate", 0)

    c4.metric("Avg Flow", f"{avg_flow_lpm:.1f} L/m")



    tight_layout = dict(margin=dict(l=10, r=10, t=30, b=10), height=350)



    # Render one detail section at a time. Unlike st.tabs, this avoids building
    # and transferring every inactive chart on each run-navigation click.
    # Rooms only where room temperatures are analysed (not for hot-water runs, where
    # the tab only said it was skipped).
    sections = ["Efficiency", "Hydraulics", "AI Data"] if selected_run["run_type"] == "DHW" \
        else ["Efficiency", "Hydraulics", "Rooms", "AI Data"]
    if st.session_state.get("run_detail_section") not in sections + [None]:
        del st.session_state["run_detail_section"]  # back to the default (Efficiency)
    detail_section = st.segmented_control(
        "Run detail section",
        sections,
        default="Efficiency",
        key="run_detail_section",
        label_visibility="collapsed",
    ) or "Efficiency"  # clicking the selected option deselects it (None)



    # ------------------------------------------------------------------

    # SECTION 1: Efficiency

    # ------------------------------------------------------------------

    if detail_section == "Efficiency":

        fig = make_subplots(specs=[[{"secondary_y": True}]])



        fig.add_trace(

            go.Scatter(

                x=run_data.index,

                y=run_data.get("Heat_Clean", run_data["Heat"]),

                name="Heat",

                fill="tozeroy",

                line=dict(color="orange", width=0),

            ),

            secondary_y=False,

        )



        fig.add_trace(

            go.Scatter(

                x=run_data.index,

                y=run_data.get("Power_Clean", run_data["Power"]),

                name="Power",

                line=dict(color="red", width=1),

            ),

            secondary_y=False,

        )



        if "Indoor_Power" in run_data.columns:

            fig.add_trace(

                go.Scatter(

                    x=run_data.index,

                    y=run_data["Indoor_Power"],

                    name="Indoor",

                    line=dict(color="purple", width=1, dash="dot"),

                ),

                secondary_y=False,

            )



        if "COP_Graph" in run_data.columns:

            fig.add_trace(

                go.Scatter(

                    x=run_data.index,

                    y=run_data["COP_Graph"],

                    name="COP",

                    line=dict(color="blue", dash="dot", width=1),

                ),

                secondary_y=True,

            )



        # Shade defrost episodes
        if "is_defrost" in run_data.columns and run_data["is_defrost"].any():
            on = run_data["is_defrost"].astype(bool)
            starts = run_data.index[on & ~on.shift(fill_value=False)]
            ends = run_data.index[on & ~on.shift(-1, fill_value=False)]
            for i, (s, e) in enumerate(zip(starts, ends)):
                fig.add_vrect(
                    x0=s, x1=e + pd.Timedelta(minutes=1),
                    fillcolor="lightskyblue", opacity=0.35, line_width=0, layer="below",
                    annotation_text="Defrost" if i == 0 else None,
                    annotation_position="top left",
                )

        fig.update_layout(

            **tight_layout,

            title="Power & Efficiency",

            hovermode="x unified",

        )

        # Readable axes (not 2.179 / 1.447 / -0.017 on the COP axis).
        fig.update_yaxes(title_text="Power / heat (W)", rangemode="tozero", secondary_y=False)
        if "COP_Graph" in run_data.columns:
            cop_vals = pd.to_numeric(run_data["COP_Graph"], errors="coerce")
            cop_top = float(np.ceil(cop_vals.max())) if cop_vals.notna().any() else 5.0
            fig.update_yaxes(title_text="COP", range=[0, max(cop_top, 1.0)], dtick=1,
                             showgrid=False, secondary_y=True)

        charts.show(fig, key="run_power_chart")

        # Net heat accounting for this run
        d_loss = selected_run.get("defrost_heat_loss_kwh", 0.0) or 0.0
        t_loss = selected_run.get("transient_heat_loss_kwh", 0.0) or 0.0
        n_def = selected_run.get("defrost_count", 0) or 0
        if d_loss > 0 or t_loss > 0:
            gross = selected_run.get("heat_gross_kwh", 0.0) or 0.0
            elec = selected_run.get("electricity_kwh", 0.0) or 0.0
            gross_cop = safe_div(gross, elec)
            st.caption(
                f"COP is **net** of heat taken back: {d_loss:.2f} kWh in {n_def} "
                f"defrost{'s' if n_def != 1 else ''}, {t_loss:.2f} kWh at start/stop "
                f"(gross COP would be {gross_cop:.2f})."
            )



    # ------------------------------------------------------------------

    # SECTION 2: Hydraulics (incl. Heating during DHW)
    # ------------------------------------------------------------------

    if detail_section == "Hydraulics":

        if selected_run["run_type"] == "DHW":

            # Shown only with zone signals and when detected; nothing is said otherwise.
            if (selected_run.get("heating_during_dhw_detection_source") == "zones"
                    and selected_run.get("heating_during_dhw_detected")):
                st.markdown("**Heating during DHW:** ⚠️ **Detected** (zone pumps running during hot water)")
                st.caption("⚠️ *A heating zone drew heat while the cylinder was being heated. This reduces DHW "
                           "efficiency.*")
                st.divider()

        is_dhw = selected_run["run_type"] == "DHW"

        # ΔT chart (own legend)

        if "DeltaT" in run_data.columns:

            fig_dt = go.Figure()

            fig_dt.add_trace(

                go.Scatter(

                    x=run_data.index,

                    y=run_data["DeltaT"],

                    name="ΔT",

                    line=dict(color="green"),

                )

            )

            fig_dt.add_hline(y=5.0, line_dash="dash", line_color="red")

            fig_dt.update_layout(

                title="Delta T",

                margin=dict(l=10, r=10, t=30, b=10),

                height=200,

                hovermode="x unified",

                showlegend=True,

                legend=dict(

                    orientation="h",

                    yanchor="top",

                    y=-0.12,

                    xanchor="left",

                    x=0,

                ),

            )

            charts.show(fig_dt, key="run_hydro_dt")



        # Flow Rate chart (own legend)

        if "FlowRate" in run_data.columns:

            fig_flow = go.Figure()

            fig_flow.add_trace(

                go.Scatter(

                    x=run_data.index,

                    y=run_data["FlowRate"],

                    name="Flow",

                    line=dict(color="cyan"),

                )

            )

            fig_flow.update_layout(
                title="Flow Rate",
                margin=dict(l=10, r=10, t=30, b=80),
                height=200,
                hovermode="x unified",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.35,
                    xanchor="left",
                    x=0,
                ),
            )

            charts.show(fig_flow, key="run_hydro_flow")



        # Active Zones chart (own legend)

        zone_cols = [

            c for c in run_data.columns 

            if c.startswith("Zone_") and c != "Zone_Config"

        ]

        has_dhw = "DHW_Active" in run_data.columns



        if zone_cols:

            fig_zones = go.Figure()



            # Build friendly zone labels from user mapping

            zone_labels = {}

            if has_dhw:

                zone_labels["DHW_Active"] = "Hot Water"

            for z in zone_cols:

                zone_labels[z] = _get_friendly_name(z, user_config)



            ordered_keys = []

            if has_dhw:

                ordered_keys.append("DHW_Active")

            ordered_keys.extend(sorted(zone_cols))

            zone_offsets = {key: idx for idx, key in enumerate(ordered_keys)}



            for key in ordered_keys:

                if key not in run_data.columns:

                    continue

                base_y = zone_offsets[key]



                def _zone_active(val):

                    """Check if zone is active, handling both numeric and string values."""

                    if pd.isna(val):

                        return None

                    if isinstance(val, str):

                        v = val.strip().lower()

                        is_active = v in ("on", "true", "1", "yes", "active")

                        return base_y + 0.8 if is_active else None

                    try:

                        is_active = float(val) > 0

                        return base_y + 0.8 if is_active else None

                    except:

                        return None



                y_vals = run_data[key].apply(_zone_active)

                fig_zones.add_trace(

                    go.Scatter(

                        x=run_data.index,

                        y=y_vals,

                        name=zone_labels[key],

                        mode="lines",

                        line=dict(width=15),

                        connectgaps=False,

                    )

                )



            y_tick_vals = [zone_offsets[key] + 0.4 for key in ordered_keys]

            y_tick_labels = [zone_labels[key] for key in ordered_keys]

            fig_zones.update_yaxes(

                tickvals=y_tick_vals,

                ticktext=y_tick_labels,

                range=[0, len(ordered_keys)],

            )

            fig_zones.update_layout(
                title="Active Zones",
                margin=dict(l=10, r=10, t=30, b=80),
                height=max(220, 140 + 20 * len(ordered_keys)),
                hovermode="x unified",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.35,
                    xanchor="left",
                    x=0,
                ),
            )

            charts.show(fig_zones, key="run_hydro_zones")



        # DHW / Return temps chart (own legend)

        if is_dhw and ("DHW_Temp" in run_data.columns or "ReturnTemp" in run_data.columns):

            fig_temp = go.Figure()

            if "DHW_Temp" in run_data.columns:

                fig_temp.add_trace(

                    go.Scatter(

                        x=run_data.index,

                        y=run_data["DHW_Temp"],

                        name="DHW Tank",

                        line=dict(color="orange", width=2),

                    )

                )

            if "ReturnTemp" in run_data.columns:

                fig_temp.add_trace(

                    go.Scatter(

                        x=run_data.index,

                        y=run_data["ReturnTemp"],

                        name="Return",

                        line=dict(color="grey", width=1, dash="dot"),

                    )

                )

            fig_temp.update_layout(

                title="Hot Water / Return Temps",

                margin=dict(l=10, r=10, t=30, b=100),

                height=220,

                hovermode="x unified",

                showlegend=True,

                legend=dict(

                    orientation="h",

                    yanchor="top",

                    y=-0.36,

                    xanchor="left",

                    x=0,

                ),

            )

            charts.show(fig_temp, key="run_hydro_temps")



    # ------------------------------------------------------------------

    # SECTION 3: Rooms

    # ------------------------------------------------------------------

    if detail_section == "Rooms":

        if selected_run["run_type"] in ("Heating", "Cooling"):

            fig3 = go.Figure()



            # Dynamically detect zones (exclude Zone_Config which is a string column)

            detected_zones = [

                z for z in run_data.columns 

                if z.startswith("Zone_") and z != "Zone_Config"

            ]



            # Which zones were active?

            active_zones = [

                z for z in detected_zones if run_data[z].sum() > 0

            ]



            # ================================================================

            #   FIXED: USE USER'S rooms_per_zone CONFIGURATION

            # ================================================================

            rooms_per_zone = _get_rooms_per_zone_config(user_config)

            

            # Build allowed rooms from user's configuration

            allowed_rooms = set()

            

            if rooms_per_zone:

                # Use user's explicit zone → rooms mapping

                for z in active_zones:

                    rooms_in_zone = rooms_per_zone.get(z, [])

                    allowed_rooms.update(rooms_in_zone)

            

            # If no explicit mapping or empty result, show all rooms

            if not allowed_rooms:

                allowed_rooms = set([

                    c for c in run_data.columns if c.startswith("Room_")

                ])



            room_cols = [

                c for c in run_data.columns

                if c.startswith("Room_") and c in allowed_rooms

            ]



            deltas = selected_run.get("room_deltas", {}) or {}



            # ================================================================

            #   FIXED: USE FRIENDLY ROOM NAMES FROM USER MAPPING

            # ================================================================

            for col in room_cols:

                # Get friendly name from user mapping (e.g., entity_id)

                friendly_name = _get_friendly_name(col, user_config)

                

                # Check if this room is relevant (has delta data)

                is_relevant = col in deltas



                fig3.add_trace(

                    go.Scatter(

                        x=run_data.index,

                        y=run_data[col],

                        name=(f"* {friendly_name}" if is_relevant else friendly_name),

                        mode="lines+markers",

                        line=dict(width=3 if is_relevant else 1),

                        opacity=1.0 if is_relevant else 0.5,

                    )

                )



            if "OutdoorTemp" in run_data.columns:

                fig3.add_trace(

                    go.Scatter(

                        x=run_data.index,

                        y=run_data["OutdoorTemp"],

                        name="Outdoor",

                        line=dict(color="grey", width=2, dash="dash"),

                        yaxis="y2",

                    )

                )



            fig3.update_layout(

                title="Temperature Response",

                hovermode="x unified",

                yaxis2=dict(

                    title="Outdoor",

                    overlaying="y",

                    side="right",

                    showgrid=False,

                ),

                height=350,

            )

            charts.show(fig3, key="run_rooms_chart")

        else:

            st.info("Room temperature analysis is skipped for DHW runs.")



    # ------------------------------------------------------------------

    # SECTION 4: AI Data

    # ------------------------------------------------------------------

    if detail_section == "AI Data":

        st.markdown("### Single Run AI Context")

        st.info("Copy this JSON to analyze this specific run with the AI.")

        show_ai_payload = st.checkbox(
            "Show Raw JSON Payload",
            value=False,
            key="show_single_run_ai_payload",
        )
        if show_ai_payload:
            st.json(build_run_ai_payload(selected_run, run_data, user_config))
        else:
            st.caption("The run payload is generated only when requested.")

        # AI Data is the final section in this renderer.
        return
