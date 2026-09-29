# view_trends.py

import streamlit as st

import charts
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

from config import (
    CALC_VERSION,
    AI_METRIC_DEFINITIONS,
    build_ai_system_context,
    cop_provenance,
    physics_assumptions,
)
from utils import safe_div


def _build_system_context(user_config: dict | None, include_heating_note: bool) -> str:
    """Compose AI system context from the default prompt plus user-provided freetext."""
    parts: list[str] = []

    base = build_ai_system_context("LONG_TERM_TRENDS").strip()
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

def _build_tariff_summary(user_config: dict | None) -> dict:
    """Return a structured tariff summary for AI exports."""
    summary: dict = {}
    if not isinstance(user_config, dict):
        return summary

    currency = user_config.get("currency", "€")
    tariff = user_config.get("tariff_structure")

    if isinstance(tariff, dict):
        day_rate = tariff.get("day_rate")
        night_rate = tariff.get("night_rate", day_rate)
        night_start = tariff.get("night_start", "00:00")
        night_end = tariff.get("night_end", "07:00")
        if night_rate is None or day_rate == night_rate:
            summary = {
                "mode": "Flat",
                "currency": currency,
                "rate": day_rate if day_rate is not None else night_rate,
            }
        else:
            summary = {
                "mode": "Day/Night",
                "currency": currency,
                "day_rate": day_rate,
                "night_rate": night_rate,
                "night_start": night_start,
                "night_end": night_end,
            }
    elif isinstance(tariff, list) and tariff:
        first = tariff[0] or {}
        rules_out: list[dict] = []
        for r in first.get("rules", []) or []:
            rules_out.append(
                {
                    "name": r.get("name"),
                    "start": r.get("start"),
                    "end": r.get("end"),
                    "rate": r.get("rate"),
                }
            )
        summary = {
            "mode": "Custom bands",
            "currency": currency,
            "valid_from": first.get("valid_from"),
            "name": first.get("name"),
            "rules": rules_out,
        }

    return summary

def _period_diagnostics(daily_df: pd.DataFrame) -> dict:
    """Period totals/ratios for the daily diagnostic metrics (sums, not means of ratios)."""
    def total(col):
        return float(pd.to_numeric(daily_df.get(col, pd.Series(dtype=float)), errors="coerce").sum())

    def ratio(num, den):
        return round(num / den, 3) if den > 0 else None

    out = {}
    if "Starts" in daily_df.columns:
        starts_heating = total("Starts_Heating")
        short = total("Short_Cycles_Count")
        out.update({
            "starts": int(total("Starts")),
            "starts_heating": int(starts_heating),
            "starts_dhw": int(total("Starts_DHW")),
            "short_cycles_count": int(short),
            "very_short_cycles_count": int(total("Very_Short_Cycles_Count")),
            "cycling_severity_index": ratio(short, starts_heating),
        })
        if "Run_Detection_Mins" in daily_df.columns:
            mins = pd.to_numeric(daily_df["Run_Detection_Mins"], errors="coerce").fillna(0)
            expected = pd.to_numeric(
                daily_df.get("Run_Detection_Expected_Mins", pd.Series(1440, index=daily_df.index)),
                errors="coerce",
            ).fillna(1440)
            window = daily_df.attrs.get("run_detection_window", {}) or {}
            out["run_detection"] = {
                "start": window.get("start"),
                "end": window.get("end"),
                "days_full": int((mins >= expected).sum()),
                "days_partial": int(((mins > 0) & (mins < expected)).sum()),
            }
    if "DHW_SCOP" in daily_df.columns:
        out["dhw_scop"] = ratio(total("Heat_DHW_kWh"), total("Electricity_DHW_kWh"))
    if "Night_Share_Of_Total_HP_Elec" in daily_df.columns:
        multi = daily_df["Night_Share_Of_Total_HP_Elec"].notna()
        metered = daily_df.get("Metered_Electricity_kWh", pd.Series(dtype=float))
        out["night_share_of_total_hp_elec"] = ratio(
            float(daily_df.loc[multi, "Cheap_Rate_Electricity_kWh"].sum()),
            float(metered[multi].sum()),
        )
    if "Effective_Avg_Tariff" in daily_df.columns:
        out["effective_avg_tariff"] = ratio(total("Daily_Cost_Euro"), total("Metered_Electricity_kWh"))
    if "Virtual_FTlim_Time_Mins" in daily_df.columns:
        out["virtual_ftlim_time_mins"] = int(total("Virtual_FTlim_Time_Mins"))
        out["virtual_ftlim_events"] = int(total("Virtual_FTlim_Events"))
        if "FTlim_Input_Coverage" in daily_df.columns:
            out["virtual_ftlim_days_covered"] = int(
                pd.to_numeric(daily_df["FTlim_Input_Coverage"], errors="coerce").ge(0.90).sum()
            )
    return out


def build_trends_ai_payload(daily_df: pd.DataFrame, runs_list: list, user_config: dict | None = None) -> dict:
    """LONG_TERM_TRENDS AI payload (pure: no Streamlit), so its schema is testable."""
    total_heat = float(daily_df.get("Total_Heat_kWh", pd.Series(dtype=float)).sum())
    total_elec = float(daily_df.get("Total_Electricity_kWh", pd.Series(dtype=float)).sum())
    total_cost = float(daily_df.get("Daily_Cost_Euro", pd.Series(dtype=float)).sum())
    # Heat-pump-only COP (immersion excluded); falls back for older daily tables
    cop = period_cop(daily_df)
    scop = cop if cop is not None else safe_div(total_heat, total_elec)
    cop_h4 = period_cop(daily_df, whole_system=True)

    # Prepare daily_df for JSON
    json_ready = daily_df.copy().reset_index()
    if json_ready.columns.size > 0:
        json_ready = json_ready.rename(
            columns={json_ready.columns[0]: "date"}
        )
    json_ready["date"] = json_ready["date"].astype(str)

    # Drop columns that are entirely NaN to reduce noise in AI payloads
    json_ready = json_ready.dropna(axis=1, how="all")

    # Drop metric columns whose counts are zero for the entire period
    drop_cols: set[str] = set()
    for col in list(json_ready.columns):
        if col.endswith("_count"):
            try:
                total = float(json_ready[col].fillna(0).sum())
            except Exception:
                continue
            if total == 0:
                drop_cols.add(col)
                base = col[: -len("_count")]
                for suffix in ("_mean", "_min", "_max"):
                    cand = f"{base}{suffix}"
                    if cand in json_ready.columns:
                        drop_cols.add(cand)
    if drop_cols:
        json_ready = json_ready.drop(columns=list(drop_cols), errors="ignore")

    float_cols = json_ready.select_dtypes(include=[float]).columns
    json_ready[float_cols] = json_ready[float_cols].round(2)

    period_summary = {
        "days": int(len(json_ready)),
        "total_heat_kwh": round(total_heat, 2),
        "total_electricity_kwh": round(total_elec, 2),
        "total_cost_eur": round(total_cost, 2),
        # period_scop keeps its name for older prompts; it is the energy-weighted
        # period COP, heat pump only (outdoor unit, SEPEMO H2)
        "period_scop": round(float(scop), 2),
        "period_cop_whole_system": round(float(cop_h4), 2) if cop_h4 is not None else None,
        "immersion_kwh": round(float(daily_df.get("Immersion_kWh", pd.Series(dtype=float)).sum()), 2),
        # Net heat accounting: total_heat_kwh = gross - defrost loss - transient loss
        "gross_heat_kwh": round(float(daily_df.get("Gross_Heat_kWh", pd.Series(dtype=float)).sum()), 2),
        "defrost_heat_loss_kwh": round(float(daily_df.get("Defrost_Heat_Loss_kWh", pd.Series(dtype=float)).sum()), 2),
        "transient_heat_loss_kwh": round(float(daily_df.get("Transient_Heat_Loss_kWh", pd.Series(dtype=float)).sum()), 2),
        "defrost_events": int(daily_df.get("Defrost_Events", pd.Series(dtype=float)).sum()),
        # Active cooling, excluded from the heating totals above
        "cooling_heat_removed_kwh": round(float(daily_df.get("Cooling_Heat_Removed_kWh", pd.Series(dtype=float)).sum()), 2),
        "cooling_electricity_kwh": round(float(daily_df.get("Electricity_Cooling_kWh", pd.Series(dtype=float)).sum()), 2),
    }
    period_summary.update(_period_diagnostics(daily_df))

    include_heating_note = any(
        bool(r.get("heating_during_dhw_detected")) for r in runs_list or []
    )

    # Preserve user-supplied AI context entries (empty strings removed upstream)
    ai_context_inputs = {}
    if isinstance(user_config, dict):
        ai_context_inputs = {
            k: v for k, v in (user_config.get("ai_context") or {}).items() if isinstance(v, str) and v.strip()
        }

    ai_payload = {
        "meta": {
            "report_type": "LONG_TERM_TRENDS",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "calc_version": CALC_VERSION,
        },
        "system_context": _build_system_context(user_config, include_heating_note),
        "tariff_summary": _build_tariff_summary(user_config),
        "physics_assumptions": physics_assumptions(user_config, daily_df.attrs.get("heat_source")),
        "cop_provenance": cop_provenance(user_config, daily_df.attrs.get("heat_source")),
        "metric_definitions": AI_METRIC_DEFINITIONS,
        "ai_context": ai_context_inputs,
        "config_history": user_config.get("config_history", []) if isinstance(user_config, dict) else [],
        "period_summary": period_summary,
        "daily_metrics": json_ready.to_dict(orient="records"),
    }

    return ai_payload


# Minimum daily electricity (kWh) for a day to appear on the Heating / DHW COP line.
MODE_COP_MIN_ELEC_KWH = 0.2


def period_cop(daily_df: pd.DataFrame, whole_system: bool = False) -> float | None:
    """Energy-weighted COP over the rows given: total heat / total electricity.

    Heat-pump electricity only (immersion excluded); whole_system adds the indoor
    unit (SEPEMO H4). None when the inputs are missing.
    """
    if "HP_Electricity_kWh" not in daily_df.columns:
        return None
    elec = float(daily_df["HP_Electricity_kWh"].sum())
    if whole_system:
        if "Indoor_Electricity_kWh" not in daily_df.columns:
            return None
        elec += float(daily_df["Indoor_Electricity_kWh"].sum())
    heat = float(daily_df.get("Total_Heat_kWh", pd.Series(dtype=float)).sum())
    return safe_div(heat, elec) if elec > 0 else None



# Headline names per All / Heating / DHW choice.
MODE_KPI_NAMES = {
    "All": {"heat": "Total heat", "elec": "Electricity", "cost": "Total cost", "cop": "Period COP",
            "system_cop": "System COP", "for": ""},
    "Heating": {"heat": "Space heating heat", "elec": "Heating electricity", "cost": "Heating cost",
                "cop": "Heating COP", "system_cop": "Heating system COP", "for": " for space heating"},
    "DHW": {"heat": "Hot-water heat", "elec": "Hot-water electricity", "cost": "Hot-water cost",
            "cop": "Hot-water COP", "system_cop": "Hot-water system COP", "for": " for hot water"},
}
# (heat, heat-pump electricity, indoor unit while the mode runs, cost, minutes) per mode
_MODE_COLUMNS = {
    "Heating": ("Heat_Heating_kWh", "Electricity_Heating_kWh", "Indoor_Heating_kWh", "Heating_Cost", "Heating_Mins"),
    "DHW": ("Heat_DHW_kWh", "Electricity_DHW_HP_kWh", "Indoor_DHW_kWh", "DHW_Cost", "DHW_Mins"),
}


def _sum(df: pd.DataFrame, col: str):
    return float(df[col].sum()) if col in df.columns else None


def _share(part, what: str):
    return f"{part:.0%} {what}" if part is not None else None


def mode_totals(daily_df: pd.DataFrame, mode: str) -> dict:
    """Headline figures for All, Heating or DHW: days, heat, heat-pump electricity (what the COP divides by,
    the sidebar's figure), cost, COP (H2) and system COP (H4), plus the mode's share of the period's heat,
    electricity and cost. Cost is None when the result predates per-mode costs."""
    total_heat = _sum(daily_df, "Total_Heat_kWh") or 0.0
    total_elec = _sum(daily_df, "HP_Electricity_kWh")
    total_cost = _sum(daily_df, "Daily_Cost_Euro")
    if mode not in _MODE_COLUMNS:
        return {"days": len(daily_df), "heat": total_heat, "elec": total_elec, "cost": total_cost or 0.0,
                "cop": period_cop(daily_df), "system_cop": period_cop(daily_df, whole_system=True),
                "heat_share": None, "elec_share": None, "cost_share": None}
    heat_col, elec_col, indoor_col, cost_col, mins_col = _MODE_COLUMNS[mode]
    heat = _sum(daily_df, heat_col) or 0.0
    elec = _sum(daily_df, elec_col)
    cost = _sum(daily_df, cost_col)
    cop = safe_div(heat, elec) if elec else None
    system_cop = None
    if elec is not None and "Indoor_Electricity_kWh" in daily_df.columns and indoor_col in daily_df.columns:
        whole = elec + float(daily_df[indoor_col].sum())
        system_cop = safe_div(heat, whole) if whole else None
    days = int((daily_df[mins_col] > 0).sum()) if mins_col in daily_df.columns else len(daily_df)
    return {"days": days, "heat": heat, "elec": elec, "cost": cost, "cop": cop, "system_cop": system_cop,
            "heat_share": heat / total_heat if total_heat > 0 else None,
            "elec_share": elec / total_elec if elec is not None and total_elec else None,
            "cost_share": cost / total_cost if cost is not None and total_cost else None}


@st.fragment
def render_long_term_trends(daily_df: pd.DataFrame, raw_df: pd.DataFrame, runs_list: list, user_config: dict | None = None) -> None:
    """
    Long-term performance view:
    - KPI cards
    - Daily stacked energy balance
    - Environmental charts
    - Weather-compensation scatter (heating)
    - DHW run scatter
    - AI JSON export for long-term analysis
    """
    st.title(" Long-Term Performance")

    if daily_df is None or daily_df.empty:
        st.warning("No valid daily data found.")
        return

    # Underlined tabs; they report which one is open (on_change="rerun"), so only the open tab is built.
    perf_tab, ai_tab = st.tabs(["Performance", "AI report"], key="long_term_tab", on_change="rerun")
    trends_section = "AI Report" if ai_tab.open else "Performance"

    # ----------------------------------------------------------------------
    # PERFORMANCE SECTION
    # ----------------------------------------------------------------------
    with perf_tab:
        if trends_section == "Performance":
            # All / Heating / DHW: one choice for the whole section. The headline cards, the energy bars and
            # the COP lines on the environment charts all follow it; the sidebar's summary stays the whole period.
            energy_mode = st.pills(
                "Show", ["All", "Heating", "DHW"], default="All", key="long_term_energy_mode",
                label_visibility="collapsed",
            ) or "All"  # clicking the selected pill deselects it (None)

            # 1. KPI cards
            currency = (user_config or {}).get("currency", "€") if isinstance(user_config, dict) else "€"
            totals = mode_totals(daily_df, energy_mode)
            names = MODE_KPI_NAMES[energy_mode]

            def _part(share_value, what):
                return _share(share_value, what) if energy_mode != "All" else None

            # (keyed: app.py puts the cards two to a row on phones, st-key-therm-kpis)
            k1, k2, k3, k4, k5 = st.container(key="therm-kpis").columns(5)
            k1.metric(names["heat"], f"{totals['heat']:,.0f} kWh", border=True,
                      delta=_part(totals["heat_share"], "of all heat"), delta_color="off", delta_arrow="off")
            k2.metric(
                names["elec"], f"{totals['elec']:,.0f} kWh" if totals["elec"] is not None else "–", border=True,
                delta=_part(totals["elec_share"], "of all electricity"), delta_color="off", delta_arrow="off",
                help=f"The heat pump's electricity while it runs{names['for']}: what the COP divides by. The "
                     "indoor unit (System COP) and the immersion heater (never in the COP) are shown in the chart.",
            )
            cost, heat = totals["cost"], totals["heat"]
            k3.metric(
                names["cost"], f"{currency}{cost:,.2f}" if cost is not None else "–", border=True,
                delta=f"{currency}{cost / heat:.3f}/kWh of heat" if cost and heat > 0 else None,
                delta_color="off", delta_arrow="off",
                help="Electricity cost ÷ heat delivered is the cost of heat. The Run Inspector ranks runs by it, also "
                     "at today's prices, to show which behaviour gives the cheapest heat."
                     + (" Only this mode's electricity, at the price of the minutes it ran." if energy_mode != "All"
                        else ""),
            )
            k4.metric(
                names["cop"], f"{totals['cop']:.2f}" if totals["cop"] is not None else "–", border=True,
                delta="Heat pump only", delta_color="off", delta_arrow="off",
                help=f"Heat ÷ electricity{names['for']}, outdoor unit only (SEPEMO H2). Immersion is excluded "
                     "and shown separately.",
            )
            k5.metric(
                names["system_cop"], f"{totals['system_cop']:.2f}" if totals["system_cop"] is not None else "–",
                border=True, delta="Incl. indoor unit", delta_color="off", delta_arrow="off",
                help=f"As {names['cop']}, plus the indoor unit's controls and water pump"
                     + (" while it runs" if energy_mode != "All" else "")
                     + " (SEPEMO H4). Needs Indoor Power mapped. Immersion is excluded.",
            )
            provenance = cop_provenance(user_config, daily_df.attrs.get("heat_source"))
            st.caption(
                f"COP basis (primary Power): {provenance.get('display', 'Not specified')}. "
                "This does not describe optional Indoor Power."
            )

            # 2. Daily Energy (Stacked)
            st.subheader("Daily energy balance")
            st.caption("Heat and electricity per day, stacked by component")

            has_heat_energy = (
                ("Heat_Heating_kWh" in daily_df.columns)
                or ("Heat_DHW_kWh" in daily_df.columns)
            )

            if has_heat_energy:
                fig = go.Figure()

                # (column, legend name, colour, stack, modes shown in, hover label)
                # Left stack (offsetgroup 0) is heat output, right (1) electricity.
                # DHW electricity is heat-pump only when the Power meter includes the
                # immersion (Electricity_DHW_HP_kWh); the element has its own bar.
                dhw_elec_col = (
                    "Electricity_DHW_HP_kWh" if "Electricity_DHW_HP_kWh" in daily_df.columns
                    else "Electricity_DHW_kWh"
                )
                components = [
                    ("Heat_Heating_kWh", "Space Heat", "#ffa600", 0, {"All", "Heating"}, "Heat Output<br>Space Heat"),
                    ("Heat_DHW_kWh", "DHW Heat", "#ffd580", 0, {"All", "DHW"}, "Heat Output<br>DHW Heat"),
                    ("Electricity_Heating_kWh", "Space Elec", "#003f5c", 1, {"All", "Heating"}, "Electricity Input<br>Space Elec"),
                    (dhw_elec_col, "DHW Elec", "#58508d", 1, {"All", "DHW"}, "Electricity Input<br>DHW Elec"),
                    # Indoor unit (whole-system figures): the whole day incl. standby
                    # under All, only while the mode runs under Heating / DHW
                    ("Indoor_Electricity_kWh", "Indoor Unit", "#7a8b99", 1, {"All"}, "Electricity Input<br>Indoor Unit"),
                    ("Indoor_Heating_kWh", "Indoor Unit", "#7a8b99", 1, {"Heating"}, "Electricity Input<br>Indoor Unit"),
                    ("Indoor_DHW_kWh", "Indoor Unit", "#7a8b99", 1, {"DHW"}, "Electricity Input<br>Indoor Unit"),
                    ("Immersion_kWh", "Immersion", "#bc5090", 1, {"All", "DHW"}, "Electricity Input<br>Immersion (not in COP)"),
                ]
                bases = {
                    0: pd.Series(0.0, index=daily_df.index),
                    1: pd.Series(0.0, index=daily_df.index),
                }
                # Heating or DHW fades the other mode's components instead of removing them, so the scale and
                # the shape of each day stay the same. (The indoor unit has one bar per mode: its whole day under
                # All, only while that mode runs under Heating / DHW.)
                indoor_by_mode = {"Indoor_Electricity_kWh", "Indoor_Heating_kWh", "Indoor_DHW_kWh"}
                for col, name, colour, group, modes, hover in components:
                    if col not in daily_df.columns:
                        continue
                    if col in indoor_by_mode and energy_mode not in modes:
                        continue
                    values = pd.to_numeric(daily_df[col], errors="coerce").fillna(0)
                    fig.add_trace(
                        go.Bar(
                            x=daily_df.index,
                            y=values,
                            name=name,
                            marker_color=colour,
                            opacity=1.0 if energy_mode in modes else 0.18,
                            offsetgroup=group,
                            base=bases[group],
                            legendgroup="Output" if group == 0 else "Input",
                            hovertemplate=hover + ": %{y:.1f} kWh",
                        )
                    )
                    bases[group] = bases[group] + values

                fig.update_layout(
                    yaxis_title="Energy (kWh)",
                    barmode="group",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                    margin=dict(t=40, b=20),
                )
                charts.show(fig, key="daily_energy_chart")

                # Net heat accounting note (heat taken back during defrost / start-stop)
                defrost_loss = float(daily_df.get("Defrost_Heat_Loss_kWh", pd.Series(dtype=float)).sum())
                transient_loss = float(daily_df.get("Transient_Heat_Loss_kWh", pd.Series(dtype=float)).sum())
                defrost_events = int(daily_df.get("Defrost_Events", pd.Series(dtype=float)).sum())
                if defrost_loss > 0 or transient_loss > 0:
                    st.caption(
                        f"Heat output is shown **net**: the {defrost_loss:.1f} kWh the heat pump took back during "
                        f"{defrost_events} defrost{'s' if defrost_events != 1 else ''} and the "
                        f"{transient_loss:.1f} kWh lost at run start/stop are subtracted, as the heat pump "
                        "standards count it (EN 14511 / EN 14825)."
                    )

                # Home Assistant retention boundary: days only partly at full resolution
                incomplete = daily_df.get("Incomplete_Mins", pd.Series(dtype=float))
                incomplete = incomplete[incomplete > 0] if len(incomplete) else incomplete
                if len(incomplete):
                    parts = ", ".join(
                        f"{pd.Timestamp(d_).strftime('%d %b')} ({m / 60:.1f} h)" for d_, m in incomplete.items()
                    )
                    st.caption(
                        f"⚠️ Before full-resolution data starts (Home Assistant retention boundary): {parts}. "
                        "Energy for that time comes from hourly averages; runs aren't analysed there."
                    )

                # Active cooling is kept out of every heating figure; summarise it here
                cool_removed = float(daily_df.get("Cooling_Heat_Removed_kWh", pd.Series(dtype=float)).sum())
                if cool_removed > 0:
                    cool_elec = float(daily_df.get("Electricity_Cooling_kWh", pd.Series(dtype=float)).sum())
                    cool_days = int((daily_df.get("Cooling_Mins", pd.Series(dtype=float)) > 0).sum())
                    st.caption(
                        f"❄️ Cooling on {cool_days} day{'s' if cool_days != 1 else ''}: "
                        f"{cool_removed:.1f} kWh removed for {cool_elec:.1f} kWh electricity "
                        f"(EER {safe_div(cool_removed, cool_elec):.2f}). Not included in the "
                        "heating figures above."
                    )
            else:
                st.info(
                    "Daily energy balance is disabled because no Heat output is "
                    "available (no Flow Rate or Heat sensor mapped)."
                )

            # 3. Environmental Charts (COP lines follow the All / Heating / DHW toggle)
            st.divider()
            # One chart per row: side by side, each chart's two right-hand axes (humidity
            # and COP) had too little room and their tick labels ran together.
            c1, c2 = st.container(), st.container()
            cop_name, h2_col, elec_col = {
                "All": ("Daily COP", "Daily_COP", "HP_Electricity_kWh"),
                "Heating": ("Heating COP", "Heating_COP", "Electricity_Heating_kWh"),
                "DHW": ("DHW COP", "DHW_COP", "Electricity_DHW_HP_kWh"),
            }[energy_mode]
            # Outdoor unit (H2) only; the whole-system figure is the Period COP
            # (whole system) KPI above. Days with under 0.2 kWh of electricity are
            # left empty: a few minutes at the edge of a run (e.g. 0.14 kWh heat /
            # 0.03 kWh on a summer "heating" day) would otherwise plot as spikes.
            cop_lines = []
            if h2_col in daily_df.columns:
                enough = pd.to_numeric(
                    daily_df.get(elec_col, pd.Series(np.inf, index=daily_df.index)), errors="coerce"
                ) >= MODE_COP_MIN_ELEC_KWH
                series = pd.to_numeric(daily_df[h2_col], errors="coerce").where(enough)
                if series.notna().any():   # no line (or legend entry) when no day qualifies
                    cop_lines.append((series, cop_name, "solid"))
            # COP gets its own whole-number axis from zero on both charts (stretched if
            # a day goes above five).
            cop_max = max([5.0] + [float(s.max()) for s, _, _ in cop_lines])
            cop_range = [0, float(np.ceil(cop_max))]

            # Respect user-selected units for wind speed (default m/s)
            wind_unit = "m/s"
            wind_factor = 1.0
            try:
                cfg_units = st.session_state.get("system_config", {}).get("units", {})
            except Exception:
                cfg_units = {}
            if isinstance(cfg_units, dict):
                user_unit = cfg_units.get("Wind_Speed")
                if user_unit in ["m/s", "km/h", "mph"]:
                    wind_unit = user_unit
                    wind_factor = {"m/s": 1.0, "km/h": 3.6, "mph": 2.23693629}[wind_unit]

            with c1:
                fig_env = make_subplots(specs=[[{"secondary_y": True}]])
                if "Wind_Avg" in daily_df.columns:
                    wind_series = pd.to_numeric(daily_df["Wind_Avg"], errors="coerce") * wind_factor
                    fig_env.add_trace(
                        go.Scatter(
                            x=daily_df.index,
                            y=wind_series,
                            name="Wind",
                            line=dict(color="grey"),
                            connectgaps=True,
                            hovertemplate=f"Wind: %{{y:.1f}} {wind_unit}",
                        ),
                        secondary_y=False,
                    )

                if "Humidity_Avg" in daily_df.columns:
                    fig_env.add_trace(
                        go.Scatter(
                            x=daily_df.index,
                            y=daily_df["Humidity_Avg"],
                            name="Humidity",
                            line=dict(color="blue", dash="dot"),
                            connectgaps=True,
                            hovertemplate="Humidity: %{y:.1f} %",
                        ),
                        secondary_y=True,
                    )

                # COP on a third axis (right, outside humidity) so its 1-5 scale isn't
                # flattened by humidity's 0-100 %. No gap bridging: days without a
                # qualifying COP stay empty.
                for series, name, dash in cop_lines:
                    fig_env.add_trace(
                        go.Scatter(
                            x=daily_df.index,
                            y=series,
                            name=name,
                            mode="lines+markers",   # a lone qualifying day still shows
                            marker=dict(size=5),
                            line=dict(color="green", dash=dash),
                            connectgaps=False,
                            yaxis="y3",
                            hovertemplate=name + ": %{y:.2f}",
                        )
                    )

                fig_env.update_layout(
                    title="Wind & Humidity",
                    height=300,
                    hovermode="x unified",
                    # Legend above the plot: on the right it covered the extra COP axis
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig_env.update_yaxes(title_text=f"Wind ({wind_unit})", tickformat=".0f", secondary_y=False)
                fig_env.update_yaxes(title_text="Humidity (%)", tickformat=".0f", showgrid=False, secondary_y=True)
                if cop_lines:
                    # Full-width chart (see c1 above), so the humidity and COP axes on the
                    # right have room; at half width their tick labels overlapped.
                    fig_env.update_layout(
                        xaxis=dict(domain=[0, 0.9]),
                        yaxis3=dict(
                            title=dict(text=cop_name, font=dict(color="green")),
                            tickfont=dict(color="green"),
                            overlaying="y", side="right", anchor="free", position=1.0,
                            range=cop_range, dtick=1, showgrid=False,
                        ),
                    )
                charts.show(fig_env, key="env_chart")

            with c2:
                fig_sol = make_subplots(specs=[[{"secondary_y": True}]])
                if "Solar_Avg" in daily_df.columns:
                    fig_sol.add_trace(
                        go.Bar(
                            x=daily_df.index,
                            y=daily_df["Solar_Avg"],
                            name="Solar",
                            marker_color="orange",
                            hovertemplate="Solar: %{y:.1f} W/m²",
                        ),
                        secondary_y=False,
                    )

                for series, name, dash in (cop_lines if has_heat_energy else []):
                    fig_sol.add_trace(
                        go.Scatter(
                            x=daily_df.index,
                            y=series,
                            name=name,
                            mode="lines+markers",
                            marker=dict(size=5),
                            line=dict(color="green", dash=dash),
                            hovertemplate=name + ": %{y:.2f}",
                        ),
                        secondary_y=True,
                    )

                fig_sol.update_layout(
                    title="Solar Gain vs Efficiency",
                    height=300,
                    hovermode="x unified",
                    # Legend above the plot: on the right it covered the extra COP axis
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig_sol.update_yaxes(title_text="Solar (W/m²)", secondary_y=False)
                if cop_lines and has_heat_energy:
                    fig_sol.update_yaxes(title_text=cop_name, range=cop_range, dtick=1, showgrid=False,
                                         secondary_y=True)
                charts.show(fig_sol, key="solar_chart")

            # 4. Heating Weather Compensation (space heating runs)
            st.divider()
            st.subheader("1. Space Heating Run Averages: Weather Compensation Curve")
            st.caption(
                "Target: Diagonal line downwards (one dot representing the "
                "average of each run)."
            )

            heating_runs = [
                r
                for r in (runs_list or [])
                if r.get("run_type") == "Heating" and r.get("avg_flow_temp", 0) > 25
            ]

            if heating_runs:
                df_heat_scatter = pd.DataFrame(heating_runs)
                fig_wc = px.scatter(
                    df_heat_scatter,
                    x="avg_outdoor",
                    y="avg_flow_temp",
                    color="run_cop",
                    color_continuous_scale="RdYlGn",
                    title="Space Heating Run Averages: Flow Temp vs Outdoor Temp",
                    opacity=0.9,
                    labels={
                        "avg_outdoor": "Avg Outdoor Temp (°C)",
                        "avg_flow_temp": "Avg Flow Temp (°C)",
                        "run_cop": "COP",
                    },
                    hover_data={
                        "avg_outdoor": ":.1f",
                        "avg_flow_temp": ":.1f",
                        "run_cop": ":.2f",
                        # Run start timestamp
                        "start": "|%d-%m-%Y %H:%M",
                    },
                )

                # Inefficient Zone (> 43°C)
                fig_wc.add_shape(
                    type="rect",
                    x0=10,
                    y0=43,
                    x1=20,
                    y1=60,
                    line=dict(color="red", width=1, dash="dot"),
                    fillcolor="rgba(0,0,0,0)",
                    opacity=0.3,
                    layer="below",
                )
                fig_wc.add_annotation(
                    x=15,
                    y=58,
                    text="Inefficient Zone (>43°C)",
                    showarrow=False,
                    font=dict(color="red", size=9),
                    opacity=0.6,
                )
                charts.show(fig_wc, key="wc_heating_chart")
            else:
                st.info("No Space Heating runs detected.")

            # 5. DHW Temperature Consistency
            st.subheader("2. Hot Water Run Averages: Temperature Consistency")
            st.caption(
                "Target: Flat horizontal cluster (one dot representing the "
                "average of each run)."
            )

            dhw_runs = [
                r
                for r in (runs_list or [])
                if r.get("run_type") == "DHW" and r.get("avg_flow_temp", 0) > 25
            ]

            if dhw_runs:
                df_dhw_scatter = pd.DataFrame(dhw_runs)
                fig_dhw = px.scatter(
                    df_dhw_scatter,
                    x="avg_outdoor",
                    y="avg_flow_temp",
                    color="run_cop",
                    color_continuous_scale="RdYlGn",
                    title="Hot Water Run Averages: Flow Temp vs Outdoor Temp",
                    opacity=0.9,
                    labels={
                        "avg_outdoor": "Avg Outdoor Temp (°C)",
                        "avg_flow_temp": "Avg Flow Temp (°C)",
                        "run_cop": "COP",
                    },
                    hover_data={
                        "avg_outdoor": ":.1f",
                        "avg_flow_temp": ":.1f",
                        "run_cop": ":.2f",
                        "start": "|%d-%m-%Y %H:%M",
                    },
                )

                # Reference line at 50°C
                fig_dhw.add_hline(
                    y=50.0,
                    line_dash="dot",
                    line_color="grey",
                    annotation_text="Typical Target (50°C)",
                )
                charts.show(fig_dhw, key="wc_dhw_chart")
            else:
                st.info("No Hot Water (DHW) runs detected.")

    # ----------------------------------------------------------------------
    # AI REPORT SECTION
    # ----------------------------------------------------------------------
    with ai_tab:
        if trends_section == "AI Report":
            st.markdown("### Download AI System Context")
            st.info(
                "The JSON below contains all the data required for a full "
                "long-term analysis."
            )

            ai_payload = build_trends_ai_payload(daily_df, runs_list, user_config)

            st.download_button(
                label=" Download JSON for AI Analysis",
                data=json.dumps(ai_payload, indent=2),
                file_name=f"heat_pump_long_term_ai_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
            )

            # Rendering the full JSON is slow; only do it on demand
            if st.checkbox("Show Raw JSON Payload", value=False):
                st.json(ai_payload)
