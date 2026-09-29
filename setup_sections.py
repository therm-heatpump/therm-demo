# setup_sections.py
"""
Setup's sections, drawn from and saved into one draft.

The draft (st.session_state[DRAFT_KEY]) is a profile-shaped dict and the source of truth for every
Setup value. Streamlit forgets the state of a widget that is not drawn in a run, and the wizard draws
one step at a time, so a value kept only in a widget key would be lost on Next → Back. Each widget is
seeded from the draft when its key is missing and writes its value back after it is drawn; anything
else that changes a value (Detect, loading a profile) changes the draft and drops the widget key.

build_config(draft) turns the draft into the saved profile without drawing anything, so steps the
user never opened contribute their draft or default values.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import streamlit as st

import config
import currencies
from schema_defs import (
    AI_CONTEXT_PROMPTS, ENVIRONMENTAL_SENSORS, OPTIONAL_SENSORS, RECOMMENDED_SENSORS, REQUIRED_SENSORS,
    ROOM_SENSOR_PREFIX, SECONDARY_WEATHER_SENSORS, WEATHER_FALLBACKS, ZONE_SENSORS, get_unit_options,
)

DRAFT_KEY = "setup_draft"
DRAFT_BASE_KEY = "setup_draft_base"
ERRORS_KEY = "setup_errors"

# Name used when the user leaves Profile name empty (shown as the placeholder).
DEFAULT_PROFILE_NAME = "My Heat Pump"

ROOM_ROLES = [f"{ROOM_SENSOR_PREFIX}{i}" for i in range(1, config.MAX_ROOMS + 1)]
ROOMS_SHOWN = 8           # room slots shown before "Add another room"
ROOM_SLOTS_KEY = "room_slots"
ZONE_ROLES = list(ZONE_SENSORS)
HOT_WATER_ROLES = ["DHW_Active", "ValveMode", "DHW_Mode", "DHW_Temp", "Immersion_Mode"]
OTHER_ROLES = [r for r in OPTIONAL_SENSORS if r not in HOT_WATER_ROLES]
WEATHER_ROLES = list(ENVIRONMENTAL_SENSORS) + list(SECONDARY_WEATHER_SENSORS)
# Roles Setup offers; anything else in an old profile is not carried over.
KNOWN_ROLES = (list(REQUIRED_SENSORS) + list(RECOMMENDED_SENSORS) + ROOM_ROLES + ZONE_ROLES
               + WEATHER_ROLES + list(OPTIONAL_SENSORS))

# Keys of the widgets these sections draw (dropped whenever the draft changes underneath them).
WIDGET_PREFIXES = ("map_", "unit_", "link_", "thr_", "phys_", "wc_", "ai_", "del_hist_")
WIDGET_KEYS = {"rooms_add", ROOM_SLOTS_KEY, "currency_other", "zones_choice", "zones_use_candidates", "tariff_editor", "tariff_new_from", "tariff_add_change", "currency_pick", "timezone_pick",
               "electricity_source",
               "new_change_date", "new_change_time", "new_change_tag", "new_change_note", "config_hist_error",
               "tariff_rows_df", "tariff_rows_signature"}


# ---------------------------------------------------------------------------
# Numeric settings: one table, used both to draw them and to build the profile
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Number:
    group: str            # "thresholds" or "physics_thresholds"
    key: str
    defaults: dict        # the config table holding the default
    fallback: float
    label: str
    help: str
    step: float
    min_value: float | None = 0.0
    max_value: float | None = None
    fmt: str | None = None

    def default(self) -> float:
        return float(self.defaults.get(self.key, self.fallback))

    @property
    def widget_key(self) -> str:
        return ("thr_" if self.group == "thresholds" else "phys_") + self.key


T, P = "thresholds", "physics_thresholds"
_TH, _EN, _PH = config.THRESHOLDS, config.ENGINE_STATE_THRESHOLDS, config.PHYSICS_THRESHOLDS

RUN_DETECTION = [
    Number(T, "minimum_run_duration_min", _TH, 5, "Minimum run duration (min)",
           "Cycles shorter than this are ignored as noise.", 1.0),
    Number(T, "short_cycle_min", _TH, 20, "Short cycle threshold (min)",
           "Heating runs shorter than this are flagged as short cycles.", 1.0),
    Number(T, "very_short_cycle_min", _TH, 10, "Very short cycle threshold (min)",
           "Heating runs shorter than this are flagged as critical short cycles.", 1.0),
    Number(P, "power_on_W", _EN, 150.0, "Power on threshold (W)",
           "The unit counts as running when Power is above this. Set it midway between your standby reading "
           "and your lowest running reading (default 150 W).", 10.0),
    Number(P, "coast_down_window_min", _EN, 5, "Coast-down lookback (min)",
           "Minutes to look back to decide if a run is coasting after power drops.", 1.0),
]
HYDRAULICS = [
    Number(P, "min_flow_rate_lpm", _PH, 3.0, "Min flow rate to count water movement (L/min)",
           "Heat output forced to 0 if flow is below this value (filters idle noise).", 0.1),
    Number(T, "flow_limit_tolerance", _TH, 2.0, "Flow limit tolerance (°C)",
           "Gap allowed between target and actual flow before flagging a flow limit issue.", 0.1),
    Number(T, "flow_limit_min_duration", _TH, 15, "Flow limit minimum duration (min)",
           "Duration required to trigger a flow limit warning.", 1.0),
]
HYDRAULICS_AFTER_CURVE = [
    Number(P, "delta_on_C", _EN, 1.0, "DeltaT on threshold (°C)",
           "Minimum DeltaT required to confirm active heating.", 0.1),
    Number(T, "dhw_scop_low", _TH, 2.2, "Low DHW COP threshold",
           "DHW runs with efficiency below this are flagged as poor.", 0.1),
]
HEATING_DURING_DHW = Number(
    T, "heating_during_dhw_detection_pct", _TH, 0.15, "Heating during DHW detection fraction",
    "Share of a DHW run with a zone pump running that flags heating during DHW (e.g. 0.15 = 15 %). "
    "Needs zone pump/valve signals.", 0.01, 0.0, 1.0, "%.2f")
DIAGNOSTICS = [
    Number(T, "hdd_base_temp", _TH, 18.0, "Heating degree day base temp (°C)",
           "Base temperature for heating degree day calculations.", 0.5, None),
    Number(P, "freq_on_hz", _EN, 10.0, "Freq on threshold (Hz)",
           "Compressor frequency above this is considered active.", 0.5),
    Number(P, "freq_off_hz", _EN, 5.0, "Freq off threshold (Hz)",
           "Compressor frequency below this is considered inactive.", 0.5),
    Number(T, "flow_over_43c_pct_high", _TH, 20, "High flow temp share (%)",
           "Warn if flow temperature exceeds 43°C for more than this share of a run.", 1.0, 0.0, 100.0),
    Number(T, "high_night_share", _TH, 0.50, "High night-rate share (fraction)",
           "Night-rate share above this is considered high for load-shifting analysis.", 0.05, 0.0, 1.0, "%.2f"),
    Number(P, "flow_on_lpm", _EN, 2.0, "Flow on threshold (L/min)",
           "Flow rate above this implies pump running (fallback detection).", 0.1),
    Number(P, "flow_off_lpm", _EN, 1.0, "Flow off threshold (L/min)",
           "Flow rate below this implies pump off (fallback detection).", 0.1),
    Number(P, "delta_coast_min_C", _EN, 0.5, "DeltaT coast threshold (°C)",
           "DeltaT threshold to consider a run coasting after compressor stop.", 0.1),
    Number(P, "min_freq_for_delta_t", _PH, 5.0, "Min freq to trust DeltaT (Hz)",
           "Minimum compressor frequency to trust DeltaT calculation.", 0.5),
    Number(P, "min_freq_for_heat", _PH, 7.5, "Min freq to calculate heat (Hz)",
           "Minimum compressor frequency to calculate heat output.", 0.5),
    Number(P, "max_valid_delta_t", _PH, 15.0, "Max valid DeltaT (°C)",
           "DeltaT above this is treated as sensor error and ignored.", 0.5),
    Number(P, "min_valid_delta_t", _PH, 0.2, "Min valid DeltaT (°C)",
           "DeltaT below this is treated as noise and ignored.", 0.05),
]
# Only with zone signals: a heating run while no zone calls for heat is usually a start-up blip.
ZONE_RUN_FILTER = [
    Number(T, "min_heating_run_minutes_with_no_zones", _TH, 8, "Shortest heating run with no zone calling (min)",
           "A heating run while no zone calls for heat is ignored if it is shorter than this (start-up blips). "
           "Without zone signals every heating run is kept.", 1.0),
    Number(T, "min_heating_run_heat_kwh_with_no_zones", _TH, 0.25, "Least heat for a run with no zone calling (kWh)",
           "A heating run while no zone calls for heat is ignored if it delivers less heat than this.", 0.05),
]
NUMBERS = RUN_DETECTION + HYDRAULICS + HYDRAULICS_AFTER_CURVE + [HEATING_DURING_DHW] + DIAGNOSTICS + ZONE_RUN_FILTER

ZONES_NONE, ZONES_SIGNALS = "none", "signals"
ZONE_CHOICES = {
    ZONES_NONE: "One heating circuit, or no zone signals in Home Assistant",
    ZONES_SIGNALS: "Separate zones, each with an on/off signal in Home Assistant",
}

CURVE_FIELDS = [("design_outdoor_c", "Cold outdoor (°C)", 1.0), ("mild_outdoor_c", "Mild outdoor (°C)", 1.0),
                ("design_flow_c", "Flow at cold (°C)", 0.5), ("mild_flow_c", "Flow at mild (°C)", 0.5),
                ("min_flow_c", "Minimum flow (°C)", 0.5), ("max_flow_c", "Maximum flow (°C)", 0.5)]
OTHER_CURRENCY = "Other"


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------
def new_draft(profile: dict | None) -> dict:
    """A complete draft from a saved profile (or none): every value Setup can show, defaults filled in."""
    p = copy.deepcopy(profile or {})
    thresholds, physics = dict(p.get("thresholds") or {}), dict(p.get("physics_thresholds") or {})
    for n in NUMBERS:
        target = thresholds if n.group == T else physics
        target.setdefault(n.key, n.default())
    physics.setdefault("heat_source", config.HEAT_SOURCE_DEFAULT)
    physics.setdefault("fluid_specific_heat_kj", config.FLUID_SPECIFIC_HEAT_DEFAULT_KJ)
    physics.setdefault("power_includes_immersion", False)
    # This is the legacy boundary: old profiles have no source declaration,
    # and Setup presents/saves the explicit, safe "unknown" choice.
    electricity_source = config.resolve_electricity_source(p.get("electricity_source"))
    name = p.get("profile_name") or ""
    currency_code, currency_symbol = currencies.resolve(p)
    mapping = {k: v for k, v in (p.get("mapping") or {}).items() if v not in (None, "", "None")}
    units = dict(p.get("units") or {})
    rooms_per_zone = {k: list(v or []) for k, v in (p.get("rooms_per_zone") or {}).items()}
    heating_zones = heating_zones_of(p)
    # Zone choices put aside while zones were off come back into the draft, so switching zones on
    # again restores them.
    inactive = p.get("inactive_zones") or {}
    for role, entity in (inactive.get("mapping") or {}).items():
        mapping.setdefault(role, entity)
    for role, unit in (inactive.get("units") or {}).items():
        units.setdefault(role, unit)
    for zone, rooms in (inactive.get("rooms_per_zone") or {}).items():
        if not rooms_per_zone.get(zone):
            rooms_per_zone[zone] = list(rooms or [])
    return {
        "profile_name": "" if name == DEFAULT_PROFILE_NAME else name,
        "mapping": mapping,
        "units": units,
        "rooms_per_zone": rooms_per_zone,
        "heating_zones": heating_zones,
        "thresholds": thresholds,
        "physics_thresholds": physics,
        "electricity_source": electricity_source,
        "weather_curve": dict(config.resolve_weather_curve(p)),
        "tariff_structure": copy.deepcopy(p.get("tariff_structure") or []),
        "currency": currency_symbol,
        "currency_code": currency_code,
        "timezone": config.resolve_timezone(p.get("timezone")),
        "ai_context": dict(p.get("ai_context") or {}),
        "config_history": list(p.get("config_history") or []),
        "setup_level": p.get("setup_level"),
        # A set-up started from nothing takes Home Assistant's own settings where they exist
        # (apply_home_assistant_defaults); a loaded profile keeps its own.
        "_new": not p,
    }


def apply_home_assistant_defaults(draft: dict, ha_config: dict) -> None:
    """Time zone, currency and (if the name is still empty) location name from Home Assistant, once, for a
    new set-up. Records what was taken in draft["_from_home_assistant"] (the Review screen says so)."""
    if not draft.get("_new") or "_from_home_assistant" in draft:
        return
    taken = {}
    tz = (ha_config or {}).get("time_zone")
    if tz:
        draft["timezone"] = taken["time zone"] = tz
        st.session_state.pop("timezone_pick", None)
    currency = currencies.from_home_assistant((ha_config or {}).get("currency"))
    if currency:
        draft["currency_code"], draft["currency"] = currency
        taken["currency"] = currency[0]
        st.session_state.pop("currency_pick", None)
    location = ((ha_config or {}).get("location_name") or "").strip()
    if location and location.lower() != "home" and not draft.get("profile_name"):
        draft["profile_name"] = taken["name"] = location
        st.session_state.pop("profile_name_input", None)
    draft["_from_home_assistant"] = taken


def heating_zones_of(profile: dict) -> str:
    """'signals' or 'none'. Profiles from before the choice existed: 'signals' if a zone is mapped."""
    value = (profile or {}).get("heating_zones")
    if value in (ZONES_NONE, ZONES_SIGNALS):
        return value
    zone_mapped = any(((profile or {}).get("mapping") or {}).get(z) not in (None, "", "None") for z in ZONE_ROLES)
    return ZONES_SIGNALS if zone_mapped else ZONES_NONE


def zones_on(draft: dict) -> bool:
    return draft.get("heating_zones") == ZONES_SIGNALS


def profile_signature(profile) -> str:
    return json.dumps(profile, sort_keys=True, default=str) if isinstance(profile, dict) else "none"


def reset_widgets() -> None:
    """Drop every Setup widget's own state so the widgets are drawn again from the draft."""
    for key in list(st.session_state):
        if key in WIDGET_KEYS or key.startswith(WIDGET_PREFIXES):
            st.session_state.pop(key, None)
    st.session_state.pop(ERRORS_KEY, None)


def get_draft(system_config) -> dict:
    """The session's draft; started (again) from the active profile when that changes."""
    base = profile_signature(system_config)
    if DRAFT_KEY not in st.session_state or st.session_state.get(DRAFT_BASE_KEY) != base:
        st.session_state[DRAFT_KEY] = new_draft(system_config if isinstance(system_config, dict) else None)
        st.session_state[DRAFT_BASE_KEY] = base
        reset_widgets()
    return st.session_state[DRAFT_KEY]


def replace_draft(profile: dict) -> dict:
    """Load another profile into Setup (the active profile is unchanged until Save)."""
    st.session_state[DRAFT_KEY] = new_draft(profile)
    reset_widgets()
    return st.session_state[DRAFT_KEY]


def set_role(draft: dict, role: str, entity, unit=None) -> None:
    """Change a sensor choice from outside its widget (Detect, source name adaptation)."""
    if entity in (None, "", "None"):
        draft["mapping"].pop(role, None)
    else:
        draft["mapping"][role] = entity
    if unit:
        draft["units"][role] = unit
    st.session_state.pop(f"map_{role}", None)
    if unit:
        st.session_state.pop(f"unit_{role}", None)


def mapped(draft: dict, role: str) -> bool:
    return draft["mapping"].get(role) not in (None, "", "None")


def _unit_for(role: str, unit) -> str | None:
    options = get_unit_options(role)
    if len(options) > 1 and options != ["unknown"]:
        return unit if unit in options else options[0]
    return options[0] if options else None


def weather_curve_to_save(curve: dict) -> dict:
    on = bool(curve.get("enabled"))
    values = {k: float(curve[k]) for k, _l, _s in CURVE_FIELDS}
    return {"enabled": bool(on and values["mild_outdoor_c"] != values["design_outdoor_c"]),
            "design_outdoor_c": values["design_outdoor_c"], "design_flow_c": values["design_flow_c"],
            "mild_outdoor_c": values["mild_outdoor_c"], "mild_flow_c": values["mild_flow_c"],
            "min_flow_c": values["min_flow_c"], "max_flow_c": values["max_flow_c"]}


def build_config(draft: dict) -> tuple[dict, list[str]]:
    """(profile to save, problems that block saving) from the draft, without drawing anything."""
    import mapping_ui

    mapping = {r: draft["mapping"][r] for r in KNOWN_ROLES if mapped(draft, r)}
    units = {}
    for role in mapping:
        if role.startswith(ROOM_SENSOR_PREFIX):
            continue  # room temperatures carry no unit choice
        unit = _unit_for(role, draft["units"].get(role))
        if unit:
            units[role] = unit
    rooms_per_zone = {
        z: ([r for r in draft["rooms_per_zone"].get(z, []) if r in mapping] if z in mapping else [])
        for z in ZONE_ROLES
    }
    # Zones off: the zone choices are kept aside in the profile (switching zones on restores them) but
    # are not part of the mapping, so the sources never fetch them and the engine never sees them.
    inactive_zones = {}
    if not zones_on(draft):
        zone_map = {z: mapping.pop(z) for z in ZONE_ROLES if z in mapping}
        if zone_map:
            inactive_zones = {"mapping": zone_map,
                              "units": {z: units.pop(z) for z in zone_map if z in units},
                              "rooms_per_zone": {z: rooms_per_zone[z] for z in zone_map if rooms_per_zone[z]}}
        rooms_per_zone = {z: [] for z in ZONE_ROLES}
    thresholds = {n.key: float(draft["thresholds"].get(n.key, n.default())) for n in NUMBERS if n.group == T}
    physics = {n.key: float(draft["physics_thresholds"].get(n.key, n.default())) for n in NUMBERS if n.group == P}
    phys = draft["physics_thresholds"]
    physics["heat_source"] = config.resolve_heat_source_mode(phys.get("heat_source", config.HEAT_SOURCE_DEFAULT))
    physics["fluid_specific_heat_kj"] = float(config.resolve_fluid_specific_heat_kj(
        phys.get("fluid_specific_heat_kj", config.FLUID_SPECIFIC_HEAT_DEFAULT_KJ)))
    physics["power_includes_immersion"] = bool(phys.get("power_includes_immersion", False))
    tariff, problems = mapping_ui._tariff_structure_from_rows(
        mapping_ui._tariff_rows_frame(draft["tariff_structure"]))
    ai = {}
    for k, prompt in AI_CONTEXT_PROMPTS.items():
        text = draft["ai_context"].get(k, "") or ""
        ai[k] = "" if text.strip() == (prompt.get("placeholder", "") or "").strip() else text
    profile = {
        "profile_name": (draft.get("profile_name") or "").strip() or DEFAULT_PROFILE_NAME,
        "created_at": datetime.now().isoformat(),
        "mapping": mapping,
        "units": units,
        "ai_context": ai,
        "config_history": list(draft["config_history"]),
        "rooms_per_zone": rooms_per_zone,
        "thresholds": thresholds,
        "physics_thresholds": physics,
        "weather_curve": weather_curve_to_save(draft["weather_curve"]),
        "tariff_structure": tariff,
        "currency": currencies.symbol_for(draft["currency_code"]) or draft["currency"],
        "currency_code": draft["currency_code"],
        "timezone": config.resolve_timezone(draft["timezone"]),
        "therm_version": "2.0",
        "heating_zones": ZONES_SIGNALS if zones_on(draft) else ZONES_NONE,
        "electricity_source": config.resolve_electricity_source(draft.get("electricity_source")),
    }
    if inactive_zones:
        profile["inactive_zones"] = inactive_zones
    if draft.get("setup_level"):
        profile["setup_level"] = draft["setup_level"]
    return profile, list(st.session_state.get(ERRORS_KEY, {}).get("tariff", [])) + problems


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    """What the section renderers need besides the draft."""
    options: list                     # "None" + every entity offered
    format_opt: Callable[[str], str]
    level: str = "full"               # "guided" draws the high-impact subset of a section
    temperature_options: list | None = None   # "None" + temperature sensors only (°C/°F), if units are known


def _seed(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def sensor_row(draft: dict, ctx: Ctx, role: str, label: str, *, required=False, help_text=None,
               options: list | None = None) -> str | None:
    """One sensor choice (and its unit, where there is a choice). Returns the entity or None."""
    options = options or ctx.options
    u_opts = get_unit_options(role)
    has_unit_choice = len(u_opts) > 1 and u_opts != ["unknown"]
    # Only split the row when there is a unit to choose; otherwise the sensor box gets the full width.
    c1, c2 = st.columns([2, 1]) if has_unit_choice else (st.container(), None)
    key = f"map_{role}"
    current = draft["mapping"].get(role, "None")
    if current not in (None, "", "None") and current not in options:
        options = options + [current]   # a saved choice stays selectable even if it no longer qualifies
    _seed(key, current if current in options else "None")
    with c1:
        sel = st.selectbox(f"**{label}**" + (" *" if required else ""), options, key=key, help=help_text,
                           format_func=ctx.format_opt)
    if has_unit_choice:
        unit_key = f"unit_{role}"
        saved = draft["units"].get(role, u_opts[0])
        _seed(unit_key, saved if saved in u_opts else u_opts[0])
        with c2:
            draft["units"][role] = st.selectbox("Unit", u_opts, key=unit_key)
    if sel in (None, "", "None"):
        draft["mapping"].pop(role, None)
        return None
    draft["mapping"][role] = sel
    return sel


def number(draft: dict, n: Number, *, value=None) -> float:
    """A narrow number input for one numeric setting."""
    target = draft[n.group]
    _seed(n.widget_key, float(target.get(n.key, n.default()) if value is None else value))
    col, _ = st.columns([1, 3])
    kwargs = {"help": n.help, "step": n.step, "min_value": n.min_value, "max_value": n.max_value}
    if n.fmt:
        kwargs["format"] = n.fmt
    with col:
        target[n.key] = st.number_input(n.label, key=n.widget_key, **kwargs)
    return target[n.key]


def _sensor_rows(draft: dict, ctx: Ctx, catalogue: dict, roles=None, required=False) -> None:
    for role in roles or list(catalogue):
        d = catalogue[role]
        sensor_row(draft, ctx, role, d["label"], required=required, help_text=d.get("description"))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def render_sensors_core(draft: dict, ctx: Ctx) -> None:
    st.markdown("##### Required")
    _sensor_rows(draft, ctx, REQUIRED_SENSORS, required=True)
    st.markdown("##### Recommended")
    _sensor_rows(draft, ctx, RECOMMENDED_SENSORS)


def render_extra_sensors(draft: dict, ctx: Ctx) -> None:
    st.caption("Optional readings that add detail: indoor unit power, compressor frequency, defrost, a heat meter.")
    roles = sorted(OTHER_ROLES, key=lambda r: OPTIONAL_SENSORS[r].get("label", r).lower())
    _sensor_rows(draft, ctx, OPTIONAL_SENSORS, roles)


def render_heat_calc(draft: dict, ctx: Ctx) -> None:
    phys = draft["physics_thresholds"]
    electricity_source_keys = list(config.ELECTRICITY_SOURCE_OPTIONS)
    _seed("electricity_source", config.resolve_electricity_source(draft.get("electricity_source")))
    draft["electricity_source"] = st.selectbox(
        "Electricity reading source", options=electricity_source_keys, key="electricity_source",
        format_func=lambda key: config.ELECTRICITY_SOURCE_OPTIONS[key],
        help=("Describe the reading used for electricity in COP. This records provenance only; "
              "it does not change therm's calculations."),
    )
    heat_source_keys = list(config.HEAT_SOURCE_OPTIONS.keys())
    if ctx.level == "full" or mapped(draft, "Heat"):
        _seed("phys_heat_source", config.resolve_heat_source_mode(phys.get("heat_source", config.HEAT_SOURCE_DEFAULT)))
        col_hs, _ = st.columns([2, 2])
        with col_hs:
            phys["heat_source"] = st.selectbox(
                "Heat output source", options=heat_source_keys, key="phys_heat_source",
                format_func=lambda k: config.HEAT_SOURCE_OPTIONS[k],
                help=(
                    "Choose the heat meter option if a real heat meter or the heat pump's own heat output reading "
                    "is mapped as Heat. Choose Calculate if you have no heat meter (for example a Heat sensor that "
                    "Home Assistant calculates itself): therm then calculates heat from flow rate × ΔT using the "
                    "fluid below, and any mapped Heat sensor is kept for comparison only and never used instead. "
                    "Calculating needs Flow Rate plus either a mapped Delta T sensor or mapped Flow and Return "
                    "temperatures. With no Heat sensor mapped, heat is always calculated."
                ),
            )

    # Primary-circuit fluid → heat coefficient for Heat = c × FlowRate × ΔT
    stored_c = config.resolve_fluid_specific_heat_kj(phys.get("fluid_specific_heat_kj",
                                                              config.FLUID_SPECIFIC_HEAT_DEFAULT_KJ))
    labels = list(config.FLUID_PRESETS.keys()) + ["Custom"]
    preset = next((label for label, v in config.FLUID_PRESETS.items() if abs(v - stored_c) < 0.005), "Custom")
    _seed("phys_fluid_choice", preset)
    col_fluid, _ = st.columns([1, 3])
    with col_fluid:
        choice = st.selectbox(
            "Primary circuit fluid", options=labels, key="phys_fluid_choice",
            help=(
                "The liquid in the heat pump's primary circuit. Heat output is calculated as specific heat × flow "
                "rate × ΔT, so this scales Heat, COP and SCOP directly (water against a glycol mix is about 7 %). "
                "Outdoor monobloc units usually contain a water/glycol antifreeze mix; choose Pure water if yours "
                "has none. Not used when heat comes from a mapped heat meter."
            ),
        )
    if choice == "Custom":
        _seed("phys_fluid_custom", float(stored_c))
        col_c, _ = st.columns([1, 3])
        with col_c:
            fluid_c = st.number_input(
                "Fluid specific heat (kJ/kg·K)", key="phys_fluid_custom", step=0.01, format="%.2f",
                min_value=config.FLUID_SPECIFIC_HEAT_MIN_KJ, max_value=config.FLUID_SPECIFIC_HEAT_MAX_KJ,
                help=("From the antifreeze datasheet at your typical flow temperature. Allowed range "
                      f"{config.FLUID_SPECIFIC_HEAT_MIN_KJ}–{config.FLUID_SPECIFIC_HEAT_MAX_KJ}. "
                      "Assumes a density of ~1 kg/L."),
            )
    else:
        fluid_c = config.FLUID_PRESETS[choice]
    phys["fluid_specific_heat_kj"] = float(fluid_c)
    st.caption(f"Heat = {config.heat_coefficient_w_per_lpm_k(float(fluid_c)):.2f} W per L/min of flow per °C of ΔT")

    _seed("phys_power_includes_immersion", bool(phys.get("power_includes_immersion", False)))
    phys["power_includes_immersion"] = st.checkbox(
        "Main power meter includes the immersion element", key="phys_power_includes_immersion",
        help=(
            "Tick if the sensor mapped to Power measures the whole unit including the immersion heater. Then "
            "immersion energy is not added to electricity totals a second time. Leave unticked if Power measures "
            "only the outdoor unit and the immersion shows up on Indoor Power. Check: when the immersion switches "
            "on, does Power jump by ~3 kW?"
        ),
    )


def render_hot_water(draft: dict, ctx: Ctx) -> None:
    st.caption(
        "How therm tells hot-water runs from heating runs. A DHW active status or the 3-way valve position is "
        "best; with neither, hot-water runs are recognised from the tank temperature rising."
    )
    _sensor_rows(draft, ctx, OPTIONAL_SENSORS, HOT_WATER_ROLES)


def room_slots(draft: dict) -> int:
    """Room slots to show: at least eight, and always up to the highest room in use."""
    used = [i for i, role in enumerate(ROOM_ROLES, start=1) if mapped(draft, role)]
    return min(max([ROOMS_SHOWN, st.session_state.get(ROOM_SLOTS_KEY, 0)] + used), len(ROOM_ROLES))


def render_rooms(draft: dict, ctx: Ctx) -> None:
    st.caption(f"Room temperature sensors, shown against heating runs. Optional; up to {len(ROOM_ROLES)}."
               + (" Only temperature sensors (°C or °F) are listed." if ctx.temperature_options else ""))
    slots = room_slots(draft)
    cols = st.columns(2)
    for i, role in enumerate(ROOM_ROLES[:slots]):
        with cols[i % 2]:
            sensor_row(draft, ctx, role, f"Room Sensor {i + 1}", options=ctx.temperature_options)
    if slots < len(ROOM_ROLES) and st.button("＋ Add another room", key="rooms_add"):
        st.session_state[ROOM_SLOTS_KEY] = slots + 1
        st.rerun()


def render_zones(draft: dict, ctx: Ctx) -> None:
    _seed("zones_choice", draft.get("heating_zones", ZONES_NONE))
    draft["heating_zones"] = st.radio(
        "How is your heating controlled?", list(ZONE_CHOICES), key="zones_choice",
        format_func=lambda k: ZONE_CHOICES[k],
        help=("If a separate controller runs your zone pumps (e.g. a wiring centre or control board switched by "
              "230 V thermostats), Home Assistant can't see the zones unless you add a signal for each pump, "
              "for example a relay or optocoupler on the pump supply."),
    )
    candidates = [c for c in (st.session_state.get("autodetect_result") or {}).get("zone_candidates", [])
                  if not mapped(draft, c["role"])]
    if not zones_on(draft):
        st.caption(
            "Most systems. This includes zone valves or pumps switched by room thermostats through a wiring "
            "centre, which therm can't see. Every room sensor is used for heating runs; hot-water runs use the "
            "tank temperature. **Heating while the tank is being reheated can't be identified** without zone "
            "pump signals."
        )
        if candidates:
            st.info("Detect found possible zone signals: " + ", ".join(
                f"`{ctx.format_opt(c['entity'])}` ({c['role'].replace('_', ' ')})" for c in candidates))
            if st.button("Use separate zones with these signals", key="zones_use_candidates"):
                for c in candidates:
                    set_role(draft, c["role"], c["entity"], c.get("unit"))
                draft["heating_zones"] = ZONES_SIGNALS
                st.session_state.pop("zones_choice", None)
                st.rerun()
        return
    st.caption(
        "For each zone, the signal that is on while the zone calls for heat (a pump, valve or thermostat), and "
        "the rooms it heats. therm then shows which zones were heating in each run, and flags heating while the "
        "tank is being reheated."
    )
    rooms = {r: draft["mapping"][r] for r in ROOM_ROLES if mapped(draft, r)}
    labels = {r: f"{ctx.format_opt(v)} ({r})" for r, v in rooms.items()}
    for z_key, z_d in ZONE_SENSORS.items():
        st.markdown(f"**{z_d['label']}**")
        z_s = sensor_row(draft, ctx, z_key, "Zone Pump/Valve (Binary)", help_text=z_d["description"])
        link_key = f"link_{z_key}"
        saved = [r for r in draft["rooms_per_zone"].get(z_key, []) if r in rooms] if z_s else []
        if not z_s:
            st.session_state[link_key] = []
        elif link_key not in st.session_state or any(r not in rooms for r in st.session_state[link_key]):
            st.session_state[link_key] = saved
        draft["rooms_per_zone"][z_key] = st.multiselect(
            f"Select rooms controlled by {z_d['label']}:", options=list(rooms), key=link_key,
            format_func=lambda x: labels[x], help="Select which room sensors belong to this zone.",
        )
        st.markdown("---")
    for n in ZONE_RUN_FILTER:
        number(draft, n)


def render_weather(draft: dict, ctx: Ctx) -> None:
    st.caption(
        "Map a secondary source (e.g. OpenWeatherMap) to fill gaps in the primary sensor (e.g. a local weather "
        "station). The primary reading is always used when it is available."
    )
    names = {"OutdoorTemp": "Temperature", "Outdoor_Humidity": "Humidity", "Wind_Speed": "Wind speed",
             "Solar_Rad": "Solar radiation"}
    for primary, secondary in WEATHER_FALLBACKS.items():
        st.markdown(f"##### {names.get(primary, primary)}")
        col_p, col_s = st.columns(2)
        with col_p:
            if primary in ENVIRONMENTAL_SENSORS:
                d = ENVIRONMENTAL_SENSORS[primary]
                sensor_row(draft, ctx, primary, "Primary", help_text=d.get("description"))
            else:
                # The heat pump's outdoor sensor is chosen with the heat pump sensors.
                current = draft["mapping"].get(primary)
                st.markdown("**Primary**")
                st.caption(f"{ctx.format_opt(current) if current else 'Not mapped'}: set as "
                           f"{RECOMMENDED_SENSORS[primary]['label']} with the heat pump sensors.")
        with col_s:
            d = SECONDARY_WEATHER_SENSORS[secondary]
            sensor_row(draft, ctx, secondary, "Secondary", help_text=d.get("description"))


def render_run_detection(draft: dict, ctx: Ctx) -> None:
    for n in RUN_DETECTION:
        number(draft, n)


def render_hydraulics(draft: dict, ctx: Ctx) -> None:
    for n in HYDRAULICS:
        number(draft, n)
    curve = draft["weather_curve"]
    st.markdown("**Weather curve (flow target for flow-limit detection)**")
    if mapped(draft, "Target_Flow"):
        st.caption("A Target Flow sensor is mapped, so it is used instead of this curve.")
    else:
        st.caption(
            "Enter two points from your heat pump's weather-compensation settings. The defaults are THERM's "
            "original curve; set your own or flow-limited minutes will be judged against a curve your system "
            "doesn't use."
        )
    _seed("wc_enabled", bool(curve.get("enabled")))
    curve["enabled"] = st.checkbox("Use weather curve", key="wc_enabled",
                                   help="Untick to disable flow-limit detection when no Target Flow sensor is mapped.")
    cols = st.columns(3)
    for i, (key, label, step) in enumerate(CURVE_FIELDS):
        _seed(f"wc_{key}", float(curve[key]))
        with cols[i // 2]:
            curve[key] = st.number_input(label, key=f"wc_{key}", step=step)
    if curve["mild_outdoor_c"] == curve["design_outdoor_c"]:
        st.warning("The two outdoor temperatures must differ; the curve is disabled until they do.")
    for n in HYDRAULICS_AFTER_CURVE:
        number(draft, n)


def render_diagnostics(draft: dict, ctx: Ctx) -> None:
    # Heating during DHW needs zone/pump signals (processing.detect_runs): the setting is shown only when a
    # zone is mapped, and kept unchanged otherwise.
    if zones_on(draft) and any(mapped(draft, z) for z in ZONE_ROLES):
        number(draft, HEATING_DURING_DHW)
    for n in DIAGNOSTICS:
        number(draft, n)


def render_costs(draft: dict, ctx: Ctx) -> None:
    st.caption("Used for the cost figures and to put days and tariff bands in your local time.")
    known = {code: (symbol, name) for code, symbol, name in currencies.CURRENCIES}
    code = draft.get("currency_code") or "EUR"
    _seed("currency_pick", code if code in known else OTHER_CURRENCY)
    pick = st.selectbox(
        "Currency", list(known) + [OTHER_CURRENCY], key="currency_pick",
        format_func=lambda c: f"{known[c][1]} ({known[c][0]})" if c in known else "Other (enter the code)",
        help="Used for prices and costs.",
    )
    if pick == OTHER_CURRENCY:
        _seed("currency_other", "" if code in known else code)
        typed = st.text_input("Currency code (3 letters, e.g. ZAR)", key="currency_other", max_chars=3)
        resolved = currencies.from_home_assistant(typed)
        if resolved:
            draft["currency_code"], draft["currency"] = resolved
        elif typed:
            st.warning("Enter a three-letter currency code (ISO 4217).")
    else:
        draft["currency_code"], draft["currency"] = pick, known[pick][0]
    try:
        from zoneinfo import available_timezones
        tz_options = sorted(available_timezones())
    except Exception:
        tz_options = [config.DEFAULT_TIMEZONE, "UTC"]
    tz_now = config.resolve_timezone(draft["timezone"])
    if "time zone" in draft.get("_from_home_assistant", {}):
        st.caption("Time zone and currency are taken from Home Assistant's settings.")
    if tz_now not in tz_options:
        tz_options = [tz_now] + tz_options
    _seed("timezone_pick", tz_now)
    draft["timezone"] = st.selectbox(
        "Time zone", tz_options, key="timezone_pick",
        help=("Where the heat pump is. Home Assistant exports timestamps in UTC; they are converted to this zone "
              "so tariff bands, daily totals and run times use local clock time, including summer time. Grafana "
              "exports are already local."),
    )
    render_tariff(draft, draft["currency"])


def render_tariff(draft: dict, currency: str) -> None:
    """Electricity prices: one table of bands, each with the date it applies from. A price change is new
    rows with a later 'Valid from'; every day is costed with the prices valid on that day."""
    import pandas as pd

    import mapping_ui

    st.markdown("**Electricity prices**")
    st.caption(
        "One row per price band. A single all-day price is one row from 00:00 to 24:00; day/night or peak prices "
        "are several rows with the same **Valid from**. When your prices change, add rows with the new date (or "
        "use *Add a price change*). Each day is costed with the prices valid on that day; the earliest prices "
        "also cover any older data."
    )
    signature = json.dumps(draft["tariff_structure"], sort_keys=True, default=str)
    # (Re)build the table from the draft when it changed underneath, or when the editor was not drawn in the
    # last run (its edits are already in the draft).
    if "tariff_editor" not in st.session_state or st.session_state.get("tariff_rows_signature") != signature:
        st.session_state["tariff_rows_signature"] = signature
        st.session_state["tariff_rows_df"] = mapping_ui._tariff_rows_frame(draft["tariff_structure"])
        st.session_state.pop("tariff_editor", None)
    edited = st.data_editor(
        st.session_state["tariff_rows_df"], num_rows="dynamic", hide_index=True, key="tariff_editor",
        column_config={
            "valid_from": st.column_config.DateColumn("Valid from", format="DD/MM/YYYY", required=True, width="small"),
            "name": st.column_config.TextColumn("Band"),
            "start": st.column_config.TextColumn("Start", required=True, width="small", help="HH:MM, e.g. 08:00"),
            "end": st.column_config.TextColumn("End", required=True, width="small",
                                               help="HH:MM; use 24:00 for midnight at the end of the day"),
            "rate": st.column_config.NumberColumn(f"{currency}/kWh", format="%.4f", width="small", step=0.001,
                                                  min_value=0.0, required=True),
        },
    )
    c_date, c_btn = st.columns([1, 2], vertical_alignment="bottom")
    with c_date:
        new_from = st.date_input("New prices from", key="tariff_new_from", format="DD/MM/YYYY")
    with c_btn:
        if st.button("Add a price change (copies the latest bands to edit)", key="tariff_add_change"):
            base = edited.dropna(subset=["valid_from"])
            if base.empty:
                base = mapping_ui._tariff_rows_frame([])
            latest = pd.to_datetime(base["valid_from"]).max()
            copied = base[pd.to_datetime(base["valid_from"]) == latest].copy()
            copied["valid_from"] = new_from
            edited = pd.concat([edited, copied], ignore_index=True)
            structure, errors = mapping_ui._tariff_structure_from_rows(edited)
            if not errors:
                draft["tariff_structure"] = structure
            st.session_state["tariff_rows_df"] = edited
            st.session_state.pop("tariff_editor", None)
            st.rerun()
    structure, errors = mapping_ui._tariff_structure_from_rows(edited)
    st.session_state.setdefault(ERRORS_KEY, {})["tariff"] = errors
    if errors:
        for err in errors:
            st.error(err)
        return
    draft["tariff_structure"] = structure
    st.session_state["tariff_rows_signature"] = json.dumps(structure, sort_keys=True, default=str)


def render_notes(draft: dict, ctx: Ctx) -> None:
    for k, p in AI_CONTEXT_PROMPTS.items():
        _seed(f"ai_{k}", draft["ai_context"].get(k, "") or "")
        draft["ai_context"][k] = st.text_area(p["label"], key=f"ai_{k}", placeholder=p.get("placeholder", ""),
                                              help=p["help"])

    st.markdown("**Heat pump configuration change log**")
    st.caption("Keep a simple log of configuration changes. You can backdate entries if needed.")
    st.session_state.setdefault("new_change_date", datetime.now().date())
    st.session_state.setdefault("new_change_time", datetime.now().time())
    st.session_state.setdefault("new_change_tag", "")
    st.session_state.setdefault("new_change_note", "")

    def _add() -> None:
        note = st.session_state.get("new_change_note", "").strip()
        if not note:
            st.session_state["config_hist_error"] = "Please enter a change note before adding."
            return
        d, t = st.session_state.get("new_change_date"), st.session_state.get("new_change_time")
        start = datetime.combine(d, t) if d and t else datetime.now()
        entry = {"start": start.isoformat(timespec="seconds"),
                 "config_tag": st.session_state.get("new_change_tag", "").strip(), "change_note": note}
        draft["config_history"] = [entry] + draft["config_history"]
        st.session_state["new_change_tag"] = ""
        st.session_state["new_change_note"] = ""
        st.session_state.pop("config_hist_error", None)

    def _delete(start_val: str) -> None:
        draft["config_history"] = [r for r in draft["config_history"] if str(r.get("start")) != str(start_val)]

    c_date, c_time = st.columns(2)
    with c_date:
        st.date_input("Date of change", key="new_change_date", max_value=datetime.now(), format="DD/MM/YYYY")
    with c_time:
        st.time_input("Time of change", key="new_change_time")
    st.text_input("Change tag (optional)", key="new_change_tag")
    st.text_area("Change note", key="new_change_note",
                 placeholder="e.g., Pump set to Constant Speed II; DHW target raised to 50C.")
    st.button("Add change", on_click=_add)
    if st.session_state.get("config_hist_error"):
        st.warning(st.session_state["config_hist_error"])

    history = sorted(draft["config_history"], key=lambda r: str(r.get("start", "")), reverse=True)
    if not history:
        st.info("No config changes logged yet.")
        return
    st.markdown("**Saved changes** (newest first)")
    for idx, row in enumerate(history):
        try:
            when = datetime.fromisoformat(str(row.get("start", ""))).strftime("%d-%m-%Y %H:%M")
        except ValueError:
            when = str(row.get("start", ""))
        c1, c2, c3 = st.columns([2, 6, 1])
        c1.write(when)
        c2.write(f"{row.get('config_tag', '')} - {row.get('change_note', '')}".strip(" -"))
        c3.button("Delete", key=f"del_hist_{idx}", on_click=_delete, args=(row.get("start", ""),))


# ---------------------------------------------------------------------------
# The Setup page (tabs); the wizard (onboarding.py) draws the same sections one at a time
# ---------------------------------------------------------------------------
PAGE_TABS = [
    ("Sensors", [("Heat pump sensors", render_sensors_core), ("Other sensors", render_extra_sensors)]),
    ("Heat & hot water", [("How heat is measured", render_heat_calc), ("Hot water detection", render_hot_water)]),
    ("Rooms & zones", [("Rooms", render_rooms), ("Heating zones", render_zones)]),
    ("Weather", [(None, render_weather)]),
    ("Costs & time zone", [(None, render_costs)]),
    ("Thresholds", [("Run detection", render_run_detection), ("Hydraulics & flow limits", render_hydraulics),
                    ("Diagnostics", render_diagnostics)]),
    ("Notes", [("AI report context & change log (optional)", render_notes)]),
]


def render_page(draft: dict, ctx: Ctx) -> None:
    """Every section, in tabs (all tabs are drawn each run, so no value is lost between them)."""
    tabs = st.tabs([name for name, _ in PAGE_TABS], key="setup_tab",
                   default=st.session_state.pop("setup_tab_default", None))
    for tab, (_name, sections) in zip(tabs, PAGE_TABS):
        with tab:
            for title, render in sections:
                if title:
                    st.subheader(title)
                render(draft, ctx)
