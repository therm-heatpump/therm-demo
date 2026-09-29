# config.py
CALC_VERSION = "v1.33.0"
TARIFF_PROFILE_ID = "Multi_Band_Smart_Tariff"

# A reusable heartbeat for periodic/numeric sensors must represent enough of
# the long analysis to be trusted. The expected report rate is still learned
# only from days when the sensor worked, so an outage never normalises itself.
HEARTBEAT_MIN_REPORTING_DAY_COVERAGE = 0.50

# Thresholds
MIN_HEAT_FREQ = 15.0          # Hz
BASELINE_JSON_PATH = "sensor_heartbeat_baseline_seasonal.json"

THRESHOLDS = {
    "short_cycle_min": 20,
    "very_short_cycle_min": 10,
    "flow_limit_tolerance": 2.0,
    "flow_limit_min_duration": 15,
    "hdd_base_temp": 18.0,
    "high_night_share": 0.50,
    "short_cycling_ratio_high": 0.3,
    "flow_over_43c_pct_high": 20,
    "dhw_scop_low": 2.2,
    "night_share_elec_low": 0.30,
    "heating_during_dhw_detection_pct": 0.15,     # Fraction of a DHW run with a zone pump on to flag heating during DHW (zone signals only)
    # Run detection guards
    "min_heating_run_minutes_with_no_zones": 8,
    "min_heating_run_heat_kwh_with_no_zones": 0.25,
}

NIGHT_HOURS = [2, 3, 4, 5]

# Room temperature sensors Setup offers (Room_1 … Room_MAX_ROOMS).
MAX_ROOMS = 16

# Weather-compensation curve used as the flow-temperature TARGET for the
# Virtual_FTlim (flow-limited minutes) metric when no Target_Flow sensor is
# mapped. Profile key "weather_curve" overrides any field. Straight line through
# (design_outdoor_c, design_flow_c) and (mild_outdoor_c, mild_flow_c), clamped
# to [min_flow_c, max_flow_c]. These defaults reproduce the original THERM
# curve: target = 45 - 0.625 x (outdoor + 2), clamped 22-43 degC.
DEFAULT_WEATHER_CURVE = {
    "enabled": True,
    "design_outdoor_c": -2.0,
    "design_flow_c": 45.0,
    "mild_outdoor_c": 18.0,
    "mild_flow_c": 32.5,
    "min_flow_c": 22.0,
    "max_flow_c": 43.0,
}

# ==========================================
# ENGINE STATE / ACTIVITY THRESHOLDS
# ==========================================

ENGINE_STATE_THRESHOLDS = {
    # Frequency-based detection (primary when available)
    "freq_on_hz": 10.0,
    "freq_off_hz": 5.0,   # optional if you later add hysteresis

    # Flow-based detection (fallback if no freq/power)
    "flow_on_lpm": 2.0,
    "flow_off_lpm": 1.0,

    # Temperature-based detection
    "delta_on_C": 1.0,          # minimum ΔT to count as active heating
    "delta_coast_min_C": 0.5,   # minimum ΔT for coast-down

    # Power-based activity detection (is_active = Power > power_on_W).
    # 150 W sits midway in the gap between standby (~0 W) and the lowest
    # sustained running draw (~340 W) seen on a Samsung Gen 6,
    # while excluding ~65 W auxiliary states (e.g. cold-night heater steps).
    "power_on_W": 150.0,

    # Coast-down lookback window (minutes)
    "coast_down_window_min": 5,

    # Minimum run duration (filter out noise runs)
    "minimum_run_duration_min": 5,
}


# ==========================================
# PHYSICS & HYDRAULIC CONSTANTS
# ==========================================
# Primary-circuit fluid. Heat is derived from hydraulics as
#   Heat [W] = c [kJ/kg·K] × 1000 / 60 × FlowRate [L/min] × ΔT [K]
# (assumes a fluid density of ~1 kg/L). The fluid is a profile setting
# (physics_thresholds["fluid_specific_heat_kj"]) because it varies by install.
FLUID_SPECIFIC_HEAT_DEFAULT_KJ = 3.9   # typical water + glycol antifreeze mix
FLUID_SPECIFIC_HEAT_MIN_KJ = 3.0       # plausible range; values outside fall back to default
FLUID_SPECIFIC_HEAT_MAX_KJ = 4.3
FLUID_PRESETS = {
    "Water + glycol antifreeze mix (3.9 kJ/kg·K)": 3.9,
    "Pure water (4.18 kJ/kg·K)": 4.18,
}


def resolve_fluid_specific_heat_kj(value) -> float:
    """Return a valid fluid specific heat (kJ/kg·K), or the default if missing/implausible."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return FLUID_SPECIFIC_HEAT_DEFAULT_KJ
    if not (FLUID_SPECIFIC_HEAT_MIN_KJ <= v <= FLUID_SPECIFIC_HEAT_MAX_KJ):
        return FLUID_SPECIFIC_HEAT_DEFAULT_KJ
    return v


# Where heat output comes from (physics_thresholds["heat_source"]).
HEAT_SOURCE_DEFAULT = "sensor"
HEAT_SOURCE_OPTIONS = {
    "sensor": "Use the Heat output sensor (heat meter) if one is mapped",
    "calculated": "Calculate in therm from flow rate x ΔT",
}


def resolve_heat_source_mode(value) -> str:
    """Return a valid heat-source setting ("sensor" or "calculated")."""
    return value if value in HEAT_SOURCE_OPTIONS else HEAT_SOURCE_DEFAULT


# Profile-level provenance for the electricity side of COP.  This deliberately
# describes the reading, rather than changing how any energy is calculated.
ELECTRICITY_SOURCE_DEFAULT = "unknown"
ELECTRICITY_SOURCE_OPTIONS = {
    "unknown": "I don't know / no dedicated heat-pump electricity reading",
    "separate_meter": "A dedicated meter measures heat-pump electricity",
    "heat_pump_estimate": "An estimate represents heat-pump electricity",
}
ELECTRICITY_SOURCE_LABELS = {
    "unknown": "electricity source unknown",
    "separate_meter": "electricity metered",
    "heat_pump_estimate": "electricity heat-pump estimate",
}


def resolve_electricity_source(value) -> str:
    """Return a canonical electricity-reading provenance, defaulting safely."""
    return value if value in ELECTRICITY_SOURCE_OPTIONS else ELECTRICITY_SOURCE_DEFAULT


def heat_coefficient_w_per_lpm_k(specific_heat_kj: float) -> float:
    """W of heat per (L/min of flow × °C of ΔT). 3.9 kJ/kg·K → 65.0."""
    return specific_heat_kj * 1000.0 / 60.0


ENGINE_HEAT_SOURCE_LABELS = {
    "native": "Heat sensor (mapped)",
    "derived_hydraulic": "Calculated from FlowRate x DeltaT",
    "unavailable": "Unavailable (required heat-source inputs missing)",
}

COP_HEAT_SOURCE_LABELS = {
    "calculated": "heat calculated",
    "mapped_sensor": "heat from mapped sensor",
    "unavailable": "heat unavailable",
    "unknown": "heat source unknown",
}


def cop_provenance(user_config, engine_heat_source=None) -> dict:
    """Describe the sources behind COP without altering its calculation.

    Engine provenance wins when it is available.  Older profiles have no
    electricity-source field, so they remain explicitly ``unknown`` instead
    of being guessed from a Power mapping.
    """
    cfg = user_config if isinstance(user_config, dict) else {}
    phys = cfg.get("physics_thresholds") or {}
    mapping = cfg.get("mapping") or {}
    if engine_heat_source == "native":
        heat_source = "mapped_sensor"
    elif engine_heat_source == "derived_hydraulic":
        heat_source = "calculated"
    elif engine_heat_source == "unavailable":
        heat_source = "unavailable"
    else:
        setting = resolve_heat_source_mode(phys.get("heat_source"))
        if setting == "calculated":
            heat_source = "calculated"
        elif "Heat" in mapping:
            heat_source = "mapped_sensor"
        else:
            heat_source = "unknown"
    electricity_source = resolve_electricity_source(cfg.get("electricity_source"))
    heat_source_display = COP_HEAT_SOURCE_LABELS[heat_source]
    electricity_source_display = ELECTRICITY_SOURCE_LABELS[electricity_source]
    return {
        "heat_source": heat_source,
        "electricity_source": electricity_source,
        "heat_source_display": heat_source_display,
        "electricity_source_display": electricity_source_display,
        "display": f"{heat_source_display} · {electricity_source_display}",
    }


def physics_assumptions(user_config, engine_heat_source=None) -> dict:
    """Heat-calculation assumptions for AI exports, so readers know what COP rests on.

    engine_heat_source is the source the engine actually used
    (df.attrs["heat_source"]); the profile setting is only a request. Sensor
    mode can fall back to hydraulics when the sensor has no data, while strict
    calculated mode reports unavailable when required inputs are missing.
    """
    cfg = user_config if isinstance(user_config, dict) else {}
    phys = cfg.get("physics_thresholds") or {}
    mapping = cfg.get("mapping") or {}
    c = resolve_fluid_specific_heat_kj(phys.get("fluid_specific_heat_kj"))
    setting = resolve_heat_source_mode(phys.get("heat_source"))
    provenance = cop_provenance(cfg, engine_heat_source)
    if engine_heat_source in ENGINE_HEAT_SOURCE_LABELS:
        heat_source = ENGINE_HEAT_SOURCE_LABELS[engine_heat_source]
    else:
        heat_source = (
            "Heat sensor (mapped)" if "Heat" in mapping and setting == "sensor"
            else "Calculated from FlowRate x DeltaT"
        ) + " (from profile; engine source not recorded)"
    return {
        "heat_source": heat_source,
        "heat_source_setting": HEAT_SOURCE_OPTIONS[setting],
        "cop_provenance": provenance,
        "electricity_source": provenance["electricity_source"],
        "electricity_source_display": provenance["electricity_source_display"],
        "fluid_specific_heat_kj_per_kg_k": c,
        "heat_coefficient_w_per_lpm_per_k": round(heat_coefficient_w_per_lpm_k(c), 2),
        "immersion_electricity": (
            "included in main Power meter" if phys.get("power_includes_immersion")
            else "metered separately (Indoor Power) and added to electricity totals"
        ),
        "heat_accounting": (
            "Net (EN 14511 / EN 14825 convention): heat taken back from the water "
            "circuit during defrost and at run start/stop is subtracted from heat output, COP and SCOP"
        ),
        "flow_target": (
            {"source": "Target_Flow sensor (mapped)"}
            if "Target_Flow" in mapping
            else {"source": "profile weather curve", **resolve_weather_curve(cfg)}
        ),
    }


def resolve_weather_curve(user_config) -> dict:
    """DEFAULT_WEATHER_CURVE with any profile "weather_curve" fields applied."""
    cfg = user_config if isinstance(user_config, dict) else {}
    curve = dict(DEFAULT_WEATHER_CURVE)
    for key, value in (cfg.get("weather_curve") or {}).items():
        if key not in curve:
            continue
        try:
            curve[key] = bool(value) if key == "enabled" else float(value)
        except (TypeError, ValueError):
            pass
    return curve


# Local time zone of the installation. Home Assistant exports timestamps in UTC;
# they are converted to this zone so tariff bands, day boundaries and run times
# are local wall-clock (Grafana exports are already local). Profile key "timezone".
DEFAULT_TIMEZONE = "Europe/Dublin"


def resolve_timezone(name) -> str:
    """Return a valid IANA time-zone name, or DEFAULT_TIMEZONE if missing/invalid."""
    try:
        from zoneinfo import ZoneInfo
        if isinstance(name, str) and name.strip():
            ZoneInfo(name.strip())
            return name.strip()
    except Exception:
        pass
    return DEFAULT_TIMEZONE


# Calculation Thresholds (Gatekeepers)
PHYSICS_THRESHOLDS = {
    "fluid_specific_heat_kj": FLUID_SPECIFIC_HEAT_DEFAULT_KJ,
    # Defrost episodes (from the Defrost / Defrost_Is_Active signal).
    # Real Samsung Gen 6 defrosts last ~4-6 min. The status is reported on change,
    # so a missed "0" can hold the flag on for hours; cap each episode.
    "defrost_max_minutes": 15,
    # Negative heat within this many minutes of an episode counts as defrost loss
    # (sensor lag and recovery); the rest counts as start/stop (transient) loss.
    "defrost_margin_min": 3,
    # Numeric defrost status codes that do NOT mean "defrosting". Samsung Gen 6:
    # 2 = starting, 3 = main defrost, 4-6 = continuing/ending (compressor on,
    # negative dT); 7 = finished/standby (compressor mostly off, dT ~ +9 C,
    # usually 1 min but sometimes held for hours). Plain 0/1 sensors unaffected.
    "defrost_ignore_codes": [7],
    # Cooling detection (active cooling = flow colder than return while running).
    # All must hold for >= cooling_min_minutes: running, not DHW, not near a
    # defrost, dT <= -cooling_dt_c, FlowTemp < cooling_max_flow_c and (if an
    # outdoor sensor exists) OutdoorTemp > cooling_min_outdoor_c. Defrosts last
    # 4-6 min in cold weather; valve transitions 1-5 min.
    "cooling_min_minutes": 10,
    "cooling_dt_c": 0.5,
    "cooling_max_flow_c": 22.0,
    "cooling_min_outdoor_c": 15.0,
    "cooling_gap_tolerance_min": 2,
    # Immersion: with an immersion signal, the element counts as on only while the
    # indoor unit draws more than this (pumps ~100 W; element ~3 kW).
    "immersion_confirm_power_w": 500,
    # Without a signal, Indoor_Power above this is assumed to be the element.
    "immersion_heuristic_power_w": 2500,
    # True if the main Power meter already includes the immersion element (then
    # immersion energy is not added to electricity totals a second time).
    # Samsung Gen 6 with a separately wired immersion: False (the unit's power moves +2 W when
    # the element adds ~3 kW).
    "power_includes_immersion": False,
    "min_flow_rate_lpm": 3.0,       # Minimum flow to count as "moving water"
    "min_freq_for_delta_t": 5.0,    # Compressor Hz required to trust Delta T
    "min_freq_for_heat": 7.5,       # Compressor Hz required to calculate Heat Output
    "max_valid_delta_t": 15.0,      # Sanity check: Delta T shouldn't exceed this
    "min_valid_delta_t": 0.2        # Ignore tiny fluctuations
}

# ==========================================
# SENSOR GROUPS (Visual Display Order)
# ==========================================
SENSOR_GROUPS = {
    "⚡ Power & Energy": ['Power', 'Indoor_Power', 'Heat'],
    "💧 Hydraulics": ['DeltaT', 'FlowRate', 'FlowTemp', 'ReturnTemp', 'Target_Flow', 'Pump_Primary', 'Pump_Secondary', 'ValveMode'],
    "🌤️ Environment (Primary)": ['OutdoorTemp', 'Solar_Rad', 'Wind_Speed', 'Outdoor_Humidity'],
    "🌤️ Environment (Secondary)": ['OutdoorTemp_Secondary', 'Solar_Rad_Secondary', 'Wind_Speed_Secondary', 'Outdoor_Humidity_Secondary'],
    "⚙️ System State": ['Heat_Pump_Active', 'DHW_Mode', 'Immersion_Mode', 'Quiet_Mode', 'DHW_Temp'],
    "🏠 Zones": ['Zone_1', 'Zone_2', 'Zone_3', 'Zone_4'],
    "ℹ️ Events": ['Defrost']
}

# ==========================================
# SENSOR EXPECTATION MODES (Data Quality)
# ==========================================
# system          - expected every minute the system is recording
# heating_active  - expected while the unit runs (counted within is_active)
# dhw_active      - expected during hot water (counted within is_DHW)
# system_slow     - sparse (hourly) sensors, forward-filled onto the minute grid
# on_change       - state sensors that only report when their value changes (can
#                   be months apart). Not scored: silence means "unchanged", not
#                   "missing". The view shows state changes per day instead.
# event_only      - rare events (defrost, immersion, quiet mode). Not scored.
UNSCORED_DQ_MODES = ("on_change", "event_only")

SENSOR_EXPECTATION_MODE = {
    # --- POWER & ENERGY ---
    'Power': 'system',
    'Indoor_Power': 'system',
    'Heat': 'heating_active',
    
    # --- HYDRAULICS ---
    'FlowTemp': 'system',
    'ReturnTemp': 'system',
    'FlowRate': 'heating_active',
    'DeltaT': 'heating_active',
    'COP_Raw': 'heating_active',
    'DHW_Temp': 'dhw_active',

    # --- ENVIRONMENT (Primary) ---
    'OutdoorTemp': 'system',
    'Outdoor_Humidity': 'system',
    'Solar_Rad': 'system',
    'Wind_Speed': 'system',
    
    # --- ENVIRONMENT (Secondary / fallback - may be sparse, e.g. OpenWeatherMap) ---
    'OutdoorTemp_Secondary': 'system_slow',
    'Outdoor_Humidity_Secondary': 'system_slow',
    'Wind_Speed_Secondary': 'system_slow',
    'Solar_Rad_Secondary': 'system_slow',

    # --- ROOM TEMPERATURES (generic placeholders; user-configured) ---
    'Room_1': 'system',
    'Room_2': 'system',
    'Room_3': 'system',
    'Room_4': 'system',
    'Room_5': 'system',
    'Room_6': 'system',
    'Room_7': 'system',
    'Room_8': 'system',

    # --- EVENTS / BINARY STATE (Neutral Grey Scoring) ---
    'Immersion_Mode': 'event_only',
    'Quiet_Mode': 'event_only',
    'Defrost': 'event_only',

    # --- SYSTEM STATE (Scored) ---
    'Zone_1': 'on_change',
    'Zone_2': 'on_change',
    'Zone_3': 'on_change',
    'Pump_Primary': 'on_change',
    'Pump_Secondary': 'on_change',
    'Heat_Pump_Active': 'on_change',
    'ValveMode': 'on_change',
    'DHW_Mode': 'on_change',
    'DHW_Active': 'on_change',
    'Zone_4': 'on_change',
    # Flow setpoint: reports when the target changes
    'Target_Flow': 'on_change',
}

# ==========================================
# SENSOR ROLES (Heartbeat Learning)
# ==========================================
# Every room slot, not only the first eight listed above.
SENSOR_EXPECTATION_MODE.update({f"Room_{i}": "system" for i in range(1, MAX_ROOMS + 1)})

SENSOR_ROLES = {
    # Core periodic sensors (with prefix for HA, without for Grafana)
    'sensor.heat_pump_power_ch1': 'core_periodic',
    'heat_pump_power_ch1': 'core_periodic',
    'sensor.heat_pump_heat_output': 'core_periodic',
    'heat_pump_heat_output': 'core_periodic',
    'sensor.heat_pump_flow_temperature': 'core_periodic',
    'heat_pump_flow_temperature': 'core_periodic',
    'sensor.heat_pump_return_temperature': 'core_periodic',
    'heat_pump_return_temperature': 'core_periodic',
    'sensor.heat_pump_flow_rate': 'core_periodic',
    'heat_pump_flow_rate': 'core_periodic',
    'sensor.heat_pump_indoor_power': 'core_periodic',
    'heat_pump_indoor_power': 'core_periodic',
    'sensor.heat_pump_compressor_frequency': 'core_periodic',
    'heat_pump_compressor_frequency': 'core_periodic',
    'sensor.heat_pump_flow_delta': 'core_periodic',
    'heat_pump_flow_delta': 'core_periodic',
    'sensor.heat_pump_outdoor_temperature': 'core_periodic',
    'heat_pump_outdoor_temperature': 'core_periodic',
    'sensor.weather_solar_radiation': 'core_periodic',
    'weather_solar_radiation': 'core_periodic',
    'sensor.weather_wind_speed': 'core_periodic',
    'weather_wind_speed': 'core_periodic',
    'sensor.weather_humidity': 'core_periodic',
    'weather_humidity': 'core_periodic',

    # Room temperature sensors (generic placeholders; user-configured)
    'sensor.room_1_temperature': 'room_temp',
    'room_1_temperature': 'room_temp',
    'sensor.room_2_temperature': 'room_temp',
    'room_2_temperature': 'room_temp',
    'sensor.room_3_temperature': 'room_temp',
    'room_3_temperature': 'room_temp',
    'sensor.room_4_temperature': 'room_temp',
    'room_4_temperature': 'room_temp',
    'sensor.room_5_temperature': 'room_temp',
    'room_5_temperature': 'room_temp',
    'sensor.room_6_temperature': 'room_temp',
    'room_6_temperature': 'room_temp',
    'sensor.room_7_temperature': 'room_temp',
    'room_7_temperature': 'room_temp',
    'sensor.room_8_temperature': 'room_temp',
    'room_8_temperature': 'room_temp',
    
    # Backup OpenWeather Sensors (SPARSE)
    'sensor.openweathermap_temperature': 'weather_sparse',
    'sensor.openweathermap_humidity': 'weather_sparse',
    'sensor.openweathermap_wind_speed': 'weather_sparse',
    'sensor.openweathermap_uv_index': 'weather_sparse',
    # Unprefixed OpenWeather sensors (Grafana compatibility)
    'openweathermap_temperature': 'weather_sparse',
    'openweathermap_humidity': 'weather_sparse',
    'openweathermap_wind_speed': 'weather_sparse',
    'openweathermap_uv_index': 'weather_sparse',

    # Binary state sensors (generic placeholders)
    'binary_sensor.zone_1': 'binary_state',
    'zone_1': 'binary_state',
    'binary_sensor.zone_2': 'binary_state',
    'zone_2': 'binary_state',
    'binary_sensor.zone_3': 'binary_state',
    'zone_3': 'binary_state',
    'binary_sensor.primary_pump': 'binary_state',
    'primary_pump': 'binary_state',
    'binary_sensor.secondary_pump': 'binary_state',
    'secondary_pump': 'binary_state',
    'binary_sensor.heat_pump_in_operation': 'binary_state',
    'heat_pump_in_operation': 'binary_state',
    'sensor.heat_pump_immersion_heater_mode_value': 'binary_state',
    'heat_pump_immersion_heater_mode_value': 'binary_state',
    'switch.quiet_mode': 'binary_state',
    'quiet_mode': 'binary_state',
    'sensor.heat_pump_3way_valve_position_value': 'binary_state',
    'heat_pump_3way_valve_position_value': 'binary_state',
    'sensor.heat_pump_hot_water_mode_value': 'binary_state',
    'heat_pump_hot_water_mode_value': 'binary_state',
    'sensor.heat_pump_hot_water_status_value': 'binary_state',
    'heat_pump_hot_water_status_value': 'binary_state',
    'sensor.heat_pump_hot_water_temperature': 'core_periodic',
    'heat_pump_hot_water_temperature': 'core_periodic',
    'sensor.heat_pump_defrost_status': 'rare_event',
    'heat_pump_defrost_status': 'rare_event',

    # Mapped/internal column names (used after schema mapping)
    'Zone_1': 'binary_state',
    'Zone_2': 'binary_state',
    'Zone_3': 'binary_state',
    'Zone_4': 'binary_state',
    'Room_1': 'room_temp',
    'Room_2': 'room_temp',
    'Room_3': 'room_temp',
    'Room_4': 'room_temp',
    'Room_5': 'room_temp',
    'Room_6': 'room_temp',
    'Room_7': 'room_temp',
    'Power': 'core_periodic',
    'FlowTemp': 'core_periodic',
    'ReturnTemp': 'core_periodic',
    'FlowRate': 'core_periodic',
    'OutdoorTemp': 'core_periodic',
    'Freq': 'core_periodic',
    'Indoor_Power': 'core_periodic',
    'DHW_Temp': 'core_periodic',
    'DHW_Active': 'binary_state',
    'DHW_Mode': 'binary_state',
    'ValveMode': 'binary_state',
    'Defrost': 'rare_event',
    'Target_Flow': 'setpoint',
    # Primary weather sensors: change-driven, so a steady reading is not an
    # outage. Held across normal silences before the secondary source fills in.
    'Outdoor_Humidity': 'weather_on_change',
    'Wind_Speed': 'weather_on_change',
    'Solar_Rad': 'weather_on_change',
    'OutdoorTemp_Secondary': 'weather_sparse',
    'Outdoor_Humidity_Secondary': 'weather_sparse',
    'Wind_Speed_Secondary': 'weather_sparse',
    'Solar_Rad_Secondary': 'weather_sparse',
}

# ==========================================
# ENTITY MAPPING (Raw -> Friendly)
# ==========================================
ENTITY_MAP = {
    'sensor.heat_pump_power_ch1': 'Power',
    'sensor.heat_pump_heat_output': 'Heat',
    'sensor.heat_pump_indoor_power': 'Indoor_Power',
    'sensor.heat_pump_immersion_heater_mode_value': 'Immersion_Mode',
    'sensor.heat_pump_flow_delta': 'DeltaT',
    'sensor.heat_pump_compressor_frequency': 'Freq',
    'sensor.heat_pump_flow_rate': 'FlowRate',
    'sensor.heat_pump_cop': 'COP_Raw',
    'binary_sensor.heat_pump_in_operation': 'Heat_Pump_Active', 
    'sensor.heat_pump_flow_temperature': 'FlowTemp',
    'sensor.heat_pump_return_temperature': 'ReturnTemp',
    'sensor.heat_pump_outdoor_temperature': 'OutdoorTemp',
    'sensor.heat_pump_hot_water_temperature': 'DHW_Temp',
    'sensor.heat_pump_hot_water_mode_value': 'DHW_Mode',
    'sensor.room_1_temperature': 'Room_1',
    'sensor.room_2_temperature': 'Room_2',
    'sensor.room_3_temperature': 'Room_3',
    'sensor.room_4_temperature': 'Room_4',
    'sensor.room_5_temperature': 'Room_5',
    'sensor.room_6_temperature': 'Room_6',
    'sensor.room_7_temperature': 'Room_7',
    'sensor.room_8_temperature': 'Room_8',
    'binary_sensor.zone_1': 'Zone_1',
    'binary_sensor.zone_2': 'Zone_2',
    'binary_sensor.zone_3': 'Zone_3',
    'switch.quiet_mode': 'Quiet_Mode',
    'binary_sensor.primary_pump': 'Pump_Primary',
    'binary_sensor.secondary_pump': 'Pump_Secondary',
    'sensor.heat_pump_defrost_status': 'Defrost',
    'sensor.heat_pump_3way_valve_position_value': 'ValveMode',

    # --- OPENWEATHER (secondary weather source) ---
    'sensor.openweathermap_temperature': 'OutdoorTemp_Secondary',
    'sensor.openweathermap_humidity': 'Outdoor_Humidity_Secondary',
    'sensor.openweathermap_wind_speed': 'Wind_Speed_Secondary',
}

# Every room slot has the room temperature role (Room_8 was missing).
SENSOR_ROLES.update({f"Room_{i}": "room_temp" for i in range(1, MAX_ROOMS + 1)})

ZONE_TO_ROOM_MAP = {
    'Zone_1': ['Room_1', 'Room_2'],
    'Zone_2': ['Room_3', 'Room_4'],
    'Zone_3': ['Room_5', 'Room_6'],
}

# Definitions of the diagnostic metrics named in AI_SYSTEM_CONTEXT, shipped in
# the LONG_TERM_TRENDS payload. Every metric the prompt names must be emitted by a
# payload (checked by the test suite).
AI_METRIC_DEFINITIONS = {
    "Global_SCOP": "Daily COP (name kept for compatibility): daily net heat / daily heat-pump electricity, outdoor unit only (SEPEMO H2). Immersion is excluded and reported as Immersion_kWh. Same as Daily_COP.",
    "Daily_COP_H4": "Whole-system daily COP (SEPEMO H4): as Daily_COP plus indoor unit electricity (controls + water pump, incl. idle standby; immersion excluded). Heating_COP_H4 / DHW_COP_H4 add indoor power only while that mode runs.",
    "Heating_COP": "Heat_Heating_kWh / Electricity_Heating_kWh (empty below 0.01 kWh). DHW_COP likewise for hot water, heat pump only.",
    "Starts": "Detected runs starting that day (all run types). Starts_Heating / Starts_DHW split by type.",
    "Short_Cycles_Count": "Heating runs shorter than thresholds.short_cycle_min minutes (Very_Short_Cycles_Count: shorter than very_short_cycle_min).",
    "Cycling_Severity_Index": "Short_Cycles_Count / Starts_Heating (0-1; empty with no heating starts). Above thresholds.short_cycling_ratio_high is unhealthy.",
    "Median_Run_Time_Mins": "Median heating run length that day.",
    "DHW_SCOP": "Heat_DHW_kWh / Electricity_DHW_kWh (empty below 0.01 kWh). Below thresholds.dhw_scop_low is inefficient.",
    "Night_Share_Of_Total_HP_Elec": "Share of metered heat-pump electricity used at the cheapest tariff rate in force that day (empty on flat-rate days).",
    "Effective_Avg_Tariff": "Daily_Cost_Euro / Metered_Electricity_kWh: the average price actually paid per kWh.",
    "Virtual_FTlim_Time_Mins": "Heating minutes (defrost excluded) where the flow target exceeded FlowTemp by more than thresholds.flow_limit_tolerance, counted only in stretches of at least flow_limit_min_duration minutes. Virtual_FTlim_Events counts those stretches. Both are unavailable when FTlim_Input_Coverage is below 90%.",
    "DQ_Tier": "Gold: Power coverage (DQ_Score) and Heat_Input_Coverage both >= 90%. Heat_Input_Coverage follows the engine heat source: the mapped Heat meter (physics_assumptions.heat_source), or all hydraulic inputs when heat is calculated. Silver: Power >= 90% but heat-input coverage is insufficient or unavailable. Bronze: Power coverage below 90%.",
}

AI_SYSTEM_CONTEXT_CORE = """
THERM AI ANALYSIS CORE:
Use only fields present in the payload. Distinguish measured facts from estimates,
state missing evidence explicitly, and scope conclusions to the period/window supplied.
All heat, COP and SCOP figures are net of defrost and transient heat taken back.
"""

AI_SYSTEM_CONTEXT_LONG_TERM = """
DEFAULT AI PROMPT FOR THERM ANALYSIS:
You are an expert heat pump diagnostics engine. Analyse the dataset using the structure and style below. Keep responses concise, bullet-driven, and reference the data (metrics/dates).

Executive Summary (2-3 sentences)
- State clearly whether the system is healthy or unhealthy.
- Identify major problems or improvements at a glance.

1. Flow Performance & Restriction
- Evaluate Target vs Actual flow temperature alignment (physics_assumptions.flow_target says whether the target is a sensor or the configured weather curve).
- Identify days with high Virtual_FTlim_Time_Mins / Virtual_FTlim_Events (for a single run: flow_limited_mins) and explain the likely cause (air/flow restriction vs aggressive curve vs mixing).
- Compare pre/post configuration change if applicable.

2. Cycling Behaviour & Run Stability
- Report Starts, Starts_Heating, Short_Cycles_Count and Cycling_Severity_Index (for a single run: is_short_cycle).
- Highlight unhealthy days (short cycles below the short-cycle threshold) or short runs.
- Use period_summary.run_detection and daily Run_Detection_Mins to distinguish fully covered,
  partially covered and uncovered days; never treat missing run-detection coverage as zero cycling.
- If a configuration change occurred, compare before/after.

3. Efficiency & Economics
- Report Global_SCOP trend (or run COP).
- Identify days/runs with poor SCOP/COP and explain why.
- Evaluate tariff usage (Night_Share_Of_Total_HP_Elec, Effective_Avg_Tariff).
- State whether the "super-heating" strategy is working.

4. DHW Performance
- Analyse DHW_SCOP (or run COP) and flag inefficiency.
- Explain likely causes (e.g., high tank setpoint).
- Give specific actions to improve DHW efficiency.

5. Summary Table
- Hydraulics (Good/Poor)
- Cycling (Good/Poor)
- Economics (Good/Poor)
- DHW (Good/Poor)

6. Final Verdict
- Provide 2-3 practical sentences about overall health and the next step.

CONTEXT NOTES:
- Use the system_settings fields (hp_model, property_context, operational_goals, tariff) provided in the JSON; do not assume defaults.
- physics_assumptions states how heat output was obtained. When calculated, heat (and so COP) scales directly with the fluid specific heat used; treat absolute COP accordingly.
- Active cooling (run_type "Cooling") is excluded from all heating heat, COP and SCOP figures and reported separately: cooling runs give heat removed (kWh) and EER; daily Cooling_Heat_Removed_kWh / Electricity_Cooling_kWh / Cooling_EER.
- Daily fields are in daily_metrics and period totals in period_summary; metric_definitions defines each metric. If a named metric is absent from the payload (e.g. no multi-rate tariff, or no flow target), say so rather than estimating it.
- All heat, COP and SCOP figures are NET of heat taken back during defrost and run start/stop. Defrost_Events, Defrost_Heat_Loss_kWh and Transient_Heat_Loss_kWh (daily) and heat_accounting (per run) give the breakdown; use them to assess defrost frequency against outdoor temperature and humidity.
"""

AI_RUN_METRIC_DEFINITIONS = {
    "run_cop": "Net heat delivered or removed divided by run electricity.",
    "flow_limited_mins": "Heating minutes in qualifying flow-limited stretches; null when valid target/FlowTemp coverage is below 90%.",
    "flow_limit_input_coverage": "Fraction of eligible heating minutes with both a valid flow target and FlowTemp.",
    "target_flow_source": "Mapped Target_Flow sensor, configured profile weather curve, or fixed DHW assumption.",
    "is_short_cycle": "Whether this run is shorter than short_cycle_threshold_minutes.",
}

AI_SYSTEM_CONTEXT_SINGLE_RUN = """
SINGLE RUN INSPECTOR:
- Analyse only the run described by meta.timestamp_start, meta.timestamp_end and meta.run_type.
- Report economics.run_cop and the net/gross/loss breakdown in heat_accounting.
- For heating, compare diagnostics_physics.target_flow_temp_avg with avg_flow_temp_c,
  state diagnostics_physics.target_flow_source, and report flow_limited_mins only when
  flow_limit_input_coverage is at least 0.90.
- Report run_characteristics.is_short_cycle against short_cycle_threshold_minutes.
- Use environmental_conditions and compressor_stats only where values are present.
- metric_definitions defines the run-level diagnostic fields. If a field is null, say
  the evidence is unavailable rather than estimating it.
"""


def build_ai_system_context(report_type: str) -> str:
    """Return shared plus report-specific AI instructions."""
    report = str(report_type or "").upper()
    specific = (
        AI_SYSTEM_CONTEXT_SINGLE_RUN
        if report == "SINGLE_RUN_INSPECTOR"
        else AI_SYSTEM_CONTEXT_LONG_TERM
    )
    return f"{AI_SYSTEM_CONTEXT_CORE.strip()}\n\n{specific.strip()}"


# Backward-compatible alias for callers outside the two production views.
AI_SYSTEM_CONTEXT = build_ai_system_context("LONG_TERM_TRENDS")
