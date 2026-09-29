# schema_defs.py

# ==========================================
# PART 1: CONSTANTS & PREFIXES
# ==========================================
from config import MAX_ROOMS

ROOM_SENSOR_PREFIX = "Room_"
ZONE_SENSOR_PREFIX = "Zone_"

# ==========================================
# PART 2: UI MAPPING DEFINITIONS
# ==========================================

# 1. Essential Sensors
REQUIRED_SENSORS = {
    "Power": {
        "label": "Heat Pump Power (Elec)",
        "unit": "W",
        "required": True,
        "description": "Total electrical power consumption of the heat pump unit."
    },
    "FlowTemp": {
        "label": "Flow Temperature",
        "unit": "degC",
        "required": True,
        "description": "Temperature of water leaving the heat pump."
    },
    "ReturnTemp": {
        "label": "Return Temperature",
        "unit": "degC",
        "required": True,
        "description": "Temperature of water returning to the heat pump."
    },
}

# 2. Recommended Sensors (High Priority)
# These were causing the duplicate error because they were aliased.
# Now they are distinct.
RECOMMENDED_SENSORS = {
    "FlowRate": {
        "label": "Flow Rate",
        "unit": "L/min",
        "required": False,
        "description": "Water flow rate in the primary circuit. Required for accurate Heat output and COP. If not mapped, the app runs in 'Power & Temps only' mode with no energy or COP metrics."
    },
    "OutdoorTemp": {
        "label": "Outdoor Temperature",
        "unit": "degC",
        "required": False,
        "description": "External ambient air temperature."
    },
}

# 3. Optional Sensors (Advanced / DHW)
OPTIONAL_SENSORS = {
    # DHW Sensors
    "DHW_Temp": {
        "label": "DHW Tank Temperature",
        "unit": "degC",
        "required": False,
        "description": "Current temperature of the hot water cylinder."
    },
    "DHW_Mode": {
        "label": "DHW Mode",
        "unit": "Text/Binary",
        "required": False,
        "description": "DHW Mode e.g. Eco, Standard, Power."
    },
    "ValveMode": {
        "label": "3-Way Valve Position",
        "unit": "Text",
        "required": False,
        "description": "Position of the diverter valve (e.g. 'Heating', 'DHW')."
    },
    "DHW_Active": {
        "label": "DHW Active Status",
        "unit": "0/1",
        "required": False,
        "description": "Binary sensor specifically for DHW status (1=On)."
    },

    # Advanced Diagnostics
    "Indoor_Power": {
        "label": "Indoor Unit Power (optional)",
        "unit": "W",
        "required": False,
        "description": "Indoor unit / compressor cabinet power draw, used as a proxy for immersion."
    },

    # NEW: direct heat-output sensor
    "Heat": {
        "label": "Heat Output (optional)",
        "unit": "W",
        "required": False,
        "description": "Thermal output of the heat pump in Watts (space + DHW). Use if your system already exposes a heat output sensor."
    },
    "DeltaT": {
        "label": "Flow / Return Temperature Difference (optional)",
        "unit": "degC",
        "required": False,
        "description": "Signed flow-minus-return temperature difference. When mapped, this is used instead of deriving Delta T from separate Flow and Return temperature sensors."
    },
    "Freq": {
        "label": "Compressor Frequency",
        "unit": "Hz",
        "required": False,
        "description": "Operating frequency of the compressor."
    },
    "Target_Flow": {
        "label": "Target Flow Temperature (optional)",
        "unit": "degC",
        "required": False,
        "description": "The flow temperature the controller is aiming for (weather-compensation setpoint). Used for flow-limited minutes; without it the weather curve in Setup is used."
    },
    "Defrost": {
        "label": "Defrost Status",
        "unit": "0/1",
        "required": False,
        "description": "Binary sensor indicating if defrost cycle is active."
    },
    "Immersion_Mode": {
        "label": "Immersion Heater Mode",
        "unit": "0/1",
        "required": False,
        "description": "Immersion heater on/permitted (e.g. heat_pump_immersion_heater_mode_value). With Indoor Power mapped, the element only counts while it is actually drawing power."
    },
}

# 4. Zone Sensors
ZONE_SENSORS = {
    "Zone_1": {
        "label": "Zone 1 (Call for Heat)",
        "unit": "0/1",
        "required": False,
        "description": "Thermostat or actuator signal for Zone 1."
    },
    "Zone_2": {
        "label": "Zone 2 (Call for Heat)",
        "unit": "0/1",
        "required": False,
        "description": "Thermostat or actuator signal for Zone 2."
    },
    "Zone_3": {
        "label": "Zone 3 (Call for Heat)",
        "unit": "0/1",
        "required": False,
        "description": "Thermostat or actuator signal for Zone 3."
    },
    "Zone_4": {
        "label": "Zone 4 (Call for Heat)",
        "unit": "0/1",
        "required": False,
        "description": "Thermostat or actuator signal for Zone 4."
    },
}

# 5. Room Sensors (Room_1 … Room_MAX_ROOMS)
ROOM_SENSORS = {
    f"Room_{i}": {
        "label": f"Room {i} Temperature",
        "unit": "degC",
        "required": False,
        "description": f"Temperature sensor for Room {i}.",
    }
    for i in range(1, MAX_ROOMS + 1)
}

# 6. Environmental Sensors
ENVIRONMENTAL_SENSORS = {
    "Solar_Rad": {
        "label": "Solar Radiation",
        "unit": "W/m2",
        "required": False,
        "description": "Solar irradiance sensor."
    },
    "Wind_Speed": {
        "label": "Wind Speed",
        "unit": "m/s",
        "required": False,
        "description": "External wind speed sensor."
    },
    "Outdoor_Humidity": {
        "label": "Outdoor Humidity",
        "unit": "%",
        "required": False,
        "description": "External relative humidity."
    },
}

# 6b. Secondary Weather Sensors (fallback source, e.g. an online weather service)
# Primary role -> secondary role. Where the primary sensor has no reading (or is
# not mapped), the engine fills the primary column from the secondary one
# (processing.apply_weather_fallbacks).
WEATHER_FALLBACKS = {
    "OutdoorTemp": "OutdoorTemp_Secondary",
    "Outdoor_Humidity": "Outdoor_Humidity_Secondary",
    "Wind_Speed": "Wind_Speed_Secondary",
    "Solar_Rad": "Solar_Rad_Secondary",
}

SECONDARY_WEATHER_SENSORS = {
    "OutdoorTemp_Secondary": {
        "label": "Outdoor Temperature (Secondary)",
        "unit": "degC",
        "required": False,
        "description": "Backup outdoor temperature (e.g. OpenWeatherMap). Used only where the primary Outdoor Temperature has no reading."
    },
    "Outdoor_Humidity_Secondary": {
        "label": "Outdoor Humidity (Secondary)",
        "unit": "%",
        "required": False,
        "description": "Backup outdoor humidity (e.g. OpenWeatherMap). Used only where the primary Outdoor Humidity has no reading."
    },
    "Wind_Speed_Secondary": {
        "label": "Wind Speed (Secondary)",
        "unit": "m/s",
        "required": False,
        "description": "Backup wind speed (e.g. OpenWeatherMap). Used only where the primary Wind Speed has no reading."
    },
    "Solar_Rad_Secondary": {
        "label": "Solar Radiation (Secondary)",
        "unit": "W/m2",
        "required": False,
        "description": "Backup solar irradiance. Used only where the primary Solar Radiation has no reading. Choose unit 'UV index' to use a UV index sensor (e.g. OpenWeatherMap) as a rough proxy."
    },
}

# 7. AI Context Prompts
AI_CONTEXT_PROMPTS = {
    "hp_model": {
        "label": "Heat Pump Details",
        "placeholder": "e.g., Samsung Gen 6, defrost quirks, weather compensation slope/offset, curve limits...",
        "help": "Helps the AI identify model-specific quirks and control strategy (e.g., weather compensation, defrost behaviour)."
    },
    "property_context": {
        "label": "Property Context",
        "placeholder": "e.g., 1990s detached, underfloor heating downstairs...",
        "help": "Provides context for heat loss and thermal retention."
    },
    "operational_goals": {
        "label": "Operational Goals",
        "placeholder": "e.g., Prioritize comfort over cost, minimize cycling...",
        "help": "The AI will judge performance against these specific goals."
    }
}

# ==========================================
# PART 3: UNIT CONVERSIONS & VALIDATION
# ==========================================
# Supported alternative units by sensor (UI dropdowns)
ALT_UNIT_OPTIONS = {
    # Samsung NASA integrations (ha-samsungehs power_usage, EHS-Sentinel wattmeter)
    # report power in kW; metered channels (Shelly, CT clamps) in W.
    "Power": ["kW"],
    "Indoor_Power": ["kW"],
    # Flow meters and integrations vary.
    "FlowRate": ["L/h", "m³/h"],
    # Wind speed commonly reported in m/s, km/h, or mph.
    "Wind_Speed": ["m/s", "km/h", "mph"],
    "Wind_Speed_Secondary": ["m/s", "km/h", "mph"],
    # UV index as a rough solar proxy (fallback only)
    "Solar_Rad_Secondary": ["W/m2", "UV index"],
}

# W/m2 of solar radiation per unit of UV index, by calendar month. Low sun cuts
# UV far faster than total irradiance, so the ratio is ~6x higher in winter.
# Fitted as total solar / total UV per month from one installation's Grafana exports,
# Dec 2025 - Jul 2026 (183 complete days, OpenWeatherMap UV index vs Ecowitt
# solar radiation). Measured: Jan, Mar-Jul, Dec (16-30 days each); Sep from 4
# days of HA statistics; Feb, Aug, Oct, Nov interpolated. Daily totals: 24%
# mean absolute error, -2% bias; hourly values are only indicative (~60% error)
# because the UV index is hourly and barely tracks cloud.
# Due for a refit by the date below, once a full year of solar-radiation and UV
# data exists (Aug 2026 - May 2027 adds the interpolated/thin months: Aug, Sep,
# Oct, Nov, Feb). Method: hourly means, complete days only, factor = total solar
# / total UV per month. A test fails after this date; move it on after each refit.
UV_INDEX_FACTORS_REVIEW_BY = "2027-05-31"
UV_INDEX_TO_SOLAR_W_M2_BY_MONTH = {
    1: 220.0, 2: 170.0, 3: 130.0, 4: 85.0, 5: 47.0, 6: 34.0,
    7: 36.0, 8: 45.0, 9: 65.0, 10: 110.0, 11: 150.0, 12: 195.0,
}
UV_INDEX_TO_SOLAR_W_M2 = sum(UV_INDEX_TO_SOLAR_W_M2_BY_MONTH.values()) / 12  # no timestamps

# Conversion functions to internal base units
# Base units (identity) are included so the conversion table is complete.
UNIT_CONVERSIONS = {
    "degC": lambda x: x,
    "W": lambda x: x,
    "W/m2": lambda x: x,
    "%": lambda x: x,
    "Hz": lambda x: x,
    "0/1": lambda x: x,
    "Text": lambda x: x,
    # Power
    "kW": lambda x: x * 1000.0,
    # Flow/volume helpers
    "L/min": lambda x: x,
    "L/h": lambda x: x / 60.0,
    "m³/h": lambda x: x * 1000.0 / 60.0,
    # Wind speed conversions
    "m/s": lambda x: x,
    "km/h": lambda x: x * 0.2777777778,
    "mph": lambda x: x * 0.44704,
    # Solar proxy (annual mean; convert_units applies the monthly factor when
    # the data has a time index)
    "UV index": lambda x: x * UV_INDEX_TO_SOLAR_W_M2,
}

# Validation ranges for selected sensors (optional; extend as needed)
VALIDATION_RULES = {
    "OutdoorTemp": {"type": "numeric", "min": -40, "max": 55},
    "Outdoor_Humidity": {"type": "numeric", "min": 0, "max": 100},
    "Wind_Speed": {"type": "numeric", "min": 0, "max": 60},
}

# ==========================================
# PART 4: HELPER FUNCTIONS
# ==========================================

def get_unit_options(sensor_key):
    """
    Returns a list of valid units for a given sensor key.
    Used by mapping_ui to populate the unit dropdown.
    """
    all_definitions = {}
    all_definitions.update(REQUIRED_SENSORS)
    all_definitions.update(RECOMMENDED_SENSORS)
    all_definitions.update(OPTIONAL_SENSORS)
    all_definitions.update(ZONE_SENSORS)
    all_definitions.update(ROOM_SENSORS)
    all_definitions.update(ENVIRONMENTAL_SENSORS)
    all_definitions.update(SECONDARY_WEATHER_SENSORS)

    definition = all_definitions.get(sensor_key)
    if not definition:
        return []

    default_unit = definition.get("unit")
    opts = ALT_UNIT_OPTIONS.get(sensor_key, [])
    if default_unit:
        opts = [default_unit] + opts

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in opts:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped

# ==========================================
# PART 5: ANALYSIS FEATURE FLAGS
# ==========================================

REQUIRED_COLUMNS = {
    "core": ["Power", "FlowTemp", "ReturnTemp"],
    "hydraulics": ["FlowRate"],
    "weather": ["OutdoorTemp"],
    "dhw_analysis": ["DHW_Temp"]
}

def check_feature_availability(df, user_mapping=None):
    availability = {}
    columns_present = set(df.columns)
    for feature, requirements in REQUIRED_COLUMNS.items():
        is_available = all(req in columns_present for req in requirements)
        availability[feature] = is_available
    return availability

def get_missing_columns(df, feature_name):
    if feature_name not in REQUIRED_COLUMNS:
        return []
    requirements = REQUIRED_COLUMNS[feature_name]
    return [req for req in requirements if req not in df.columns]


# Roles whose readings are discrete states or codes (held until the next change),
# never averaged or interpolated, whatever type the source stores them as.
# HA's InfluxDB integration writes e.g. a numeric defrost status (codes 0-7) to the
# numeric "value" field; without this it would be minute-averaged and interpolated.
STATE_ROLES = frozenset(
    {"ValveMode", "DHW_Mode", "DHW_Active", "DHW_Status", "Defrost", "Immersion_Mode"}
    | set(ZONE_SENSORS)
)


def state_entities(mapping) -> set:
    """Source entity ids mapped to STATE_ROLES."""
    return {
        str(v) for role, v in (mapping or {}).items()
        if role in STATE_ROLES and v not in (None, "", "None")
    }
