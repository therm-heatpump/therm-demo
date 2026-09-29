# data_normalizer.py
"""
Handles data normalization and unit conversions.
Transforms user's custom sensor names/units to the internal standard schema.
"""

import pandas as pd
import numpy as np
from schema_defs import (
    UNIT_CONVERSIONS, VALIDATION_RULES, ROOM_SENSOR_PREFIX, UV_INDEX_TO_SOLAR_W_M2_BY_MONTH,
)
from utils import _log_info, _log_warn

def apply_sensor_mapping(df, mapping_config):
    """
    Renames columns/entity IDs from user's sensor names to internal names.
    """
    # Invert mapping for pandas rename: {user_name: internal_name}
    rename_map = {v: k for k, v in mapping_config.items()}
    
    # Check if this is long format (entity_id column) or wide format (columns)
    if 'entity_id' in df.columns:
        # Long format: map entity_id values
        # We use map() but need to fillna() with original values to avoid losing unmapped rows
        # (though we usually only care about mapped ones)
        df['entity_id'] = df['entity_id'].map(rename_map).fillna(df['entity_id'])
        _log_info(f"Applied mapping to entity IDs in long format")
    else:
        # Wide format: rename columns
        existing_cols = set(df.columns)
        valid_renames = {k: v for k, v in rename_map.items() if k in existing_cols}
        df = df.rename(columns=valid_renames)
        _log_info(f"Renamed {len(valid_renames)} columns in wide format")
    
    return df

NON_NUMERIC_UNITS = {"text", "binary", "text/binary", "0/1"}


def convert_units(df, units_config):
    """
    Converts sensor values from user's units to standard internal units.
    """
    for sensor_name, user_unit in units_config.items():
        if sensor_name not in df.columns:
            continue
        
        if user_unit not in UNIT_CONVERSIONS:
            continue
        
        conversion_func = UNIT_CONVERSIONS[user_unit]

        # Skip non-numeric units (case-insensitive: profiles store "Text",
        # "Text/Binary", "0/1"). Coercing labels such as "Heating"/"DHW" to
        # numbers would blank them out.
        if str(user_unit).strip().lower() in NON_NUMERIC_UNITS:
            continue

        try:
            # Force numeric, errors='coerce' to handle potential garbage
            converted = pd.to_numeric(df[sensor_name], errors='coerce')
            # Guard: never let a "conversion" wipe out a mostly-text column
            if converted.notna().sum() < 0.5 * df[sensor_name].notna().sum():
                _log_warn(f"Skipped converting {sensor_name}: values are not numeric")
                continue
            if user_unit == "UV index" and isinstance(df.index, pd.DatetimeIndex):
                # UV-to-solar ratio is strongly seasonal: use the month's factor
                factor = df.index.month.map(UV_INDEX_TO_SOLAR_W_M2_BY_MONTH)
                df[sensor_name] = converted * np.asarray(factor, dtype=float)
            else:
                df[sensor_name] = converted.apply(conversion_func)
            _log_info(f"Converted {sensor_name} from {user_unit}")
        except Exception as e:
            _log_warn(f"Failed to convert {sensor_name}: {e}")
    
    return df

def validate_sensor_data(df, sensor_name):
    """
    Validates sensor data against defined rules.
    Clips values to valid ranges.
    """
    if sensor_name not in df.columns:
        return None, {"status": "missing"}
    
    series = df[sensor_name].copy()
    report = {"status": "ok", "warnings": [], "clipped": 0}
    
    # Get validation rules
    rules = VALIDATION_RULES.get(sensor_name, {})
    
    if not rules:
        return series, report
    
    # Type validation
    if rules.get("type") == "numeric":
        series = pd.to_numeric(series, errors='coerce')
        
        # Range validation
        if "min" in rules:
            below_min = series < rules["min"]
            if below_min.any():
                count = below_min.sum()
                series = series.clip(lower=rules["min"])
                report["warnings"].append(f"{count} values < min ({rules['min']})")
                report["clipped"] += count
        
        if "max" in rules:
            above_max = series > rules["max"]
            if above_max.any():
                count = above_max.sum()
                series = series.clip(upper=rules["max"])
                report["warnings"].append(f"{count} values > max ({rules['max']})")
                report["clipped"] += count
    
    elif rules.get("type") == "binary":
        # Normalize binary values (on/off, true/false -> 1/0)
        series = series.astype(str).str.lower()
        series = series.replace({"on": 1, "off": 0, "true": 1, "false": 0, "1.0": 1, "0.0": 0})
        series = pd.to_numeric(series, errors='coerce')
    
    return series, report

# Heat is derived from hydraulics in processing._ensure_heat_and_cop, using the
# profile's primary-circuit fluid (config.resolve_fluid_specific_heat_kj).

def enhance_zone_handling(df, user_config):
    """
    Ensures zone columns are standardized (numeric 0/1).
    """
    zone_sensors = [k for k in user_config["mapping"].keys() if k.startswith("Zone_")]
    
    for zone in zone_sensors:
        if zone in df.columns:
            df[zone] = pd.to_numeric(df[zone], errors='coerce').fillna(0).astype(int)
    
    return df

def normalize_dataframe(df, user_config):
    """
    Main entry point for data normalization.
    """
    # 1. Apply sensor mapping (already done in loader usually, but safety check)
    # Note: Loader calls apply_sensor_mapping BEFORE resampling.
    # This function is called AFTER resampling to apply Units and Validation.
    
    # 2. Convert units
    df = convert_units(df, user_config["units"])
    
    # 3. Validate
    validation_reports = {}
    for sensor_name in user_config["mapping"].keys():
        cleaned, report = validate_sensor_data(df, sensor_name)
        if cleaned is not None:
            df[sensor_name] = cleaned
            validation_reports[sensor_name] = report

    return df, validation_reports