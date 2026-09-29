# config_manager.py
"""
Manages configuration profiles and export logic.
"""
import json
from datetime import datetime

# Keys we explicitly carry over from a loaded profile when re-exporting
PRESERVED_KEYS = {
    "therm_version",
    "profile_name",
    "mapping",
    "units",
    "ai_context",
    "config_history",
    "rooms_per_zone",
    "tariff_structure",
    "thresholds",
    "physics_thresholds",
    "currency",
    "timezone",
    "weather_curve",
}
# Saved only when set (older profiles don't have them; readers infer a default).
OPTIONAL_KEYS = (
    "electricity_source",  # COP electricity-reading provenance (optional for legacy profiles)
    "heating_zones",    # "none" | "signals" (setup_sections)
    "inactive_zones",   # zone choices kept while heating_zones is "none"
    "setup_level",      # "quick" | "guided" | "full": the wizard route last used
    "currency_code",    # ISO 4217 code; "currency" stays the display symbol
)
from config import resolve_heat_source_mode
from schema_defs import REQUIRED_SENSORS, check_feature_availability

def validate_config(config):
    """
    Single validation used before processing (Setup screen) and by tests.

    - Power is required (the engine can't run without it).
    - Flow and Return temperature are required unless a native Heat output
      sensor is mapped (they are only needed to calculate heat).
    - With heat_source "calculated", Flow Rate and a Delta T source are
      required. Delta T may be mapped directly or derived from Flow and Return
      temperature. The engine never falls back to the Heat sensor.
    - Each source entity may fill only one role: the loaders invert the mapping,
      so a duplicate would silently drop one of the roles.

    Returns (ok, [messages]).
    """
    mapping = (config or {}).get("mapping") or {}
    mapping = {k: v for k, v in mapping.items() if v not in (None, "", "None")}
    if not mapping:
        return False, [
            "No sensors are mapped yet. Press 🔎 Detect my heat pump, or choose at least "
            f"{REQUIRED_SENSORS.get('Power', {}).get('label', 'Power')}, Flow Temperature and "
            "Return Temperature under Heat pump sensors."
        ]

    errors = []
    label = lambda role: REQUIRED_SENSORS.get(role, {}).get("label", role)
    if "Power" not in mapping:
        errors.append(f"{label('Power')} is required.")
    phys = (config or {}).get("physics_thresholds") or {}
    if resolve_heat_source_mode(phys.get("heat_source")) == "calculated":
        if "FlowRate" not in mapping:
            errors.append(
                f"{label('FlowRate')} is required when heat is calculated in therm "
                "(Setup → Heat & hot water → Heat output source)."
            )
        if "DeltaT" not in mapping:
            for role in ("FlowTemp", "ReturnTemp"):
                if role not in mapping:
                    errors.append(
                        f"{label(role)} is required when heat is calculated in therm "
                        "unless a Delta T sensor is mapped."
                    )
    elif "Heat" not in mapping:
        for role in ("FlowTemp", "ReturnTemp"):
            if role not in mapping:
                errors.append(f"{label(role)} is required (or map a Heat output sensor).")

    roles_by_entity = {}
    for role, entity in mapping.items():
        roles_by_entity.setdefault(entity, []).append(role)
    for entity, roles in roles_by_entity.items():
        if len(roles) > 1:
            errors.append(
                f"'{entity}' is mapped to more than one role ({', '.join(roles)}). "
                "Each sensor can fill only one role."
            )
    return len(errors) == 0, errors

def export_config_for_sharing(config):
    """
    Creates a clean version of the config for download.
    """
    # Preserve editable fields from the in-memory config (including updated profile_name)
    export_data = {}
    for k in PRESERVED_KEYS:
        # Default list-like structures to an empty list, others to an empty dict.
        # This prevents custom tariffs (lists) from being saved as empty dicts on error.
        if k in ["config_history", "tariff_structure"]:
            export_data[k] = config.get(k, [])
        else:
            export_data[k] = config.get(k, {})

    # Fill defaults where absent
    export_data["therm_version"] = export_data.get("therm_version") or "2.0"
    export_data["profile_name"] = export_data.get("profile_name") or "My Profile"
    export_data["mapping"] = export_data.get("mapping") or {}
    export_data["units"] = export_data.get("units") or {}
    export_data["ai_context"] = export_data.get("ai_context") or {}
    export_data["config_history"] = export_data.get("config_history") or []
    export_data["rooms_per_zone"] = export_data.get("rooms_per_zone") or {}
    export_data["timezone"] = export_data.get("timezone") or "Europe/Dublin"

    for k in OPTIONAL_KEYS:
        if config.get(k):
            export_data[k] = config[k]

    export_data["exported_at"] = datetime.now().isoformat()
    return export_data
