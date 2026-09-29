# mapping_ui.py
import csv
import json
import math
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from schema_defs import (
    REQUIRED_SENSORS, RECOMMENDED_SENSORS, OPTIONAL_SENSORS, 
    ZONE_SENSORS, ENVIRONMENTAL_SENSORS, AI_CONTEXT_PROMPTS, ROOM_SENSOR_PREFIX,
    WEATHER_FALLBACKS, SECONDARY_WEATHER_SENSORS
)
import config_manager
import config
import profile_store
import onboarding
import presets
import setup_sections
from setup_sections import DEFAULT_PROFILE_NAME


def _log(msg: str) -> None:
    """Best-effort console logger for profiling (disabled by default)."""
    return

def _friendly_entity_label(entity: str) -> str:
    """Trim common HA prefixes for display only."""
    if not isinstance(entity, str):
        return str(entity)
    prefixes = (
        "sensor.",
        "binary_sensor.",
        "switch.",
        "climate.",
        "input_boolean.",
        "input_number.",
    )
    for p in prefixes:
        if entity.startswith(p):
            return entity[len(p):]
    return entity


# -------------------------------------------------------------------
# FAST ENTITY SCAN (streaming) TO AVOID LOADING HUGE HA FILES
# -------------------------------------------------------------------
def _quick_entity_scan(file_obj, max_rows: int = 50_000, max_entities: int = 200):
    """
    Extract unique entity_id values from a long-form HA CSV without loading it
    fully. Stops early once a reasonable number of unique entities have been
    found, and caps total rows scanned to keep the UI responsive even for
    large history exports. We only need the first occurrence of each entity_id.
    """
    # Remember position so we can restore
    try:
        start_pos = file_obj.tell()
    except Exception:
        start_pos = None

    entities: set[str] = set()
    rows_seen = 0

    try:
        reader = csv.reader(file_obj)
        header = next(reader, None)
        if not header:
            return []

        # Locate entity_id column (case-insensitive)
        try:
            eid_idx = [h.lower() for h in header].index("entity_id")
        except ValueError:
            return []

        for row in reader:
            rows_seen += 1
            if rows_seen > max_rows:
                break
            if eid_idx < len(row):
                val = row[eid_idx].strip()
                if val and val not in entities:
                    entities.add(val)
            if len(entities) >= max_entities:
                break

    except Exception:
        return []
    finally:
        if start_pos is not None:
            try:
                file_obj.seek(start_pos)
            except Exception:
                pass

    return sorted(entities)


def get_all_unique_entities(uploaded_files):
    """
    Entity discovery with caching and chunked parsing for long-form files.
    """
    entity_cache = st.session_state.get("entity_cache", {})
    debug_scans = []
    found_entities = set()
    for file_obj in uploaded_files:
        file_obj.seek(0)
        sig = (getattr(file_obj, "name", None), getattr(file_obj, "size", None))

        # Always refresh when invoked (cache is still updated for future calls)
        cached = entity_cache.get(sig)

        t0 = time.time()
        try:
            df_head = pd.read_csv(file_obj, nrows=200)
            cols_lower = {c.lower(): c for c in df_head.columns}

            # --- LONG FORM: entity_id or similar ---
            entity_col = None
            for cand in ["entity_id", "entity id", "entity"]:
                if cand in cols_lower:
                    entity_col = cols_lower[cand]
                    break

            if entity_col:
                file_obj.seek(0)
                ents_set: set[str] = set()
                for chunk in pd.read_csv(file_obj, usecols=[entity_col], chunksize=50_000):
                    vals = (
                        chunk[entity_col]
                        .astype(str)
                        .dropna()
                        .str.strip()
                        .tolist()
                    )
                    ents_set.update([v for v in vals if v])
                    if len(ents_set) >= 5000:
                        break
                ents = sorted(ents_set)

            # --- LONG FORM: Grafana series ---
            else:
                if "series" in cols_lower:
                    series_col = cols_lower["series"]
                    file_obj.seek(0)
                    ents = set()
                    for chunk in pd.read_csv(file_obj, usecols=[series_col], chunksize=50_000):
                        vals = (
                            chunk[series_col]
                            .astype(str)
                            .dropna()
                            .str.strip()
                            .tolist()
                        )
                        ents.update([v for v in vals if v])
                        if len(ents) >= 1000:
                            break
                    ents = sorted(ents)
                else:
                    # --- WIDE FORM ---
                    ents = [
                        c
                        for c in df_head.columns
                        if c.lower()
                        not in [
                            "time",
                            "date",
                            "timestamp",
                            "datetime",
                            "last_changed",
                            "last_updated",
                            "value",
                            "state",
                        ]
                    ]

            found_entities.update(ents)
            # Update cache with the newly discovered entities (or fall back to cached)
            if found_entities:
                entity_cache[sig] = sorted(set(ents)) if 'ents' in locals() else []
                debug_scans.append({
                    "file": sig[0],
                    "cached": False,
                    "entities": len(entity_cache[sig]),
                    "secs": time.time() - t0,
                })
            elif cached:
                found_entities.update(cached)
                debug_scans.append({
                    "file": sig[0],
                    "cached": True,
                    "entities": len(cached),
                    "secs": 0.0,
                })
        except Exception:
            if cached:
                found_entities.update(cached)
                debug_scans.append({
                    "file": sig[0],
                    "cached": True,
                    "entities": len(cached),
                    "secs": 0.0,
                })
        file_obj.seek(0)
    st.session_state["entity_cache"] = entity_cache
    st.session_state["entity_scan_debug"] = debug_scans
    return sorted(list(found_entities))

_UNIT_ALIASES = {"w": "W", "kw": "kW", "l/min": "L/min", "l/h": "L/h", "m³/h": "m³/h", "m3/h": "m³/h",
                 "m/s": "m/s", "km/h": "km/h", "mph": "mph", "w/m²": "W/m2", "w/m2": "W/m2"}

_TIER_LABELS = {1: "integration", 2: "integration", 3: "sensor name", 4: "name pattern — please check"}
_OFFLINE_STATES = {"unavailable", "unknown"}


def _role_label(role: str) -> str:
    """The label Setup shows for a therm role (e.g. FlowTemp → Flow Temperature)."""
    from schema_defs import ROOM_SENSORS

    for catalogue in (REQUIRED_SENSORS, RECOMMENDED_SENSORS, OPTIONAL_SENSORS, ZONE_SENSORS,
                      ENVIRONMENTAL_SENSORS, SECONDARY_WEATHER_SENSORS, ROOM_SENSORS):
        if role in catalogue and catalogue[role].get("label"):
            return catalogue[role]["label"]
    if role.startswith(ROOM_SENSOR_PREFIX):
        return f"Room Sensor {role[len(ROOM_SENSOR_PREFIX):]}"
    return role.replace("_", " ")


_TARIFF_COLUMNS = ["valid_from", "name", "start", "end", "rate"]


def _tariff_rows_frame(tariff_structure) -> pd.DataFrame:
    """Price periods as editable rows (one per band). Flat and day/night profiles are
    shown as their bands; an empty profile starts as one all-day price."""
    import datetime as _dt

    import processing

    rows = []
    for profile in processing._parse_tariff_profiles(tariff_structure):
        vf = profile["valid_from"]
        if vf.year < 2000:  # the old "applies to everything" placeholder
            vf = _dt.date(2020, 1, 1)
        for r in profile["rules"]:
            end = r["end"].strftime("%H:%M")
            if r["end"] == _dt.time(23, 59, 59):
                end = "24:00"
            rows.append({"valid_from": vf, "name": r.get("name") or "", "start": r["start"].strftime("%H:%M"),
                         "end": end, "rate": float(r["rate"])})
    if not rows:
        rows = [{"valid_from": _dt.date(2020, 1, 1), "name": "All day", "start": "00:00", "end": "24:00",
                 "rate": 0.33}]
    return pd.DataFrame(rows, columns=_TARIFF_COLUMNS)


def _hhmm_minutes(value) -> int | None:
    """'07:30' → 450; '24:00' → 1440; invalid → None."""
    s = str(value or "").strip()
    if s in ("24", "24:00", "24:00:00"):
        return 1440
    try:
        t = datetime.strptime(s[:5], "%H:%M")
    except ValueError:
        return None
    return t.hour * 60 + t.minute


def _tariff_structure_from_rows(rows: pd.DataFrame) -> tuple[list, list]:
    """(tariff_structure, errors): one profile per 'Valid from' date, each checked to
    cover all 24 hours exactly once."""
    errors: list[str] = []
    rows = rows.dropna(subset=["valid_from"]) if "valid_from" in rows else rows.iloc[0:0]
    if rows.empty:
        return [], ["Enter at least one electricity price (one row from 00:00 to 24:00 for a single price)."]
    structure = []
    for vf, group in rows.groupby(pd.to_datetime(rows["valid_from"]).dt.date, sort=True):
        coverage = [0] * 1440
        rules = []
        for _, r in group.iterrows():
            start, end = _hhmm_minutes(r.get("start")), _hhmm_minutes(r.get("end"))
            try:
                rate = float(r.get("rate"))
            except (TypeError, ValueError):
                rate = float("nan")
            if start is None or end is None or rate != rate:
                errors.append(f"Prices from {vf:%d/%m/%Y}: every row needs a start, an end (HH:MM) and a price.")
                continue
            if start >= 1440:
                errors.append(f"Prices from {vf:%d/%m/%Y}: 24:00 is allowed only as an end time.")
                continue
            if start == end:
                errors.append(f"Prices from {vf:%d/%m/%Y}: start and end cannot be the same; use 00:00 to 24:00 for an all-day price.")
                continue
            if rate < 0:
                errors.append(f"Prices from {vf:%d/%m/%Y}: prices cannot be negative.")
                continue
            span = range(start, end) if end > start else list(range(start, 1440)) + list(range(0, end))
            for m in span:
                coverage[m % 1440] += 1
            rules.append({"name": str(r.get("name") or ""), "start": f"{start // 60:02d}:{start % 60:02d}",
                          "end": "23:59:59" if end == 1440 else f"{end // 60:02d}:{end % 60:02d}",
                          "rate": rate})
        if rules and 0 in coverage:
            gap = coverage.index(0)
            errors.append(f"Prices from {vf:%d/%m/%Y} don't cover the whole day (nothing from "
                          f"{gap // 60:02d}:{gap % 60:02d}).")
        if any(count > 1 for count in coverage):
            overlap = next(i for i, count in enumerate(coverage) if count > 1)
            errors.append(f"Prices from {vf:%d/%m/%Y} overlap at "
                          f"{overlap // 60:02d}:{overlap % 60:02d}; each minute must have one price.")
        structure.append({"valid_from": vf.isoformat(), "name": f"Prices from {vf:%d %b %Y}", "rules": rules})
    return structure, errors


def _preset_names() -> dict:
    """preset id → readable name (presets/*.json 'name'), for Detect's result table."""
    import presets

    try:
        return {p.get("id"): (p.get("name") or p.get("id")) for p in presets.load_presets()}
    except Exception:
        return {}


def _unit_option(role: str, source_unit, entity_id: str = ""):
    """The Setup unit choice matching the unit a source reports, if the role offers it."""
    from schema_defs import get_unit_options

    options = get_unit_options(role)
    if len(options) < 2:
        return None
    if role == "Solar_Rad_Secondary" and "uv" in entity_id.lower():
        return "UV index" if "UV index" in options else None
    unit = _UNIT_ALIASES.get(str(source_unit or "").strip().lower())
    return unit if unit in options else None


MODBUS_PRESET = "samsung_modbus_mim_b19n"
MODBUS_POWER_WARNING = (
    "**A separate electricity meter is required.** Samsung's Modbus interface (MIM-B19N) doesn't report "
    "the heat pump's electricity use, so therm can't calculate COP or costs from Modbus alone. "
    "Add a meter on the heat pump's supply (e.g. a Shelly EM or a CT clamp) to Home Assistant and choose "
    "it as **Heat Pump Power** under Heat pump sensors. The indoor unit's power is not the heat pump's."
)


def _render_auto_detect(available_entities, draft: dict, collapsed: bool = False):
    """'Detect my heat pump': fill empty roles from presets.auto_map (see presets/*.json)."""
    import presets

    meta = {m.get("entity_id"): m for m in (st.session_state.get("entity_metadata") or [])}
    # Sensors Home Assistant reports as unavailable/unknown right now (device offline,
    # flat battery) would give an empty chart: don't suggest them.
    offline = {e for e, m in meta.items() if str(m.get("state", "")).strip().lower() in _OFFLINE_STATES}
    entities = [
        presets.EntityInfo(
            entity_id=e,
            platform=(meta.get(e) or {}).get("platform"),
            device_class=(meta.get(e) or {}).get("device_class"),
            unit=(meta.get(e) or {}).get("unit"),
            domain=(meta.get(e) or {}).get("domain"),
        )
        for e in available_entities
        if isinstance(e, str) and e and e != "None" and e not in offline
    ]
    current = {k: v for k, v in draft["mapping"].items() if v not in (None, "", "None")}

    def _detect(replace: bool) -> None:
        existing = {} if replace else current
        result = presets.auto_map(entities, existing=existing)
        if replace:
            # "Replace" starts from a clean slate: settings Detect doesn't fill are
            # cleared too, instead of keeping an earlier (possibly offline) choice.
            for role in list(draft["mapping"]):
                setup_sections.set_role(draft, role, None)
            # The former Power selection is gone too. A newly detected Power
            # role below may replace this with its preset provenance.
            draft["electricity_source"] = "unknown"
        by_id = {e.entity_id: e for e in entities}
        names = _preset_names()
        filled, zone_candidates = [], []
        for role, eid in result.mapping.items():
            if role in result.sources:  # newly suggested (not an existing choice)
                unit = _unit_option(role, result.units.get(role), eid)
                if role in setup_sections.ZONE_ROLES and not setup_sections.zones_on(draft):
                    # Zones are optional: offer what Detect found under Rooms & zones instead of
                    # switching zones on silently.
                    zone_candidates.append({"role": role, "entity": eid, "unit": unit})
                    continue
                setup_sections.set_role(draft, role, eid, unit)
                if role == "Power":
                    # Keep a chosen/loaded source intact when Detect only fills
                    # other roles. A replacement's newly detected Power source,
                    # including a generic match, is authoritative for this draft.
                    draft["electricity_source"] = result.role_provenance.get(role, "unknown")
                preset_id, tier = result.sources[role]
                filled.append({
                    "Setting": _role_label(role),
                    "Sensor": _friendly_entity_label(eid),
                    "Unit": unit or (by_id[eid].unit if eid in by_id else "") or "",
                    "Found via": f"{names.get(preset_id, preset_id)} · {_TIER_LABELS.get(tier, tier)}",
                })
        # Which offline sensors would have been suggested? Name those (not every
        # unavailable entity in Home Assistant, which can be hundreds).
        skipped = []
        if offline:
            everything = entities + [presets.EntityInfo(entity_id=e, **{
                k: (meta.get(e) or {}).get(k) for k in ("platform", "device_class", "unit", "domain")})
                for e in sorted(offline & set(available_entities))]
            with_offline = presets.auto_map(everything, existing={} if replace else current)
            skipped = sorted(e for e in with_offline.mapping.values() if e in offline)
        st.session_state["autodetect_result"] = {
            "rows": filled, "notes": result.notes,
            "presets": [names.get(p, p) for p in result.presets_used],
            "offline": skipped,
            "zone_candidates": zone_candidates,
            # Samsung's Modbus interface (MIM-B19N) has no register for the heat pump's electricity use.
            "modbus_no_power": MODBUS_PRESET in result.presets_used and not result.mapping.get("Power"),
        }
        draft["_modbus_no_power"] = st.session_state["autodetect_result"]["modbus_no_power"]

    import os
    if os.environ.get("THERM_DEMO"):
        with st.expander("🔎 Detect my heat pump", expanded=False):
            st.info(
                "In this demo, sensor mappings reflect the bundled sample heat pump (Samsung Gen 6). "
                "In a live Home Assistant installation, this button automatically scans your real entity registry."
            )
        return

    # First visit with nothing mapped: run Detect straight away, once.
    if not current and entities and not st.session_state.get("autodetect_auto_done"):
        st.session_state["autodetect_auto_done"] = True
        _detect(replace=False)
        st.rerun()

    last = st.session_state.get("autodetect_result")
    expanded = not collapsed and (bool(last) or (not current and bool(entities)))
    with st.expander("🔎 Detect my heat pump", expanded=expanded):
        st.caption(
            "Suggests a sensor for each setting from known Samsung integrations (Modbus MIM-B19N, "
            "ESPHome, ha-samsungehs, EHS-Sentinel), metered power, weather, rooms and zones. "
            "Check the suggestions in the sections below before saving."
        )
        replace = st.checkbox("Replace my current selections", value=False, key="autodetect_replace")
        if st.button("Detect sensors", key="autodetect_btn", disabled=not entities):
            _detect(replace)
            st.rerun()

        if last:
            if last["rows"]:
                st.success(f"Found {len(last['rows'])} sensors. Check them in the sections below, then save.")
                st.dataframe(pd.DataFrame(last["rows"]), hide_index=True, width="stretch")
            else:
                st.info("No further settings could be matched automatically.")
            if last.get("modbus_no_power"):
                st.warning(MODBUS_POWER_WARNING, icon="⚡")
            if last.get("zone_candidates"):
                st.caption(f"Also found {len(last['zone_candidates'])} possible heating zone signal(s): see "
                           "**Rooms & zones** to use them.")
            if last.get("offline"):
                st.warning("Not suggested because Home Assistant reports them as unavailable (device offline or "
                           "battery flat?): " + ", ".join(f"`{_friendly_entity_label(e)}`" for e in last["offline"]))
            if last["notes"]:
                # (a popover: Streamlit does not allow an expander inside an expander)
                with st.popover("Details for experts"):
                    for note in last["notes"]:
                        st.caption(f"• {note}")


def render_configuration_interface(uploaded_files):
    """System Setup: header (profiles, name, Detect, Save) and the Setup page, all drawn from one draft
    (setup_sections). Returns the profile to save when Save is pressed and it is valid, else None."""
    import source_ui

    t_render_start = time.time()
    # Two containers: the top block stays sticky; the rest scrolls normally.
    top_section = st.container()
    body_section = st.container()

    existing_cfg = st.session_state.get("system_config")
    draft = setup_sections.get_draft(existing_cfg)
    setup_sections.apply_home_assistant_defaults(draft, source_ui.home_assistant_defaults())
    available_entities = _refresh_entities(uploaded_files)

    with top_section:
        st.markdown("<div class='setup-sticky'>", unsafe_allow_html=True)
        st.markdown("## System Setup")
        saved_names = profile_store.list_profiles()
        addon = source_ui.running_in_addon()
        direct_source = st.session_state.get("data_source_choice", source_ui.SOURCE_CSV) != source_ui.SOURCE_CSV
        first_run = not saved_names and not isinstance(existing_cfg, dict)
        # The wizard runs for the direct sources (Home Assistant, InfluxDB); with CSV uploads the entity
        # list only exists once files are uploaded, so the standalone app keeps the Setup page.
        wizard = onboarding.active(first_run and direct_source)
        if first_run and not wizard:
            save_label = "Save and continue" if addon else "2. Save and continue to analysis"
            st.info(
                "**Welcome to therm.** It has looked for your heat pump's sensors (🔎 Detect my heat pump, "
                f"below). Check the **Heat pump sensors**, give this set-up a name, then press **{save_label}**. "
                "Everything else is optional and can be changed later under ⚙️ Setup."
            )

        col_load, col_name = st.columns([1, 2])
        loaded_name = None
        with col_load:
            picked_saved = None
            if saved_names:
                # Preselect the active profile.
                active = (existing_cfg or {}).get("profile_name") if isinstance(existing_cfg, dict) else None
                active = active or profile_store.get_state().get("active_profile")
                choice = st.selectbox(
                    "Saved profiles", ["—"] + saved_names,
                    index=(saved_names.index(active) + 1)
                    if active in saved_names and not st.session_state.get("setup_new_profile") else 0,
                    key="saved_profile_pick",
                    help="Profiles saved in therm (the add-on's config folder, or ./profiles locally).",
                )
                picked_saved = None if choice == "—" else choice
            # Add-on: no file handling in the workflow. Profiles are saved inside therm when you continue,
            # and chosen from the list above (files can still be copied over the add-on's config share).
            uploaded_config = None if addon else st.file_uploader(" Load Profile", type="json", key="cfg_up")
            if uploaded_config or picked_saved:
                try:
                    signature = ((uploaded_config.name, getattr(uploaded_config, "size", None))
                                 if uploaded_config else ("saved", picked_saved))
                    previous = st.session_state.get("loaded_profile_signature")
                    if signature != previous:
                        loaded = json.load(uploaded_config) if uploaded_config else profile_store.load_profile(
                            picked_saved)
                        st.session_state["loaded_profile_signature"] = signature
                        st.session_state.pop("setup_new_profile", None)
                        # Opening Setup preselects the active profile: that is not a switch.
                        active_name = (existing_cfg or {}).get("profile_name") if isinstance(existing_cfg, dict) else None
                        if previous is not None or loaded.get("profile_name") != active_name:
                            draft = setup_sections.replace_draft(loaded)
                            st.session_state["profile_name_input"] = draft["profile_name"]
                            if previous is not None:
                                loaded_name = loaded.get("profile_name")
                except Exception:
                    st.error("Failed to load profile JSON.")

        # A profile made with one source names sensors its way (`sensor.x` in Home Assistant, `x` in InfluxDB).
        # Adapt the choices to the current source, so the dropdowns don't list both spellings.
        if available_entities:
            for role, value in source_ui.resolve_mapping(dict(draft["mapping"]), available_entities).items():
                if value != draft["mapping"].get(role):
                    setup_sections.set_role(draft, role, value)

        _render_auto_detect(available_entities, draft, collapsed=wizard)

        # Manual refresh button for entity discovery (uploads only)
        if uploaded_files:
            with st.expander("Entity Discovery", expanded=False):
                files_key = sorted((getattr(f, "name", ""), getattr(f, "size", 0)) for f in uploaded_files)
                if st.button("Refresh entities from uploaded files", type="secondary"):
                    t_scan = time.time()
                    st.session_state["available_sensors"] = get_all_unique_entities(uploaded_files)
                    st.session_state["available_sensors_files_key"] = files_key
                    st.success(f"Entities refreshed in {time.time()-t_scan:.3f}s")
                cached_entities = st.session_state.get("available_sensors", [])
                st.caption(f"{len(cached_entities)} entities cached." if cached_entities
                           else "Entities will be loaded from profile mapping unless refreshed.")

        ctx = setup_sections.Ctx(*_entity_options(draft, available_entities))

        # --- Profile name ---
        setup_sections._seed("profile_name_input", draft["profile_name"])
        with col_name:
            draft["profile_name"] = st.text_input(
                "Profile name", key="profile_name_input", placeholder=DEFAULT_PROFILE_NAME,
                help="A name for this set-up, e.g. your house or heat pump. Profiles are saved inside therm.",
            )
            if loaded_name:
                st.success(f"Loaded {loaded_name}")

        # Top action bar (sticky with CSS)
        action_bar_top = st.container()
        st.markdown("</div>", unsafe_allow_html=True)

    with body_section:
        if wizard:
            save_pressed = onboarding.render(draft, ctx)
        else:
            if direct_source:
                c_again, c_new, _ = st.columns([1, 1, 2])
                if c_again.button("Run guided set-up again", key="onb_rerun", type="tertiary"):
                    onboarding.restart()
                    st.rerun()
                if c_new.button("Start a new profile", key="onb_new_profile", type="tertiary",
                                help="Set up another heat pump (or start over) from scratch. Saved profiles are "
                                     "kept; switch between them under Saved profiles."):
                    _start_new_profile()
                    st.rerun()
            setup_sections.render_page(draft, ctx)

    config_object, problems = setup_sections.build_config(draft)
    if wizard:
        # The wizard saves from its Review screen; the header has no Save while it runs.
        return _finish(config_object, problems, t_render_start) if save_pressed else None

    with action_bar_top:
        # Direct sources (HA / InfluxDB) have no uploads: the profile is saved inside therm and the period
        # is chosen on the next screen.
        if addon:
            # One step, no file: "Save and continue" stores the profile inside therm and makes it active.
            c_btn2 = st.container()
        else:
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                export_data = config_manager.export_config_for_sharing(config_object)
                export_data["rooms_per_zone"] = config_object["rooms_per_zone"]
                export_data["config_history"] = config_object["config_history"]
                st.download_button(
                    label=" 1. Download profile (JSON)" if direct_source else " 1. Save Configuration",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"therm_profile_{config_object['profile_name'].replace(' ', '_')}.json",
                    mime="application/json",
                    type="secondary",
                )
        with c_btn2:
            if addon:
                label = "Save and continue"
            else:
                label = " 2. Save and continue to analysis" if direct_source else " 2. Process Uploaded Data"
            if st.button(label, type="primary"):
                return _finish(config_object, problems, t_render_start)
    return None


def _start_new_profile() -> None:
    """A fresh draft (Home Assistant's defaults, Detect run again) and the wizard's entry screen. The active
    profile is unchanged until the new one is saved."""
    setup_sections.replace_draft({})
    for key in ("autodetect_result", "autodetect_auto_done", "profile_name_input"):
        st.session_state.pop(key, None)
    st.session_state["loaded_profile_signature"] = ("new", time.time())
    st.session_state["setup_new_profile"] = True   # the Saved profiles picker shows "—" meanwhile
    st.session_state.pop("saved_profile_pick", None)
    onboarding.restart()


def _finish(config_object: dict, problems: list, t_start: float):
    """Validate and hand the profile to app.py (which saves it and makes it active)."""
    ok, validation = config_manager.validate_config(config_object)
    problems = list(problems) + ([] if ok else list(validation))
    if problems:
        for msg in problems:
            st.error(msg)
        return None
    # Flag the main app to scroll to top on the next render; Setup opens as the page next time.
    st.session_state["scroll_to_top"] = True
    for key in (onboarding.MODE, onboarding.STEP, "setup_new_profile"):
        st.session_state.pop(key, None)
    _log(f"render_configuration_interface done secs={time.time()-t_start:.3f} result=config_object")
    return config_object


def _entity_options(draft: dict, available_entities: list):
    """(options, format function) for every sensor dropdown: 'None' plus the source's entities and any
    entity the draft already uses, labelled with Home Assistant's own name where known."""
    combined = sorted(set([v for v in draft["mapping"].values() if v] + list(available_entities)),
                      key=lambda x: str(x).lower())
    names_by_id = {m.get("entity_id"): m.get("friendly_name")
                   for m in (st.session_state.get("entity_metadata") or []) if m.get("friendly_name")}
    lookup, counts = {}, {}
    for opt in combined:
        short = _friendly_entity_label(opt)
        friendly = names_by_id.get(opt)
        label = f"{friendly} ({short})" if friendly and friendly.lower() != short.lower() else short
        counts[label] = counts.get(label, 0) + 1
        lookup[opt] = f"{label} ({opt})" if counts[label] > 1 else label
    return (["None"] + combined, (lambda opt: lookup.get(opt, _friendly_entity_label(opt))), "full",
            _temperature_options(combined))


# Single letters are case-sensitive (presets.unit_matches): "C" is Celsius, "c" is cents.
_TEMPERATURE_UNITS = ("°C", "°F", "C", "F", "degC", "degF", "℃", "℉")


def _temperature_options(entities: list) -> list | None:
    """'None' plus the entities that report °C or °F (for room temperatures), or None (offer everything)
    when the source gives no units, as with CSV uploads."""
    meta = {m.get("entity_id"): m for m in (st.session_state.get("entity_metadata") or [])}
    if not any((m or {}).get("unit") for m in meta.values()):
        return None
    return ["None"] + [e for e in entities
                       if presets.unit_matches((meta.get(e) or {}).get("unit") or "", _TEMPERATURE_UNITS)]


def _refresh_entities(uploaded_files) -> list:
    """The entity list for the dropdowns: cached, or scanned from newly uploaded files."""
    available_entities = st.session_state.get("available_sensors", []) or []
    if not uploaded_files:
        return available_entities
    files_key = sorted((getattr(f, "name", ""), getattr(f, "size", 0)) for f in uploaded_files)
    if available_entities and st.session_state.get("available_sensors_files_key") == files_key:
        return available_entities
    try:
        refreshed = [e for e in get_all_unique_entities(uploaded_files) if isinstance(e, str) and e.strip()]
        if not refreshed:
            refreshed = _fallback_entity_scan(uploaded_files)
        if refreshed:
            st.session_state["available_sensors"] = refreshed
            st.session_state["available_sensors_files_key"] = files_key
            return refreshed
    except Exception:
        pass  # best effort: keep the existing cache if the scan fails
    return st.session_state.get("available_sensors", []) or []


def _fallback_entity_scan(uploaded_files) -> list:
    """Naive column/entity scrape of the first rows, so the dropdowns aren't empty."""
    found: set[str] = set()
    for f in uploaded_files:
        try:
            f.seek(0)
            head = pd.read_csv(f, nrows=200)
            cols = list(head.columns)
            lower = [c.lower() for c in cols]
            for name in ("entity_id", "series"):
                if name in lower:
                    col = cols[lower.index(name)]
                    found.update(v for v in head[col].astype(str).dropna().str.strip().tolist() if v)
                    break
            else:
                ignore = {"time", "timestamp", "date", "datetime", "last_changed", "last_updated"}
                found.update(c for c in cols if c.lower() not in ignore)
        except Exception:
            continue
        finally:
            try:
                f.seek(0)
            except Exception:
                pass
    return sorted(found)


def render_config_download(config: dict) -> None:
    """
    Download helper for an already-built config object.

    Uses the same export logic as the main configuration interface,
    but works with an existing config passed in from the caller.
    """
    # Derive profile name from the config, with a sensible default
    profile_name = config.get("profile_name", "My Heat Pump")

    # Start from the canonical sharing export
    export_data = config_manager.export_config_for_sharing(config)

    # Ensure key fields are preserved
    export_data["profile_name"] = profile_name
    export_data["rooms_per_zone"] = config.get("rooms_per_zone", {})

    st.download_button(
        label=" 1. Save Configuration",
        data=json.dumps(export_data, indent=2),
        file_name=f"therm_profile_{profile_name.replace(' ', '_')}.json",
        mime="application/json",
        type="secondary",
        key=f"save_btn_{profile_name.replace(' ', '_')}",
    )
