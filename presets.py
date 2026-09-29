"""
presets.py

Heat-pump presets and automatic role mapping for THERM.
Maps entities discovered from Home Assistant (or bare entity IDs from InfluxDB)
to THERM calculation roles.

Pure Python: no Streamlit, no Home Assistant imports, no network.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from schema_defs import (
    ENVIRONMENTAL_SENSORS,
    OPTIONAL_SENSORS,
    RECOMMENDED_SENSORS,
    REQUIRED_SENSORS,
    ROOM_SENSOR_PREFIX,
    ROOM_SENSORS,
    SECONDARY_WEATHER_SENSORS,
    ZONE_SENSOR_PREFIX,
    ZONE_SENSORS,
)

VALID_THERM_ROLES = (
    set(REQUIRED_SENSORS)
    | set(RECOMMENDED_SENSORS)
    | set(OPTIONAL_SENSORS)
    | set(ZONE_SENSORS)
    | set(ROOM_SENSORS)
    | set(ENVIRONMENTAL_SENSORS)
    | set(SECONDARY_WEATHER_SENSORS)
)


def unit_matches(unit: str, accepted) -> bool:
    """Whether a reported unit is one of `accepted`. Case-insensitive ("kw" = "kW"), except single letters,
    which must match exactly: "C" is degrees Celsius, but "c" is cents (e.g. Predbat's cost sensors)."""
    unit = str(unit).strip()
    if len(unit) == 1:
        return unit in accepted
    return unit.lower() in {u.lower() for u in accepted if len(u) > 1}


def _is_valid_role(role: str) -> bool:
    """Check if role is a valid THERM role."""
    if role in VALID_THERM_ROLES:
        return True
    if role.startswith(ZONE_SENSOR_PREFIX) and role[len(ZONE_SENSOR_PREFIX):].isdigit():
        return True
    if role.startswith(ROOM_SENSOR_PREFIX) and role[len(ROOM_SENSOR_PREFIX):].isdigit():
        return True
    return False


@dataclass(frozen=True)
class EntityInfo:
    entity_id: str  # "sensor.x" or bare object_id "x" (Influx tags)
    platform: str | None = None  # integration domain from registry, e.g. "samsungehs"
    translation_key: str | None = None
    unique_id: str | None = None
    device_class: str | None = None
    unit: str | None = None
    domain: str | None = None  # inferred from entity_id when it has a dot

    def __post_init__(self) -> None:
        if self.domain is None and "." in self.entity_id:
            object.__setattr__(self, "domain", self.entity_id.split(".", 1)[0])


@dataclass
class PresetMatch:
    preset_id: str
    mapping: dict[str, str]  # therm role -> entity_id (as given in input)
    comparison: dict[str, str]
    unmatched_roles: list[str]
    score: float  # fraction of the preset's roles matched
    notes: list[str]


def _evaluate_entity(
    entity: EntityInfo,
    spec: dict[str, Any],
    preset: dict[str, Any],
) -> Optional[Tuple[int, int, int, str]]:
    """
    Evaluate if entity matches spec.
    Returns (tier, key_index, len(entity_id), entity_id) if matched, else None.
    key_index ranks keys in the order the preset lists them, so e.g. a text
    `…_value` feed listed first beats its numeric twin.
    """
    # 1. Platform constraint: only a role-level "platform" filters entities. The
    # preset-level match.platform decides whether the preset applies at all
    # (_preset_applies); helper entities such as template "…_value" labels built
    # from the integration's registers must still match.
    req_platform = spec.get("platform")
    if req_platform and entity.platform is not None:
        if entity.platform.lower() != req_platform.lower():
            return None

    # 2. Domain constraint
    req_domain = spec.get("domain")
    if req_domain and entity.domain is not None:
        if entity.domain.lower() != req_domain.lower():
            return None

    # 3. Device class constraint
    req_dc = spec.get("device_class")
    if req_dc and entity.device_class is not None:
        expected_dcs = [req_dc.lower()] if isinstance(req_dc, str) else [d.lower() for d in req_dc]
        if entity.device_class.lower() not in expected_dcs:
            return None

    excluded_dcs = spec.get("exclude_device_classes")
    if excluded_dcs and entity.device_class is not None:
        if entity.device_class.lower() in {d.lower() for d in excluded_dcs}:
            return None

    # 4. Units constraint
    req_units = spec.get("units")
    if req_units and entity.unit is not None and not unit_matches(entity.unit, req_units):
        return None
    # A name pattern alone is weak evidence (a "hall" light, a "kitchen" blind):
    # when the source does report metadata (the domain is known) and the entity
    # has no unit, it is not a measurement of the kind this role needs.
    if req_units and not spec.get("keys") and entity.unit is None and entity.domain is not None:
        return None

    # 5. Excludes constraint
    obj_id = entity.entity_id.split(".", 1)[-1].lower()
    excludes = spec.get("excludes", [])
    if any(exc.lower() in obj_id for exc in excludes):
        return None

    # 6. Rank evaluation
    keys = spec.get("keys", [])
    if keys:
        tier = None
        key_index = len(keys)

        # Tier 1: exact translation_key match
        if entity.translation_key is not None:
            for i, k in enumerate(keys):
                if entity.translation_key.lower() == k.lower():
                    tier, key_index = 1, i
                    break

        # Tier 2: unique_id suffix match
        if tier is None and entity.unique_id is not None:
            uid_lower = entity.unique_id.lower()
            for i, k in enumerate(keys):
                k_lower = k.lower()
                if uid_lower == k_lower or uid_lower.endswith(f"_{k_lower}"):
                    tier, key_index = 2, i
                    break

        # Tier 3: object_id suffix match
        if tier is None:
            for i, k in enumerate(keys):
                k_lower = k.lower()
                if obj_id == k_lower or obj_id.endswith(f"_{k_lower}"):
                    tier, key_index = 3, i
                    break

        if tier is None:
            return None

        return (tier, key_index, len(entity.entity_id), entity.entity_id)

    # Generic rules without keys (e.g. contains, patterns)
    matched_pattern = False
    if spec.get("patterns"):
        for pat in spec["patterns"]:
            if re.search(pat, obj_id, re.IGNORECASE):
                matched_pattern = True
                break
        if not matched_pattern:
            return None

    if spec.get("contains"):
        if not any(c.lower() in obj_id for c in spec["contains"]):
            return None

    if spec.get("contains_all"):
        if not all(c.lower() in obj_id for c in spec["contains_all"]):
            return None

    # Tier 4: Pattern / Contains match
    return (4, 0, len(entity.entity_id), entity.entity_id)


def _match_roles(
    role_specs: dict[str, Any],
    entities: list[EntityInfo],
    assigned_entities: set[str],
    preset: dict[str, Any],
    existing_mapping: dict[str, str],
    notes: list[str] | None = None,
) -> dict[str, str]:
    """Assign entities to roles tier by tier, avoiding duplicate assignments."""
    result: dict[str, str] = dict(existing_mapping)

    for tier in (1, 2, 3, 4):
        for role, spec in role_specs.items():
            if role in result:
                continue

            candidates: list[Tuple[Tuple[int, int, int, str], EntityInfo]] = []
            for e in entities:
                if e.entity_id in assigned_entities:
                    continue
                rank = _evaluate_entity(e, spec, preset)
                if rank is not None and rank[0] == tier:
                    candidates.append((rank, e))

            if candidates:
                candidates.sort(key=lambda c: c[0])
                best_rank, best_entity = candidates[0]
                result[role] = best_entity.entity_id
                assigned_entities.add(best_entity.entity_id)
                if notes is not None and spec.get("note"):
                    if spec["note"] not in notes:
                        notes.append(spec["note"])

    return result


def load_presets(directory: str | Path = "presets") -> list[dict]:
    """
    Load and validate all preset JSON definitions from directory.
    Raises ValueError on any unknown therm role in 'roles'.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        fallback = Path(__file__).resolve().parent / directory
        if fallback.exists():
            dir_path = fallback

    if not dir_path.exists() or not dir_path.is_dir():
        return []

    presets: list[dict] = []
    for json_file in sorted(dir_path.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # "Room_n" is a template for every room slot (Room_1 … Room_MAX_ROOMS), in that order.
        if "Room_n" in data.get("roles", {}):
            from config import MAX_ROOMS

            expanded = {}
            for role, spec in data["roles"].items():
                if role == "Room_n":
                    expanded.update({f"{ROOM_SENSOR_PREFIX}{i}": spec for i in range(1, MAX_ROOMS + 1)})
                else:
                    expanded[role] = spec
            data["roles"] = expanded

        # Validate therm roles
        roles = data.get("roles", {})
        for role in roles:
            if not _is_valid_role(role):
                raise ValueError(
                    f"Unknown therm role '{role}' in preset '{data.get('id', json_file.name)}'"
                )

        presets.append(data)

    return presets


def match_preset(
    preset: dict,
    entities: list[EntityInfo],
    existing: dict[str, str] | None = None,
) -> PresetMatch:
    """
    Match entities against a single preset.
    Respects existing mappings without overriding them.
    """
    existing_map = dict(existing) if existing else {}
    assigned_entities: set[str] = set(existing_map.values())
    notes = list(preset.get("notes", []))

    roles_dict = preset.get("roles", {})
    mapping = _match_roles(
        roles_dict,
        entities,
        assigned_entities,
        preset,
        existing_mapping=existing_map,
        notes=notes,
    )

    comp_dict = preset.get("comparison", {})
    comp_mapping = _match_roles(
        comp_dict,
        entities,
        assigned_entities,
        preset,
        existing_mapping={},
        notes=notes,
    )

    unmatched = [r for r in roles_dict if r not in mapping]
    # Zones are optional: a preset matches in full on a system without zone signals.
    base_roles = [r for r in roles_dict if not r.startswith(ZONE_SENSOR_PREFIX)]
    if base_roles:
        scored_total = len(base_roles) + len([r for r in mapping if r not in base_roles])
        score = len([r for r in mapping if r in roles_dict]) / scored_total
    else:
        score = (len(roles_dict) - len(unmatched)) / len(roles_dict) if roles_dict else 0.0

    return PresetMatch(
        preset_id=str(preset.get("id", "")),
        mapping=mapping,
        comparison=comp_mapping,
        unmatched_roles=unmatched,
        score=score,
        notes=notes,
    )


def suggest_mapping(
    entities: list[EntityInfo],
    presets: list[dict] | None = None,
    existing: dict[str, str] | None = None,
) -> list[PresetMatch]:
    """
    Suggest mappings across presets, sorted by match score descending.
    Never overrides roles present in existing and never assigns one entity to two roles.
    """
    if presets is None:
        presets = load_presets()
    existing_map = dict(existing) if existing else {}

    matches: list[PresetMatch] = []
    for p in presets:
        m = match_preset(p, entities, existing=existing_map)
        matches.append(m)

    # Sort descending by score, tie-break by preset_id
    matches.sort(key=lambda m: (-m.score, m.preset_id))
    return matches


# ----------------------------------------------------------------------
# Combined auto-mapping across presets
# ----------------------------------------------------------------------
@dataclass
class AutoMapResult:
    mapping: dict[str, str]                 # therm role -> entity_id
    sources: dict[str, tuple[str, int]]     # therm role -> (preset_id, tier 1-4; 4 = name pattern only)
    role_provenance: dict[str, str]         # newly assigned role -> source provenance (or "unknown")
    comparison: dict[str, str]              # reference channels (e.g. heat pump's own heat output)
    units: dict[str, str]                   # therm role -> unit reported by the source, where known
    presets_used: list[str]
    notes: list[str]


def _preset_applies(preset: dict, entities: list[EntityInfo]) -> bool:
    """A preset tied to an integration applies only if that integration's entities
    are present, or if no entity carries integration info (e.g. InfluxDB names)."""
    platform = (preset.get("match") or {}).get("platform")
    if not platform:
        return True
    known = [e.platform for e in entities if e.platform]
    if not known:
        return True
    return any(p.lower() == platform.lower() for p in known)


def auto_map(
    entities: list[EntityInfo],
    presets: list[dict] | None = None,
    existing: dict[str, str] | None = None,
) -> AutoMapResult:
    """
    One mapping from all applicable presets. Presets run in `priority` order
    (lower first; default 50), ties by match score, so e.g. a metered power
    channel beats the heat pump's own power estimate, an integration preset
    beats generic name patterns, and gaps in one integration (samsungehs has no
    valve/defrost) are filled from the next. Existing roles are never changed and
    no entity is used twice.
    """
    if presets is None:
        presets = load_presets()
    mapping = {k: v for k, v in (existing or {}).items() if v not in (None, "", "None")}
    sources: dict[str, tuple[str, int]] = {}
    role_provenance: dict[str, str] = {}
    comparison: dict[str, str] = {}
    used: list[str] = []
    notes: list[str] = []
    by_id = {e.entity_id: e for e in entities}

    applicable = [p for p in presets if _preset_applies(p, entities)]
    scored = [(p, match_preset(p, entities).score) for p in applicable]
    scored.sort(key=lambda ps: (ps[0].get("priority", 50), -ps[1], str(ps[0].get("id", ""))))

    for preset, _score in scored:
        assigned = set(mapping.values()) | set(comparison.values())
        before = dict(mapping)
        for tier in (1, 2, 3, 4):
            for role, spec in (preset.get("roles") or {}).items():
                if role in mapping:
                    continue
                best = None
                for e in entities:
                    if e.entity_id in assigned:
                        continue
                    rank = _evaluate_entity(e, spec, preset)
                    if rank is not None and rank[0] == tier:
                        if best is None or rank < best[0]:
                            best = (rank, e)
                if best:
                    mapping[role] = best[1].entity_id
                    sources[role] = (str(preset.get("id", "")), tier)
                    role_provenance[role] = str(
                        (preset.get("role_provenance") or {}).get(role, "unknown")
                    )
                    assigned.add(best[1].entity_id)
                    if spec.get("note") and spec["note"] not in notes:
                        notes.append(spec["note"])
        for role, spec in (preset.get("comparison") or {}).items():
            if role in comparison:
                continue
            cands = [(r, e) for e in entities if e.entity_id not in assigned
                     for r in [_evaluate_entity(e, spec, preset)] if r is not None]
            if cands:
                comparison[role] = min(cands, key=lambda c: c[0])[1].entity_id
                assigned.add(comparison[role])
        if mapping != before:
            used.append(str(preset.get("id", "")))
            notes.extend(n for n in preset.get("notes", []) if n not in notes)

    units = {role: by_id[eid].unit for role, eid in mapping.items()
             if eid in by_id and by_id[eid].unit}
    return AutoMapResult(mapping, sources, role_provenance, comparison, units, used, notes)
