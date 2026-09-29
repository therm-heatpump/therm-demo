"""
heartbeat_store.py

Automatic reusable heartbeat for the Home Assistant add-on (and any install with
persistent storage). Replaces the manual "generate in Data Quality, download,
upload before the next analysis" loop:

- After a long analysis (at least MIN_DAYS of raw reports) the heartbeat is
  built from that analysis' raw report history and saved per profile under
  $THERM_DATA_DIR/heartbeats/.
- It replaces the saved one only when it covers at least as many days, or the
  saved one ended more than REFRESH_AFTER_DAYS before the new one (seasonal
  drift, a changed sensor).
- Before processing, the saved heartbeat for the active profile is loaded, so a
  short period inherits the long-period reporting cadence and normal silences.
  Roles whose mapped entity has changed since the heartbeat was built are dropped.

No Streamlit dependency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import baselines as heartbeat_baselines
import profile_store

MIN_DAYS = 28
REFRESH_AFTER_DAYS = 90


def _path(profile_name: str) -> Path:
    """heartbeat_<safe name>_<hash of the full name>.json: names that sanitise
    alike ('Site/A', 'Site?A') still get their own file."""
    name = str(profile_name or "profile")
    folder = profile_store.data_dir() / "heartbeats"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"heartbeat_{profile_store.safe_stem(name)}_{profile_store.name_hash(name)}.json"
    if not path.exists():
        # Saved before the hash was added: adopt it only if it names this profile.
        legacy = folder / f"heartbeat_{profile_store.safe_stem(name)}.json"
        try:
            if legacy.exists() and json.loads(legacy.read_text(encoding="utf-8")).get("meta", {}).get(
                    "profile_name") == name:
                os.replace(legacy, path)
        except (OSError, json.JSONDecodeError):
            pass
    return path


def _object_id(entity_id: Any) -> str:
    return str(entity_id).split(".", 1)[-1]


def load(profile_name: str, mapping: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(baselines, meta) saved for this profile, or ({}, {}).

    With `mapping`, roles whose saved source entity is no longer the one mapped
    are left out (their cadence belongs to another sensor)."""
    path = _path(profile_name)
    if not path.exists():
        return {}, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}
    baselines = data.get("baselines") or {}
    meta = data.get("meta") or {}
    if mapping:
        kept = {}
        for role, entry in baselines.items():
            sources = {_object_id(s) for s in (entry.get("source_entity_ids") or [])}
            current = mapping.get(role)
            if sources and current and _object_id(current) not in sources:
                continue
            kept[role] = entry
        baselines = kept
    return baselines, meta


def _period(raw_events: pd.DataFrame) -> Tuple[int, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    times = pd.to_datetime(raw_events["last_changed"], errors="coerce")
    stamps = times.array
    valid = ~np.isnat(stamps.to_numpy(dtype="datetime64[ns]"))
    if not valid.any():
        return 0, None, None
    wall = stamps.tz_localize(None) if stamps.tz is not None else stamps
    days = np.unique(wall.asi8[valid] // (86_400 * 10**9)).size  # calendar days, like .dt.date
    return int(days), times.min(), times.max()


def maybe_update(profile_name: str, raw_events: Optional[pd.DataFrame], sensor_roles: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Build and save a heartbeat from this analysis when it is long enough and
    better than the saved one. Returns the saved meta, or None when unchanged."""
    if raw_events is None or raw_events.empty or "last_changed" not in raw_events.columns:
        return None
    days, start, end = _period(raw_events)
    if days < MIN_DAYS:
        return None

    _old, old_meta = load(profile_name)
    if old_meta:
        old_days = int(old_meta.get("days_analyzed") or 0)
        old_end = pd.to_datetime(old_meta.get("period_end"), errors="coerce", utc=True)
        new_end = pd.Timestamp(end)
        new_end = new_end.tz_localize("UTC") if new_end.tzinfo is None else new_end.tz_convert("UTC")
        stale = pd.isna(old_end) or (new_end - old_end) > pd.Timedelta(days=REFRESH_AFTER_DAYS)
        if days < old_days and not stale:
            return None

    # Data Quality's manual build drops unmapped reports before counting active
    # days; do the same, but only copy when there are unmapped rows.
    history = raw_events
    if "is_mapped" in history.columns and not history["is_mapped"].fillna(False).all():
        history = history[history["is_mapped"].fillna(False).astype(bool)]
    built = heartbeat_baselines.build_offline_aware_seasonal_baseline(history, sensor_roles)
    if not any(v.get("has_baseline") for v in built.values()):
        return None
    if built == _old:
        return None

    payload = heartbeat_baselines.heartbeat_baseline_payload(
        built, tag="automatic", days_in_history=days, profile_name=profile_name,
        period_start=start.isoformat() if start is not None else None,
        period_end=end.isoformat() if end is not None else None,
    )
    path = _path(profile_name)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return payload["meta"]


def describe(meta: Dict[str, Any]) -> str:
    """'saved automatically from 214 days (2026-03-01 – 2026-09-26)'."""
    days = meta.get("days_analyzed")
    start = str(meta.get("period_start") or "")[:10]
    end = str(meta.get("period_end") or "")[:10]
    span = f" ({start} – {end})" if start and end else ""
    return f"saved automatically from {days} days{span}" if days else "saved automatically"
