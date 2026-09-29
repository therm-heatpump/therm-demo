# profile_store.py
"""
Persistent profile storage.

Profiles are saved as JSON under $THERM_DATA_DIR/profiles (the add-on sets
THERM_DATA_DIR=/config, its add-on config folder, which HA includes in backups; locally it defaults to
./profiles next to the app). No Streamlit dependency.

Connection settings and secrets are never stored in a profile (it can still hold
names and notes, so users should check it before sharing); save_profile strips any keys listed in
SECRET_KEYS defensively.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

SECRET_KEYS = {"influx", "influx_password", "password", "token", "api_key", "secrets"}
_SAFE = re.compile(r"[^A-Za-z0-9._ -]+")


def data_dir() -> Path:
    """$THERM_DATA_DIR (the add-on's /config), else the app folder."""
    base = os.environ.get("THERM_DATA_DIR")
    return Path(base) if base else Path(__file__).resolve().parent


def profiles_dir() -> Path:
    path = data_dir() / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(*parts: str) -> Path:
    """Data cache folder ($THERM_DATA_DIR/cache/...); safe to delete at any time."""
    return data_dir().joinpath("cache", *parts)


def name_hash(name: str) -> str:
    """Short digest of the full (unsanitised) name: 'Site/A' and 'Site?A' sanitise
    to the same stem, so the digest keeps their files apart."""
    import hashlib

    return hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:8]


def safe_stem(name: str) -> str:
    return (_SAFE.sub("_", str(name)).strip(" ._") or "profile")[:80]


def _filename(name: str) -> str:
    return f"therm_profile_{safe_stem(name)}_{name_hash(name)}.json"


def list_profiles() -> list[str]:
    """Saved profile names, newest first."""
    files = sorted(profiles_dir().glob("therm_profile_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    names = []
    for f in files:
        try:
            names.append(json.loads(f.read_text(encoding="utf-8")).get("profile_name") or f.stem)
        except (OSError, json.JSONDecodeError):
            continue
    return names


def _path_for(name: str) -> Path:
    for f in profiles_dir().glob("therm_profile_*.json"):
        try:
            if json.loads(f.read_text(encoding="utf-8")).get("profile_name") == name:
                return f
        except (OSError, json.JSONDecodeError):
            continue
    return profiles_dir() / _filename(name)


def save_profile(profile: Dict[str, Any]) -> Path:
    """Write the profile atomically; returns the file path."""
    name = str(profile.get("profile_name") or "profile")
    clean = {k: v for k, v in profile.items() if k not in SECRET_KEYS}
    clean["profile_name"] = name
    path = _path_for(name)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(clean, indent=2, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_profile(name: str) -> Dict[str, Any]:
    path = _path_for(name)
    if not path.exists():
        raise FileNotFoundError(f"No saved profile named {name!r}")
    return json.loads(path.read_text(encoding="utf-8"))


STATE_FILE = ".therm_state.json"


def get_state() -> Dict[str, Any]:
    """Installation preferences: active profile name, preferred data source."""
    path = profiles_dir() / STATE_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def update_state(**changes: Any) -> Dict[str, Any]:
    state = {**get_state(), **changes}
    path = profiles_dir() / STATE_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return state


def load_active_profile() -> Dict[str, Any] | None:
    """The profile last saved or chosen, or None if there is none (or it was deleted)."""
    name = get_state().get("active_profile")
    if not name:
        return None
    try:
        return load_profile(name)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def delete_profile(name: str) -> bool:
    path = _path_for(name)
    if path.exists():
        path.unlink()
        return True
    return False
