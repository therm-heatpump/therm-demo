# processed_cache.py
"""
Disk cache of processed results (engine frame, runs, daily table, …) so a
period that has been analysed before reopens without re-running the physics,
run detection and daily aggregation (~35–60 s for ten months).

Entries live in $THERM_DATA_DIR/cache/processed/<key>/ :
    <name>.parquet  for each DataFrame (exact dtypes, compact on disk)
    objects.pkl     for everything else (runs list, patterns, stats, …)
    objects.sig     HMAC-SHA256 of objects.pkl; the key is kept outside the
                    Samba-visible config share (see _secret_dir), and an entry
                    whose signature does not match is deleted unread

The key must identify everything the result depends on: the data request and a
fingerprint of the fetched data, the profile, the engine version and the active
heartbeat baseline (see app.get_processed_data). Only the newest MAX_ENTRIES
entries are kept. No Streamlit import; failures never break processing.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

import profile_store

MAX_ENTRIES = 2
_OBJECTS = "objects.pkl"
_SIGNATURE = "objects.sig"
_COMPLETE = ".complete"


def _secret_dir() -> Path:
    """
    Where the signing key lives: $THERM_SECRET_DIR, else the add-on's private
    /data folder (not on the Samba-visible /config share), else the data folder
    (a local install, where the cache and the key have the same owner anyway).
    """
    configured = os.environ.get("THERM_SECRET_DIR")
    if configured:
        return Path(configured)
    private = Path("/data")
    if private.is_dir() and os.access(private, os.W_OK):
        return private
    return profile_store.data_dir()


def _signing_key() -> bytes:
    path = _secret_dir() / ".therm_cache_key"
    try:
        key = path.read_bytes()
        if len(key) >= 32:
            return key
    except OSError:
        pass
    key = os.urandom(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(key)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)
    return key


def _sign(data: bytes) -> str:
    import hmac

    return hmac.new(_signing_key(), data, hashlib.sha256).hexdigest()


def cache_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _root(folder: str = "processed") -> Path:
    return profile_store.cache_dir(folder)


_FRAME_META = "__frame_meta__"


def _mixed_object_columns(df: pd.DataFrame) -> list:
    """Object columns holding more than one Python type (Parquet needs one type per column),
    e.g. state columns whose leading minutes are back-filled with 0 before the first label."""
    mixed = []
    for col in df.columns:
        if df[col].dtype == object:
            types = set(df[col].dropna().map(type))
            if len(types) > 1:
                mixed.append(col)
    return mixed


def load(key: str, root: Optional[Path] = None, only: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
    """The stored result, or None. `only`: names of the DataFrames to read (others are skipped)."""
    folder = (root or _root()) / key
    if not (folder / _COMPLETE).exists():
        return None
    try:
        import hmac

        data = (folder / _OBJECTS).read_bytes()
        signature = (folder / _SIGNATURE).read_text(encoding="ascii").strip()
        # Unpickling runs code, and /config is writable over Samba: only load
        # what this installation wrote, verified with a key kept outside /config.
        # A missing or wrong signature discards the entry.
        if not hmac.compare_digest(signature, _sign(data)):
            raise ValueError("processed-cache signature mismatch")
        result: Dict[str, Any] = pickle.loads(data)
        meta = result.pop(_FRAME_META, {})
        for f in folder.glob("*.parquet"):
            if only is not None and f.stem not in only:
                continue
            df = pd.read_parquet(f)
            info = meta.get(f.stem, {})
            for col, series in info.get("mixed", {}).items():
                df[col] = series.values
            if info.get("columns") is not None:
                df = df[info["columns"]]
            if info.get("freq") is not None and isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.DatetimeIndex(df.index, freq=info["freq"])
            df.attrs = info.get("attrs", {})
            result[f.stem] = df
        os.utime(folder / _COMPLETE)  # mark as recently used
        return result
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        return None


def save(key: str, result: Dict[str, Any], root: Optional[Path] = None,
         max_entries: int = MAX_ENTRIES, protect: Optional[set] = None) -> bool:
    """Store `result`; returns False (and leaves no partial entry) on any failure."""
    root = root or _root()
    folder = root / key
    tmp = root / f".{key}.tmp"
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        objects: Dict[str, Any] = {}
        meta: Dict[str, Dict[str, Any]] = {}
        for name, value in result.items():
            if isinstance(value, pd.DataFrame):
                mixed = _mixed_object_columns(value)
                try:
                    value.drop(columns=mixed).to_parquet(tmp / f"{name}.parquet")
                    meta[name] = {
                        "columns": list(value.columns),
                        "mixed": {c: value[c] for c in mixed},  # pickled; exact values and types
                        "attrs": dict(value.attrs),
                        "freq": getattr(value.index, "freq", None),  # Parquet drops index freq
                    }
                    continue
                except Exception:
                    (tmp / f"{name}.parquet").unlink(missing_ok=True)  # fall back to pickle
            objects[name] = value
        objects[_FRAME_META] = meta
        data = pickle.dumps(objects, protocol=pickle.HIGHEST_PROTOCOL)
        (tmp / _OBJECTS).write_bytes(data)
        (tmp / _SIGNATURE).write_text(_sign(data), encoding="ascii")
        (tmp / _COMPLETE).write_text(str(time.time()))
        shutil.rmtree(folder, ignore_errors=True)
        os.replace(tmp, folder)
        _prune(root, max_entries, (protect or set()) | {key})
        return True
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        return False


def _prune(root: Path, max_entries: int = MAX_ENTRIES, protect: Optional[set] = None) -> None:
    """Keep the newest `max_entries` entries, and never remove a protected one (it may take the
    store above its limit: correctness over disk space)."""
    protect = protect or set()
    entries = [p for p in root.iterdir() if p.is_dir() and (p / _COMPLETE).exists()]
    entries.sort(key=lambda p: (p / _COMPLETE).stat().st_mtime, reverse=True)
    for kept, entry in enumerate(entries, start=1):   # protected entries count, but are never removed
        if kept > max_entries and entry.name not in protect:
            shutil.rmtree(entry, ignore_errors=True)


class Store:
    """
    A separately pruned set of entries, e.g. the per-month results of long
    analyses (chunked.py): cache/<folder>/<key>/, newest `max_entries` kept,
    same signing and file layout as the whole-period entries.
    """

    PINS = "pins.json"

    def __init__(self, folder: str, max_entries: int):
        self.folder, self.max_entries = folder, max_entries

    def load(self, key: str, only: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        return load(key, _root(self.folder), only)

    def exists(self, key: str) -> bool:
        return (_root(self.folder) / key / _COMPLETE).exists()

    def save(self, key: str, result: Dict[str, Any], protect: Optional[set] = None) -> bool:
        """`protect`: keys that must survive this save's pruning (e.g. the months already
        saved by the analysis in progress), in addition to every pinned key."""
        return save(key, result, _root(self.folder), self.max_entries, (protect or set()) | self.pinned())

    def pin(self, owner: str, keys) -> None:
        """Record that the whole-period result `owner` (a processed-cache key) refers to `keys`;
        they are not pruned while that result is still in the processed cache."""
        pins = self._read_pins()
        pins[owner] = sorted(set(keys))
        root = _root(self.folder)
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / (self.PINS + ".tmp")
        tmp.write_text(json.dumps(pins), encoding="utf-8")
        os.replace(tmp, root / self.PINS)

    def pinned(self) -> set:
        """Keys pinned by whole-period results that still exist (stale pins are ignored)."""
        live = {o: k for o, k in self._read_pins().items() if (_root() / o / _COMPLETE).exists()}
        return {key for keys in live.values() for key in keys}

    def _read_pins(self) -> Dict[str, list]:
        try:
            return json.loads((_root(self.folder) / self.PINS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
