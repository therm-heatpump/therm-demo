# sources/ha_api.py
"""
Home Assistant history source for THERM.

Reads recorder history over HA's REST API and returns the long-form frame
ha_loader.process_ha_frame expects (`entity_id`, `state`, `last_changed` in UTC),
the same shape as an HA history CSV export.

Connection:
- Inside the add-on: SUPERVISOR_TOKEN is set (homeassistant_api: true) and the
  API is reached through the Supervisor proxy at http://supervisor/core/api.
- Elsewhere (e.g. running locally for testing): THERM_HA_URL (e.g.
  http://homeassistant.local:8123) and THERM_HA_TOKEN (a long-lived access token).

Long-term statistics and the entity registry need HA's WebSocket API (not used yet).
No Streamlit import; errors never include the token.
"""

from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Optional

import pandas as pd
import requests

SUPERVISOR_API = "http://supervisor/core/api"


@dataclass
class HASettings:
    base_url: str          # ends with /api
    token: str
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> Optional["HASettings"]:
        """Supervisor proxy when running as an add-on, else THERM_HA_URL/THERM_HA_TOKEN, else None."""
        sup = os.environ.get("SUPERVISOR_TOKEN")
        if sup:
            return cls(base_url=SUPERVISOR_API, token=sup)
        url = os.environ.get("THERM_HA_URL", "").strip().rstrip("/")
        token = os.environ.get("THERM_HA_TOKEN", "").strip()
        if url and token:
            return cls(base_url=url if url.endswith("/api") else f"{url}/api", token=token)
        return None


def _get(settings: HASettings, path: str, params: dict | None = None, session: Any = None) -> Any:
    sess = session or requests.Session()
    headers = {"Authorization": f"Bearer {settings.token}", "Content-Type": "application/json"}
    try:
        resp = sess.get(f"{settings.base_url}{path}", params=params, headers=headers, timeout=settings.timeout)
    except Exception as e:
        raise RuntimeError(f"Home Assistant API request failed ({type(e).__name__})") from None
    status = getattr(resp, "status_code", 200)
    if status == 401:
        raise RuntimeError("Home Assistant API rejected the token (401)")
    if status >= 400:
        raise RuntimeError(f"Home Assistant API error {status} for {path.split('?')[0]}")
    return resp.json()


def _utc(value: Any, tz: str) -> pd.Timestamp:
    """A date/datetime in local `tz` (naive) or any aware timestamp → UTC."""
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, time.min)
    ts = pd.Timestamp(value)
    ts = ts.tz_localize(tz) if ts.tzinfo is None else ts
    return ts.tz_convert("UTC")


KNOWN_PLATFORMS = ("samsungehs", "ehs_sentinel", "esphome", "modbus", "mqtt", "shelly", "template")


def entity_platforms(settings: HASettings, platforms=KNOWN_PLATFORMS, session: Any = None) -> dict:
    """
    {entity_id: integration} for the given integrations, via HA's template API
    (`integration_entities()`), which works over REST with the add-on's
    Supervisor token. Unknown or unloaded integrations simply contribute nothing.
    Returns {} if the template API is unavailable.
    """
    names = ",".join(f"'{p}'" for p in platforms if p.replace("_", "").isalnum())
    template = (
        "{% for p in [" + names + "] %}{% for e in integration_entities(p) %}"
        "{{ e }}|{{ p }}\n{% endfor %}{% endfor %}"
    )
    sess = session or requests.Session()
    headers = {"Authorization": f"Bearer {settings.token}", "Content-Type": "application/json"}
    try:
        resp = sess.post(f"{settings.base_url}/template", json={"template": template},
                         headers=headers, timeout=settings.timeout)
        if getattr(resp, "status_code", 200) >= 400:
            return {}
        text = resp.text if hasattr(resp, "text") else str(resp.json())
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if "|" in line:
            eid, platform = line.strip().split("|", 1)
            out.setdefault(eid, platform)
    return out


def list_states(settings: HASettings, session: Any = None) -> pd.DataFrame:
    """
    Current entities with the metadata presets need: entity_id, domain,
    device_class, unit, friendly_name, state. (Registry fields such as platform
    and translation_key need the WebSocket API; see module docstring.)
    """
    rows = []
    for s in _get(settings, "/states", session=session):
        attrs = s.get("attributes", {}) or {}
        eid = s.get("entity_id", "")
        rows.append({
            "entity_id": eid,
            "domain": eid.split(".", 1)[0],
            "device_class": attrs.get("device_class"),
            "unit": attrs.get("unit_of_measurement"),
            "friendly_name": attrs.get("friendly_name"),
            "state": s.get("state"),
        })
    return pd.DataFrame(rows, columns=["entity_id", "domain", "device_class", "unit", "friendly_name", "state"])


def instance_config(settings: HASettings, session: Any = None) -> dict:
    """
    Instance configuration from GET /api/config for wizard defaults.

    Returns {"time_zone": str | None, "currency": str | None, "location_name": str | None, "country": str | None}
    normalised, or {} on any HTTP or JSON problem. Errors never include the token.
    """
    try:
        data = _get(settings, "/config", session=session)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    tz_raw = data.get("time_zone")
    time_zone: str | None = None
    if isinstance(tz_raw, str) and tz_raw.strip():
        import zoneinfo

        try:
            zoneinfo.ZoneInfo(tz_raw.strip())
            time_zone = tz_raw.strip()
        except Exception:
            time_zone = None

    cur_raw = data.get("currency")
    currency: str | None = None
    if isinstance(cur_raw, str) and cur_raw.strip():
        currency = cur_raw.strip().upper()

    loc_raw = data.get("location_name")
    location_name: str | None = None
    if isinstance(loc_raw, str) and loc_raw.strip():
        location_name = loc_raw.strip()

    country_raw = data.get("country")
    country: str | None = None
    if isinstance(country_raw, str) and country_raw.strip():
        country = country_raw.strip()

    return {
        "time_zone": time_zone,
        "currency": currency,
        "location_name": location_name,
        "country": country,
    }


def _iso_z(ts: pd.Timestamp) -> str:
    """UTC timestamp as 2026-09-20T23:00:00Z (no '+' to be mangled in URLs)."""
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


_TIME_KEYS = ("last_changed", "lc", "last_updated", "lu", "last_reported", "lr")


def _item_time(item: dict) -> Optional[str]:
    """
    Timestamp of one history item as an ISO string. The first item of each
    entity is a full state; later items (minimal_response) may use compact keys
    (`lc`/`lu`, epoch seconds) depending on the HA version. Reading only
    `last_changed` kept just the first row per entity (seen on HA 2026.9).
    """
    for key in _TIME_KEYS:
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return pd.Timestamp(value, unit="s", tz="UTC").isoformat()
        return str(value)
    return None


def _log_chunk(start: pd.Timestamp, end: pd.Timestamp, data: Any, secs: float = 0.0) -> None:
    """One diagnostic line per request in the add-on log (no token, no values)."""
    series = [s for s in (data or []) if s]
    rows = sum(len(s) for s in series)
    print(f"[therm ha] history {_iso_z(start)} -> {_iso_z(end)}: {len(series)} entities, {rows} rows "
          f"in {secs:.1f}s", flush=True)


def fetch_history(
    settings: HASettings,
    entity_ids: list[str],
    start: Any,
    end: Any,
    timezone: str = "Europe/Dublin",
    chunk_days: int = 2,
    session: Any = None,
) -> pd.DataFrame:
    """
    Recorder history for `entity_ids` over [start, end) (local dates or datetimes).

    Returns columns `entity_id`, `state`, `last_changed` (UTC ISO strings, as in
    an HA CSV export), sorted by time. Chunked by `chunk_days` to keep responses
    small. Each chunk's leading "state at start" rows are dropped after the
    first chunk, so a held value is not reported as a change at every boundary.
    """
    cols = ["entity_id", "state", "last_changed"]
    if not entity_ids:
        return pd.DataFrame(columns=cols)

    start_utc, end_utc = _utc(start, timezone), _utc(end, timezone)
    step = pd.Timedelta(days=max(1, chunk_days))
    records: list[dict] = []
    first_chunk = True
    cur = start_utc
    while cur < end_utc:
        nxt = min(cur + step, end_utc)
        params = {
            "filter_entity_id": ",".join(entity_ids),
            "end_time": _iso_z(nxt),
            "minimal_response": "",
            "no_attributes": "",
            "significant_changes_only": "0",
        }
        # "…Z" rather than "+00:00": a "+" in the URL path can be decoded as a
        # space by an intermediate proxy (the add-on goes through the Supervisor).
        t_req = _time.monotonic()
        data = _get(settings, f"/history/period/{_iso_z(cur)}", params=params, session=session)
        _log_chunk(cur, nxt, data, _time.monotonic() - t_req)
        for series in data or []:
            if not series:
                continue
            eid = series[0].get("entity_id")
            for i, item in enumerate(series):
                changed = _item_time(item)
                if changed is None:
                    continue
                # HA returns the state in force at the chunk start as the first row,
                # stamped at the chunk start. Keep it only for the very first chunk.
                if i == 0 and not first_chunk and pd.Timestamp(changed) <= cur:
                    continue
                state = item["state"] if "state" in item else item.get("s")  # compact key "s"
                records.append({"entity_id": item.get("entity_id", eid), "state": state,
                                "last_changed": changed})
        first_chunk = False
        cur = nxt

    df = pd.DataFrame(records, columns=cols)
    if df.empty:
        return df
    # format="ISO8601": timestamps with and without fractional seconds are mixed;
    # an inferred format silently turns the later ones into NaT.
    order = pd.to_datetime(df["last_changed"], utc=True, errors="coerce", format="ISO8601")
    return df.assign(_t=order).dropna(subset=["_t"]).sort_values("_t", kind="stable").drop(columns="_t").reset_index(drop=True)


# ----------------------------------------------------------------------
# Day cache + parallel requests
# ----------------------------------------------------------------------
# HA's history API is the slow part of the Home Assistant source (~10 s per
# 2-day request for ~30 entities, one after another).
# Completed local days are saved as one Parquet file per day, with a sidecar
# listing the entities it holds, so a repeat load only asks HA for today, and a
# newly mapped entity only for itself. Days are also kept after HA's recorder
# purges them. Missing days are requested in parallel.
HISTORY_COLUMNS = ["entity_id", "state", "last_changed"]
# A day older than this may already be partly purged by the recorder
# (keep_days defaults to 10), so it is used from the cache but never saved.
SAFE_CACHE_AGE_DAYS = 8
PARALLEL_REQUESTS = 4


def _day_paths(root, day: date):
    from pathlib import Path

    base = Path(root) / f"{day:%Y-%m-%d}"
    return base.with_suffix(".parquet"), base.with_suffix(".entities.json")


def _read_day(root, day: date) -> tuple[pd.DataFrame, set]:
    import json

    data, meta = _day_paths(root, day)
    if not (data.exists() and meta.exists()):
        return pd.DataFrame(columns=HISTORY_COLUMNS), set()
    try:
        return pd.read_parquet(data), set(json.loads(meta.read_text(encoding="utf-8")))
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS), set()


def _write_day(root, day: date, df: pd.DataFrame, entities: set) -> None:
    import json

    data, meta = _day_paths(root, day)
    data.parent.mkdir(parents=True, exist_ok=True)
    out = df.reindex(columns=HISTORY_COLUMNS).astype("string")
    tmp = data.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, data)
    tmp = meta.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(entities)), encoding="utf-8")
    os.replace(tmp, meta)  # written last: a day only counts once its data is complete


def _covered_entities(fetched: pd.DataFrame, requested: list, day_start: pd.Timestamp) -> set:
    """
    Entities whose history for the day is provably complete: HA starts each
    entity's history with the state in force at the window start, but only if
    it still holds an earlier state. With no such row the recorder may have
    purged the start of the day (retention shorter than SAFE_CACHE_AGE_DAYS) or the entity is new, so the day is not saved for it and
    is asked for again next time.
    """
    if fetched is None or fetched.empty:
        return set()
    ts = pd.to_datetime(fetched["last_changed"], utc=True, errors="coerce", format="ISO8601")
    leading = set(fetched.loc[ts <= day_start, "entity_id"])
    return {e for e in requested if e in leading}


def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    order = pd.to_datetime(df["last_changed"], utc=True, errors="coerce", format="ISO8601")
    return df.assign(_t=order).dropna(subset=["_t"]).sort_values("_t", kind="stable")


def fetch_history_cached(
    settings: HASettings,
    entity_ids: list[str],
    start: Any,
    end: Any,
    timezone: str = "Europe/Dublin",
    cache_root: Any = None,
    today: Optional[date] = None,
    workers: int = PARALLEL_REQUESTS,
    session: Any = None,
) -> pd.DataFrame:
    """
    fetch_history for local dates [start, end), one request per local day, with
    completed days read from / saved to `cache_root` (None = no cache).

    The result matches fetch_history: every day's leading "state at start" row
    is dropped except on the first day.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not entity_ids:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()
    today = today or pd.Timestamp.now(tz=timezone).date()
    wanted = set(entity_ids)
    days = [start_d + timedelta(days=i) for i in range((end_d - start_d).days)]

    frames: dict = {}
    jobs = []  # (day, entities to request, cached frame)
    for day in days:
        cached, have = _read_day(cache_root, day) if cache_root is not None else (None, set())
        complete = day < today
        missing = [e for e in dict.fromkeys(entity_ids) if not complete or e not in have]
        if cached is not None and complete and have:
            cached = cached[cached["entity_id"].isin(wanted)]
        else:
            cached = None
        if missing:
            jobs.append((day, missing, cached, have))
        else:
            frames[day] = cached

    def run(job):
        day, missing, _cached, _have = job
        lo = _utc(day, timezone)
        hi = _utc(day + timedelta(days=1), timezone)
        sess = session or requests.Session()
        return fetch_history(settings, missing, lo, hi, timezone=timezone, chunk_days=1, session=sess)

    if jobs:
        if session is not None or workers <= 1 or len(jobs) == 1:
            results = [run(j) for j in jobs]
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                results = list(pool.map(run, jobs))
        for (day, missing, cached, have), fetched in zip(jobs, results):
            parts = [f for f in (cached, fetched) if f is not None and not f.empty]
            frame = _sort_by_time(pd.concat(parts, ignore_index=True)).drop(columns="_t") if parts \
                else pd.DataFrame(columns=HISTORY_COLUMNS)
            frames[day] = frame
            too_old = (today - day).days > SAFE_CACHE_AGE_DAYS
            if cache_root is not None and day < today and not too_old:
                covered = _covered_entities(fetched, missing, _utc(day, timezone))
                if covered:
                    full, full_have = _read_day(cache_root, day)
                    new_rows = fetched[fetched["entity_id"].isin(covered)]
                    merged = pd.concat([full, new_rows], ignore_index=True) if not new_rows.empty else full
                    _write_day(cache_root, day, merged, full_have | covered)

    out = []
    for i, day in enumerate(days):
        f = frames.get(day)
        if f is None or f.empty:
            continue
        f = f.reset_index(drop=True)
        if i > 0:
            # Leading "state at start" rows: the first row per entity stamped at
            # (or before) midnight. The previous day already carries the state.
            ts = pd.to_datetime(f["last_changed"], utc=True, errors="coerce", format="ISO8601")
            lead = ~f["entity_id"].duplicated() & (ts <= _utc(day, timezone))
            f = f[~lead]
        out.append(f[HISTORY_COLUMNS])
    if not out:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.concat(out, ignore_index=True)
    df = _sort_by_time(df).drop(columns="_t").reset_index(drop=True).astype(object)
    # Parquet gives <NA> for a missing state; the loader expects None (as from HA's JSON).
    return df.where(df.notna(), None)
