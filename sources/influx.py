# sources/influx.py
"""
InfluxDB 1.x data source for THERM.

Queries Home Assistant's InfluxDB v1 integration using InfluxQL over HTTP
and produces long-form pandas tables compatible with data_loader.load_and_clean_frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import os
import re
from typing import Any, Optional

import pandas as pd
import requests


@dataclass
class InfluxSettings:
    host: str
    port: int = 8086
    database: str = "homeassistant"
    username: str | None = None
    password: str | None = None
    ssl: bool = False
    timeout: float = 60.0
    version: int = 1

    @classmethod
    def from_env(cls) -> Optional["InfluxSettings"]:
        """Settings from THERM_INFLUX_* env vars, or None when no host is set."""
        host = os.environ.get("THERM_INFLUX_HOST", "").strip()
        if not host:
            return None
        port_raw = os.environ.get("THERM_INFLUX_PORT", "8086")
        try:
            port = int(port_raw)
        except ValueError:
            port = 8086

        database = os.environ.get(
            "THERM_INFLUX_DB",
            os.environ.get("THERM_INFLUX_DATABASE", "homeassistant"),
        )
        username = os.environ.get(
            "THERM_INFLUX_USER",
            os.environ.get("THERM_INFLUX_USERNAME"),
        )
        password = os.environ.get("THERM_INFLUX_PASSWORD")
        ssl_raw = os.environ.get("THERM_INFLUX_SSL", "").strip().lower()
        ssl = ssl_raw in ("1", "true", "yes")

        timeout_raw = os.environ.get("THERM_INFLUX_TIMEOUT", "60.0")
        try:
            timeout = float(timeout_raw)
        except ValueError:
            timeout = 60.0

        try:
            version = int(os.environ.get("THERM_INFLUX_VERSION", "1"))
        except ValueError:
            version = 1

        return cls(
            version=version,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            ssl=ssl,
            timeout=timeout,
        )


def _build_url(settings: InfluxSettings) -> str:
    scheme = "https" if settings.ssl else "http"
    return f"{scheme}://{settings.host}:{settings.port}/query"


def _escape_entity_id(eid: str) -> str:
    """Escape an InfluxQL string literal: backslashes first, so a backslash before a quote can't end it."""
    return eid.replace("\\", "\\\\").replace("'", "\\'")


# Entity names. HA's InfluxDB integration stores an entity as two tags: `entity_id` (the object id, e.g.
# `heat_pump_hot_water_mode`) and `domain` (e.g. `sensor`). therm names it the way Home Assistant does,
# `sensor.heat_pump_hot_water_mode`, so a profile reads the same with either source and two entities that
# share an object id in different domains (a sensor and an input_select helper, say) are never mixed. A
# bare name (a series without a domain tag, or an older profile) selects by entity_id alone.
_DOMAIN_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def split_name(name: str) -> tuple[str, Optional[str]]:
    """'sensor.x' -> ('x', 'sensor'); a bare name 'x' -> ('x', None)."""
    name = str(name)
    if "." in name:
        domain, object_id = name.split(".", 1)
        if _DOMAIN_RE.match(domain):
            return object_id, domain
    return name, None


def full_name(entity_id_tag: str, domain_tag: Optional[str]) -> str:
    return f"{domain_tag}.{entity_id_tag}" if domain_tag else str(entity_id_tag)


def _entity_filter(names) -> str:
    """InfluxQL condition selecting exactly these entities (with their domain when the name has one)."""
    parts = []
    for name in names:
        object_id, domain = split_name(name)
        cond = f"\"entity_id\" = '{_escape_entity_id(object_id)}'"
        if domain:
            cond = f"({cond} AND \"domain\" = '{_escape_entity_id(domain)}')"
        parts.append(cond)
    return " OR ".join(parts)


def _series_name(tags: dict, requested: set) -> str:
    """The requested name a returned series belongs to: `domain.x` when that was asked for, else `x`."""
    eid = tags.get("entity_id")
    full = full_name(eid, tags.get("domain"))
    return full if full in requested else eid


def _series_tags(settings: InfluxSettings, session: Optional[Any] = None):
    """(measurement, tags) for every series (SHOW SERIES)."""
    for r in _query(settings, "SHOW SERIES", session):
        for s in r.get("series", []):
            for row in s.get("values", []):
                key = str(row[0]) if row else ""
                # series key: <measurement>,<tag>=<value>,… (commas in values are escaped as "\,")
                parts = re.split(r"(?<!\\),", key)
                measurement = parts[0].replace("\\,", ",").replace("\\ ", " ")
                tags = {k: v.replace("\\ ", " ") for k, v in (p.split("=", 1) for p in parts[1:] if "=" in p)}
                yield measurement, tags


def _query(
    settings: InfluxSettings,
    q: str,
    session: Optional[Any] = None,
    **extra: str,
) -> list[dict[str, Any]]:
    """
    Run one InfluxQL query and return its `results` list.

    Credentials travel as HTTP basic auth, never in the URL, and error messages
    carry only host, status and Influx's own message, so a password cannot leak
    into logs or the UI.
    """
    if settings.version != 1:
        raise NotImplementedError("InfluxDB v2 support is planned")

    params = {"db": settings.database, "q": q, **extra}
    auth = (settings.username, settings.password or "") if settings.username else None
    sess = session or requests.Session()
    try:
        resp = sess.get(_build_url(settings), params=params, auth=auth, timeout=settings.timeout)
    except Exception as e:
        raise RuntimeError(
            f"InfluxDB HTTP request to {settings.host}:{settings.port} failed ({type(e).__name__})"
        ) from None

    status = getattr(resp, "status_code", 200)
    if status >= 400:
        detail = ""
        try:
            detail = str(resp.json().get("error", ""))
        except Exception:
            pass
        raise RuntimeError(f"InfluxDB HTTP error {status}" + (f": {detail}" if detail else ""))

    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"InfluxDB error: {data['error']}")
    results = data.get("results", [])
    for r in results:
        if "error" in r:
            raise RuntimeError(f"InfluxDB query error: {r['error']}")
    return results


def _is_not_null(val: Any) -> bool:
    """Check if value is non-null, non-NaN, and non-empty string."""
    if val is None:
        return False
    if pd.isna(val):
        return False
    if isinstance(val, str) and not val.strip():
        return False
    return True


def list_entities(settings: InfluxSettings, session: Optional[Any] = None) -> list[str]:
    """Every entity in the database, named `domain.object_id` as in Home Assistant (bare when a series has
    no domain tag), from SHOW SERIES."""
    return sorted({full_name(tags["entity_id"], tags.get("domain"))
                   for _m, tags in _series_tags(settings, session) if tags.get("entity_id")})


def list_entity_metadata(settings: InfluxSettings, session: Optional[Any] = None) -> list[dict]:
    """
    [{"entity_id", "domain", "unit"}] from SHOW SERIES, named like list_entities. HA's InfluxDB integration
    names each measurement after the entity's unit (e.g. "°C", "W", "L/min"), or after the entity id when it
    has none, and tags every point with `domain` and `entity_id`: enough for preset matching on names and
    units.
    """
    out: dict[str, dict] = {}
    for measurement, tags in _series_tags(settings, session):
        eid = tags.get("entity_id", "")
        if not eid:
            continue
        domain = tags.get("domain") or None
        unit = None if (not measurement or measurement == f"{domain}.{eid}" or measurement == eid) else measurement
        name = full_name(eid, domain)
        entry = out.setdefault(name, {"entity_id": name, "domain": domain, "unit": unit})
        if entry["unit"] is None and unit is not None:
            entry["unit"] = unit
    return sorted(out.values(), key=lambda d: d["entity_id"])


def _to_utc_timestamp(dt_val: Any, tz_str: str) -> pd.Timestamp:
    """
    Interprets dt_val as LOCAL time in `tz_str` and converts to UTC Timestamp.
    Supports datetime.date, datetime.datetime, strings, or pd.Timestamp.
    """
    if isinstance(dt_val, date) and not isinstance(dt_val, datetime):
        dt_val = datetime.combine(dt_val, time.min)

    ts = pd.Timestamp(dt_val)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz_str)
    else:
        ts = ts.tz_convert(tz_str)
    return ts.tz_convert("UTC")


def _format_utc_iso(ts: pd.Timestamp) -> str:
    """Format UTC Timestamp into standard InfluxQL ISO8601 string."""
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


RAW_COLUMNS = ["Time", "value", "state", "entity_id"]


def fetch_frames(
    settings: InfluxSettings,
    entity_ids: list[str],
    start: Any,
    end: Any,
    timezone: str = "Europe/Dublin",
    chunk_days: int = 7,
    session: Optional[Any] = None,
    retries: int = 2,
    state_entities: Optional[set] = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Fetch sensor readings from InfluxDB 1.x in chunked date ranges and return
    ([numeric_df], [state_df]) tables formatted for data_loader.load_and_clean_frames.

    Parameters:
        settings: InfluxSettings instance
        entity_ids: List of tag entity_id values to query
        start: Start date or datetime in local `timezone` (inclusive)
        end: End date or datetime in local `timezone` (exclusive)
        timezone: Profile timezone (e.g. "Europe/Dublin")
        chunk_days: Number of days per InfluxQL chunk query
        session: Optional requests.Session-like instance for test injection
        retries: extra attempts per chunk after a dropped connection
        state_entities: entities to treat as states (see classify_frames)
    """
    raw = fetch_raw(settings, entity_ids, start, end, timezone, chunk_days, session, retries)
    return classify_frames(raw, state_entities)


def fetch_raw(
    settings: InfluxSettings,
    entity_ids: list[str],
    start: Any,
    end: Any,
    timezone: str = "Europe/Dublin",
    chunk_days: int = 2,
    session: Optional[Any] = None,
    retries: int = 2,
) -> pd.DataFrame:
    """
    Unclassified rows `Time` (tz-naive local), `value` (float), `state` (str),
    `entity_id` for [start, end). Queried in `chunk_days` pieces; a chunk whose
    connection drops is retried `retries` times with a short backoff, so one
    failure does not lose a long fetch. This is the unit the month cache stores.
    """
    if not entity_ids:
        return pd.DataFrame(columns=RAW_COLUMNS)

    start_utc = _to_utc_timestamp(start, timezone)
    end_utc = _to_utc_timestamp(end, timezone)

    if start_utc >= end_utc:
        return pd.DataFrame(columns=RAW_COLUMNS)

    chunk_step = pd.Timedelta(days=max(1, chunk_days))
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_start = start_utc
    while cur_start < end_utc:
        cur_end = min(cur_start + chunk_step, end_utc)
        chunks.append((cur_start, cur_end))
        cur_start = cur_end

    entity_filter = _entity_filter(entity_ids)
    requested = set(entity_ids)
    raw_records: list[dict[str, Any]] = []

    for c_start, c_end in chunks:
        q = (
            f'SELECT "value", "state" FROM /.*/ '
            f"WHERE time >= '{_format_utc_iso(c_start)}' AND time < '{_format_utc_iso(c_end)}' "
            f'AND ({entity_filter}) GROUP BY "entity_id", "domain"'
        )
        for r in _query_with_retry(settings, q, session, retries):
            for s in r.get("series", []):
                tag_eid = _series_name(s.get("tags", {}), requested)
                cols = s.get("columns", [])
                if not cols:
                    continue
                t_idx = cols.index("time") if "time" in cols else 0
                v_idx = cols.index("value") if "value" in cols else None
                s_idx = cols.index("state") if "state" in cols else None

                for row in s.get("values", []):
                    val = row[v_idx] if v_idx is not None and v_idx < len(row) else None
                    st_val = row[s_idx] if s_idx is not None and s_idx < len(row) else None
                    raw_records.append({
                        "ts_ms": row[t_idx],
                        "value": val,
                        "state": st_val,
                        "entity_id": tag_eid,
                    })

    if not raw_records:
        return pd.DataFrame(columns=RAW_COLUMNS)

    df_all = pd.DataFrame(raw_records)

    # Epoch ms → UTC → the profile's zone, kept time-zone aware: dropping the zone
    # merged the repeated autumn hour and invented the skipped spring hour.
    df_all["Time"] = pd.to_datetime(df_all["ts_ms"], unit="ms", utc=True).dt.tz_convert(timezone)
    df_all["value"] = pd.to_numeric(df_all["value"], errors="coerce")
    df_all["state"] = df_all["state"].where(df_all["state"].map(_is_not_null), None)
    return df_all[RAW_COLUMNS]


def fetch_period(
    settings: InfluxSettings,
    entity_ids: list[str],
    start: Any,
    end: Any,
    timezone: str = "Europe/Dublin",
    state_entities: Optional[set] = None,
    cache_root: Optional[Any] = None,
    progress: Optional[Any] = None,
    today: Optional[date] = None,
    session: Optional[Any] = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Frames for data_loader.load_and_clean_frames over [start, end), the way the
    add-on loads long periods:

    - numeric entities: 1-minute means from InfluxDB (fetch_minute);
    - `state_entities` (entities mapped to schema_defs.STATE_ROLES), and any
      entity with no numeric value field (text-only sensors): every change
      event (fetch_raw), so states are held, never averaged;
    - month by month through the disk cache in `cache_root` (sources.cache):
      completed months are fetched once, the current month every time.
    """
    from sources import cache as month_cache

    states = set(state_entities or ())

    def fetch(ents: list[str], s: date, e: date) -> pd.DataFrame:
        numeric = [x for x in ents if x not in states]
        minute = (fetch_minute(settings, numeric, s, e, timezone, session=session) if numeric
                  else pd.DataFrame(columns=MINUTE_COLUMNS))
        found = set(minute["entity_id"]) if not minute.empty else set()
        raw_ids = [x for x in ents if x in states or x not in found]
        raw = (fetch_raw(settings, raw_ids, s, e, timezone, session=session) if raw_ids
               else pd.DataFrame(columns=RAW_COLUMNS))
        minute = minute.assign(state=None)
        return pd.concat([minute, raw], ignore_index=True)

    rows = month_cache.cached_fetch(fetch, cache_root, entity_ids, start, end, today=today, progress=progress)
    return classify_frames(rows, states)


MINUTE_COLUMNS = ["Time", "value", "count", "entity_id"]


def fetch_minute(
    settings: InfluxSettings,
    entity_ids: list[str],
    start: Any,
    end: Any,
    timezone: str = "Europe/Dublin",
    chunk_days: int = 7,
    session: Optional[Any] = None,
    retries: int = 2,
) -> pd.DataFrame:
    """
    1-minute means of numeric `value` computed by InfluxDB:

        SELECT MEAN("value"), COUNT("value") ... GROUP BY time(1m), "entity_id" FILL(none)

    Returns `Time` (minute start, local time with its zone), `value` (mean), `count`
    (readings in that minute), `entity_id`. This equals what the Grafana-CSV
    path computes from minute-truncated export rows (mean of every reading in the
    minute), while transferring one row per entity-minute instead of every
    reading (for ~30 sensors over 9 months, 2.5 M instead of 6.6 M rows).
    Minute buckets align with local minutes for any whole-minute UTC offset.
    """
    if not entity_ids:
        return pd.DataFrame(columns=MINUTE_COLUMNS)
    start_utc = _to_utc_timestamp(start, timezone)
    end_utc = _to_utc_timestamp(end, timezone)
    if start_utc >= end_utc:
        return pd.DataFrame(columns=MINUTE_COLUMNS)

    entity_filter = _entity_filter(entity_ids)
    requested = set(entity_ids)
    step = pd.Timedelta(days=max(1, chunk_days))
    records: list[tuple] = []
    cur = start_utc
    while cur < end_utc:
        nxt = min(cur + step, end_utc)
        q = (
            'SELECT MEAN("value") AS "value", COUNT("value") AS "count" FROM /.*/ '
            f"WHERE time >= '{_format_utc_iso(cur)}' AND time < '{_format_utc_iso(nxt)}' "
            f'AND ({entity_filter}) GROUP BY time(1m), "entity_id", "domain" FILL(none)'
        )
        for r in _query_with_retry(settings, q, session, retries):
            for s in r.get("series", []):
                eid = _series_name(s.get("tags", {}), requested)
                cols = s.get("columns", [])
                t_i, v_i, c_i = cols.index("time"), cols.index("value"), cols.index("count")
                for row in s.get("values", []):
                    if row[v_i] is not None and row[c_i]:
                        records.append((row[t_i], float(row[v_i]), int(row[c_i]), eid))
        cur = nxt

    if not records:
        return pd.DataFrame(columns=MINUTE_COLUMNS)
    df = pd.DataFrame(records, columns=["ts_ms", "value", "count", "entity_id"])
    # An entity whose unit (= measurement) changed appears as two series; combine
    # them into one count-weighted mean per minute.
    if df.duplicated(["ts_ms", "entity_id"]).any():
        df["weighted"] = df["value"] * df["count"]
        df = df.groupby(["ts_ms", "entity_id"], as_index=False).agg(weighted=("weighted", "sum"), count=("count", "sum"))
        df["value"] = df["weighted"] / df["count"]
    # Time-zone aware local time (see fetch_raw): clock-change days keep 23 / 25 hours.
    df["Time"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(timezone)
    return df[MINUTE_COLUMNS].sort_values(["entity_id", "Time"], kind="stable").reset_index(drop=True)


def _query_with_retry(settings: InfluxSettings, q: str, session: Optional[Any], retries: int) -> list[dict]:
    import time as _time

    for attempt in range(retries + 1):
        try:
            return _query(settings, q, session, epoch="ms")
        except RuntimeError as e:
            # Only a dropped/failed connection is worth retrying; Influx errors are not.
            if "request to" not in str(e) or attempt == retries:
                raise
            _time.sleep(1.5 * (attempt + 1))
    return []


def classify_frames(
    df_all: pd.DataFrame,
    state_entities: Optional[set] = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Split raw rows into ([numeric_df], [state_df]) for data_loader.load_and_clean_frames.

    An entity is a state if it has any text `state` over the range, or if it is in
    `state_entities` (entities mapped to schema_defs.STATE_ROLES): HA's InfluxDB
    integration stores numeric states such as a defrost code only in `value`.
    """
    if df_all is None or df_all.empty:
        return [], []
    forced_state = set(state_entities or ())

    numeric_frames: list[pd.DataFrame] = []
    state_frames: list[pd.DataFrame] = []

    # Classify per entity over the WHOLE range:
    # - any non-null state → state entity (columns Time, value, state, entity_id; keep if value or state non-null)
    # - otherwise numeric (columns Time, value, entity_id; drop null value, coerce to float)
    for eid, sub_df in df_all.groupby("entity_id", sort=False, observed=True):
        has_non_null_state = eid in forced_state or sub_df["state"].map(_is_not_null).any()

        if has_non_null_state:
            keep_mask = sub_df["value"].map(_is_not_null) | sub_df["state"].map(_is_not_null)
            clean_sub = sub_df.loc[keep_mask, ["Time", "value", "state", "entity_id"]].copy()
            # State rows are few; plain strings keep the (Time, entity) pivot free of empty categories.
            clean_sub["entity_id"] = clean_sub["entity_id"].astype(str)
            if not clean_sub.empty:
                state_frames.append(clean_sub)
        else:
            sub_copy = sub_df[["Time", "value", "entity_id"]].copy()
            sub_copy["value"] = pd.to_numeric(sub_copy["value"], errors="coerce")
            clean_sub = sub_copy.dropna(subset=["value"])
            if not clean_sub.empty:
                numeric_frames.append(clean_sub)

    numeric_dfs: list[pd.DataFrame] = []
    if numeric_frames:
        num_cat = pd.concat(numeric_frames, ignore_index=True)
        num_cat = num_cat.sort_values("Time").reset_index(drop=True)
        # One code per row instead of one Python string: ten months (7 M rows) went
        # from ~530 MB to ~120 MB. Readers group with observed=True.
        if not isinstance(num_cat["entity_id"].dtype, pd.CategoricalDtype):
            num_cat["entity_id"] = num_cat["entity_id"].astype("category")
        num_cat["entity_id"] = num_cat["entity_id"].cat.remove_unused_categories()
        numeric_dfs.append(num_cat)

    state_dfs: list[pd.DataFrame] = []
    if state_frames:
        st_cat = pd.concat(state_frames, ignore_index=True)
        st_cat = st_cat.sort_values("Time").reset_index(drop=True)
        state_dfs.append(st_cat)

    return numeric_dfs, state_dfs
