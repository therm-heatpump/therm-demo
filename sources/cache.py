# sources/cache.py
"""
Month-by-month disk cache for raw source rows.

Long periods are fetched one calendar month at a time and saved as one Parquet
file per (month, entity):

    <root>/<YYYY-MM>/<entity>.parquet      columns: Time, value, state, entity_id, count

- A month that ended before today is complete: it is fetched once and then read
  from disk. Adding an entity to a profile fetches only that entity.
- The current (incomplete) month is always fetched fresh and never saved.
- Entities with no data in a month are saved as empty files, so they are not
  re-queried every time.

The fetcher is any callable (entity_ids, start_date, end_date) -> raw DataFrame,
so the cache is independent of InfluxDB. No Streamlit import.
"""

from __future__ import annotations

import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

# `count` = readings aggregated into the row (1-minute rows from InfluxDB); NaN for raw rows.
RAW_COLUMNS = ["Time", "value", "state", "entity_id", "count"]
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

Fetcher = Callable[[list[str], date, date], pd.DataFrame]
Progress = Callable[[int, int, str], None]


def month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar months overlapping [start, end), as full-month [first, next_first) pairs."""
    months = []
    cur = date(start.year, start.month, 1)
    while cur < end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        months.append((cur, nxt))
        cur = nxt
    return months


def _entity_file(root: Path, month: date, entity: str) -> Path:
    return root / f"{month:%Y-%m}" / f"{_SAFE.sub('_', entity)}.parquet"


def _empty() -> pd.DataFrame:
    df = pd.DataFrame(columns=RAW_COLUMNS)
    df["Time"] = pd.to_datetime(df["Time"])
    df["value"] = df["value"].astype(float)
    df["count"] = df["count"].astype(float)
    return df


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Fetcher output in the cache schema (a fetcher may omit `count`)."""
    out = df.reindex(columns=RAW_COLUMNS)
    out["count"] = pd.to_numeric(out["count"], errors="coerce").astype(float)
    return out


def _save(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    out = _normalise(df) if not df.empty else _empty()
    out["state"] = out["state"].astype("string")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def cached_fetch(
    fetch: Fetcher,
    root: Optional[Path],
    entity_ids: Iterable[str],
    start: date,
    end: date,
    today: Optional[date] = None,
    progress: Optional[Progress] = None,
) -> pd.DataFrame:
    """
    Raw rows for `entity_ids` over [start, end) (local dates), using and filling
    the month cache under `root`. With root=None nothing is cached.
    """
    entities = sorted(set(entity_ids))
    today = today or date.today()
    months = month_ranges(start, end)
    parts: list[pd.DataFrame] = []
    # Entity ids as one shared category from the first file read: a Python string per
    # row made ten months of rows (7 M) ~4x larger. The same dtype keeps concat categorical.
    ids = pd.CategoricalDtype(entities)

    def _ids(df: pd.DataFrame) -> pd.DataFrame:
        df["entity_id"] = df["entity_id"].astype(str).astype(ids) if df["entity_id"].dtype == object \
            else df["entity_id"].astype(ids)
        return df

    for i, (m_start, m_end) in enumerate(months, start=1):
        complete = m_end <= today and root is not None
        label = f"{m_start:%b %Y}"
        if complete:
            missing = [e for e in entities if not _entity_file(root, m_start, e).exists()]
            if missing:
                if progress:
                    progress(i, len(months), f"Fetching {label} ({len(missing)} entities)")
                fetched = fetch(missing, m_start, m_end)
                for e in missing:
                    rows = fetched[fetched["entity_id"] == e] if not fetched.empty else fetched
                    _save(_entity_file(root, m_start, e), rows)
            elif progress:
                progress(i, len(months), f"{label} from cache")
            for e in entities:
                df = pd.read_parquet(_entity_file(root, m_start, e))
                if not df.empty:
                    parts.append(_ids(df))
        else:
            if progress:
                progress(i, len(months), f"Fetching {label} (current month)")
            lo, hi = max(m_start, start), min(m_end, end)
            fetched = fetch(entities, lo, hi)
            if not fetched.empty:
                parts.append(_ids(_normalise(fetched)))

    if not parts:
        return _empty()
    return _combine(parts, start, end)


def _combine(parts: list, start: date, end: date) -> pd.DataFrame:
    """Concatenate, window and sort. (An all-Arrow version measured worse: Arrow's
    memory pool kept ~0.7 GB of freed buffers resident for ten months of data.)"""
    out = pd.concat(parts, ignore_index=True)
    out["state"] = out["state"].astype(object).where(out["state"].notna(), None)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    tz = out["Time"].dt.tz
    if tz is not None:  # local-date bounds in the rows' own zone (sources keep aware times)
        lo = lo.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
        hi = hi.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
    out = out[(out["Time"] >= lo) & (out["Time"] < hi)]
    return out.sort_values("Time", kind="stable").reset_index(drop=True)


def cache_size_mb(root: Path) -> float:
    if not root or not root.exists():
        return 0.0
    return sum(p.stat().st_size for p in root.rglob("*.parquet")) / 1e6
