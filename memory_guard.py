# memory_guard.py
"""
Refuse analyses that would not fit in memory, instead of letting the kernel's
OOM killer end the add-on (and, in a global OOM, put Home Assistant itself at
risk). No Streamlit import.

The estimate is linear in entity-minutes, calibrated on a real installation:
10 months (426,240 minutes) × 29 entities peaked at 1.66 GB in the minute-level
InfluxDB pipeline, i.e. ~134 bytes per entity-minute. A 1.3 safety factor
covers Linux/glibc allocation overhead versus the Windows measurement.
"""

from __future__ import annotations

import os
from typing import Optional

BASE_BYTES = 250e6              # interpreter, Streamlit, libraries, session
BYTES_PER_ENTITY_MINUTE = 134 * 1.3
HEADROOM = 0.85                 # use at most this share of available memory


import threading

# One heavy analysis at a time per therm process: two browser
# sessions could otherwise each pass the check and together exhaust memory.
ANALYSIS_LOCK = threading.Lock()


def _host_available() -> Optional[float]:
    """MemAvailable from /proc/meminfo (the HA VM's, inside the add-on); None elsewhere."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _read_number(path: str) -> Optional[float]:
    try:
        with open(path, encoding="ascii") as fh:
            text = fh.read().strip()
    except OSError:
        return None
    return None if not text or text == "max" else float(text)


def _cgroup_available() -> Optional[float]:
    """Room left under the container's own memory limit (cgroup v2, else v1); None if unlimited."""
    for limit_path, usage_path in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                                   ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                                    "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        limit = _read_number(limit_path)
        if limit is None or limit >= 2 ** 60:  # "max" / v1's huge "unlimited" value
            continue
        usage = _read_number(usage_path) or 0.0
        return max(limit - usage, 0.0)
    return None


def available_bytes() -> Optional[float]:
    """The smaller of the host's available memory and the container's cgroup headroom; None if unknown."""
    values = [v for v in (_host_available(), _cgroup_available()) if v is not None]
    return min(values) if values else None


# Month-by-month processing (chunked.py, InfluxDB periods over more than one month):
# only one month window (a month plus a day either side) is processed at a time; what
# grows with the period is the fetched rows (categorical ids; rows only where a sensor
# reported) and the kept raw events (last 92 days). Measured on a real installation:
# fetched rows 3.9 B and events ~26 B per entity-minute; ten months peaked at 618 MB
# instead of 2.1 GB, five months of InfluxDB at 407 MB instead of 911 MB. The factors
# below allow ~2.5x denser sensors than that, plus the usual 1.3 allocation overhead.
MONTH_WINDOW_MINUTES = 33 * 1440
BYTES_PER_ENTITY_MINUTE_FETCHED = 10 * 1.3
RECENT_EVENT_MINUTES = 92 * 1440
BYTES_PER_ENTITY_MINUTE_EVENTS = 30 * 1.3


def estimate_peak_bytes(minutes: float, entities: int, monthly: bool = False) -> float:
    n = max(entities, 1)
    if not monthly:
        return BASE_BYTES + minutes * n * BYTES_PER_ENTITY_MINUTE
    return (BASE_BYTES
            + min(minutes, MONTH_WINDOW_MINUTES) * n * BYTES_PER_ENTITY_MINUTE
            + minutes * n * BYTES_PER_ENTITY_MINUTE_FETCHED
            + min(minutes, RECENT_EVENT_MINUTES) * n * BYTES_PER_ENTITY_MINUTE_EVENTS)


def check(minutes: float, entities: int, available: Optional[float] = None, monthly: bool = False) -> dict:
    """
    {"ok", "estimate", "available", "max_days"}. `ok` is True when memory is
    unknown (not Linux) or the guard is disabled with THERM_MEMORY_GUARD=off.
    `monthly`: the period is processed month by month (chunked.py).
    """
    if available is None:
        available = available_bytes()
    estimate = estimate_peak_bytes(minutes, entities, monthly)
    if available is None or os.environ.get("THERM_MEMORY_GUARD", "").lower() == "off":
        return {"ok": True, "estimate": estimate, "available": available, "max_days": None}
    budget = available * HEADROOM
    n = max(entities, 1)
    if monthly:
        fixed = estimate_peak_bytes(MONTH_WINDOW_MINUTES + RECENT_EVENT_MINUTES, entities, True) \
            - (MONTH_WINDOW_MINUTES + RECENT_EVENT_MINUTES) * n * BYTES_PER_ENTITY_MINUTE_FETCHED
        per_day = 1440 * n * BYTES_PER_ENTITY_MINUTE_FETCHED
    else:
        fixed, per_day = BASE_BYTES, 1440 * n * BYTES_PER_ENTITY_MINUTE
    max_days = max(int((budget - fixed) // per_day), 0)
    return {"ok": estimate <= budget, "estimate": estimate, "available": available, "max_days": max_days}
