# chunked.py
"""
Month-by-month analysis of long periods (Grafana/Influx frames). No Streamlit import.

Processing a whole period at once holds every sensor at 1-minute resolution in
memory, so the period length was capped by the host's RAM (memory_guard). Here
the period is processed one calendar month at a time, so the peak memory is one
month's worth whatever the period:

- Each month is processed with WARMUP extra time on both sides, so runs and held
  or interpolated values that cross the month boundary are computed as in a
  single pass. The result is then trimmed back to the month: minute rows, days
  and raw events inside it, and runs that *start* inside it (a run is never
  counted twice or lost).
- State sensors (valve, DHW, defrost…) report on change only and are held until
  the next change. Each window is seeded with every state entity's last value
  before the window, so a state set weeks earlier is still known.
- A finished month's result is saved (``store``) under a key made of its data
  fingerprint, the profile, the heartbeat and the calculation version, so later
  loads only process months whose data or settings changed.
- The whole-period result keeps no whole-period minute frame: "df" is a
  MonthFrames that loads a month's minutes from the store when a run is opened
  (and one month at a time for per-run costs). Days, runs, global stats and the
  provenance summary are combined from the months; raw events are kept for the
  last RECENT_EVENTS only (the automatic heartbeat needs 28+ days).

Checked against a single pass over ten months of real data: identical heat,
electricity, COP, cost, defrosts, all 1,946 runs and every minute row, and no missing-value
differences in the daily sensor columns; peak memory about 0.7 GB instead of 2.3 GB.

Reporting patterns (how long a silent sensor is held) are learned once for the whole request
and passed to every month; runs still open at a window's end are finished from the saved
months (_close_open_runs). Known difference from a single pass: a state is held across a data
outage that spans a month end only up to its normal limit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from typing import Any, Callable, Dict, Optional

import pandas as pd

import data_loader
import pipeline
import processing

WARMUP = pd.Timedelta(days=1)
# Raw report events kept for the whole-period result: the automatic heartbeat needs 28+ days,
# and all of them (7 M rows for ten months) was the largest single object.
RECENT_EVENTS = pd.Timedelta(days=92)
# Part of every month key: bump when the saved month layout changes.
MONTH_FORMAT = "month-v3"  # v3: whole-request reporting patterns, open-run flag

Progress = Callable[[int, int, str], None]


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar months of [start, end) (local dates, end exclusive), clipped to the period."""
    out = []
    cur = start
    while cur < end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        out.append((cur, min(nxt, end)))
        cur = nxt
    return out


def _bound(value, like: pd.Series | pd.Index) -> pd.Timestamp:
    """A local date/time as a Timestamp comparable with `like` (aware or naive)."""
    ts = pd.Timestamp(value)
    tz = getattr(getattr(like, "dt", like), "tz", None)
    if tz is not None and ts.tzinfo is None:
        return ts.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
    if tz is None and ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def slice_frames(numeric_dfs: list, state_dfs: list, lo, hi) -> tuple[list, list]:
    """
    Long-form frames restricted to [lo, hi). State frames also get each entity's
    last row before `lo`, re-stamped at `lo`, so held states survive the cut.
    """
    numeric_out = []
    for f in numeric_dfs:
        if f is None or f.empty:
            continue
        t = f["Time"]
        part = f[(t >= _bound(lo, t)) & (t < _bound(hi, t))]
        if not part.empty:
            numeric_out.append(part)
    state_out = []
    for f in state_dfs:
        if f is None or f.empty:
            continue
        t = f["Time"]
        lo_t, hi_t = _bound(lo, t), _bound(hi, t)
        part = f[(t >= lo_t) & (t < hi_t)]
        before = f[t < lo_t]
        if not before.empty:
            seed = before.groupby("entity_id", sort=False, observed=True).tail(1).copy()
            seed["Time"] = lo_t
            # An entity's own first row at exactly lo wins over the seed (kept last).
            part = pd.concat([seed, part], ignore_index=True)
        if not part.empty:
            state_out.append(part)
    return numeric_out, state_out


def _frames_fingerprint(numeric_dfs: list, state_dfs: list) -> str:
    digest = hashlib.sha256()
    for f in list(numeric_dfs) + list(state_dfs):
        cols = sorted(map(str, f.columns))
        digest.update("|".join(cols).encode())
        part = f[cols]
        obj = [c for c in cols if part[c].dtype == object]
        if obj:
            part = part.astype({c: str for c in obj})
        digest.update(pd.util.hash_pandas_object(part, index=False).to_numpy().tobytes())
    return digest.hexdigest()[:24]


def _in(index, lo, hi):
    return (index >= _bound(lo, index)) & (index < _bound(hi, index))


def _trim(result: Dict[str, Any], lo, hi) -> Dict[str, Any]:
    """Keep only what belongs to [lo, hi): minute rows, days, events and runs starting inside.
    The month's global stats and provenance counts are taken here, from its own minutes.
    A run still going at the window's last minute is flagged ``_open_end``: it may continue
    past the one-day warm-up, and process_period finishes it from the following months."""
    df = result["df"]
    window_last = df.index[-1] if len(df) else None
    df = df[_in(df.index, lo, hi)]
    out = {"df": df, "global_stats": processing.compute_global_stats(df), "frame_summary": frame_summary(df)}
    prov = result.get("provenance")
    if isinstance(prov, pd.DataFrame) and not prov.empty:
        out["provenance_summary"] = data_loader.summarize_provenance(prov[_in(prov.index, lo, hi)])
    else:
        out["provenance_summary"] = {}
    daily = result["daily"]
    out["daily"] = daily[_in(daily.index, lo, hi)] if daily is not None and not daily.empty else daily
    ev = result.get("raw_events")
    if isinstance(ev, pd.DataFrame) and not ev.empty and "last_changed" in ev:
        t = ev["last_changed"]
        out["raw_events"] = ev[(t >= _bound(lo, t)) & (t < _bound(hi, t))].reset_index(drop=True)
    else:
        out["raw_events"] = ev
    runs = result.get("runs") or []
    if runs:
        idx = pd.DatetimeIndex([r["start"] for r in runs])
        keep = _in(idx, lo, hi)
        out["runs"] = [r for r, k in zip(runs, keep) if k]
        for r in out["runs"]:
            if window_last is not None and r["end"] >= window_last:
                r["_open_end"] = True
    else:
        out["runs"] = []
    out["patterns"] = result.get("patterns")
    out["unmapped_entities"] = result.get("unmapped_entities", [])
    return out


def data_span(numeric_dfs: list) -> Optional[tuple]:
    """First and last minute with a numeric reading in the whole period (a single pass's timeline)."""
    times = [f["Time"] for f in numeric_dfs if f is not None and not f.empty]
    if not times:
        return None
    return min(t.min() for t in times).floor("min"), max(t.max() for t in times).floor("min")


def _window_span(span: Optional[tuple], w_lo, w_hi) -> Optional[tuple]:
    """The minutes a window's timeline must cover: its edges, within the period's data."""
    if span is None:
        return None
    first, last = span
    lo_t, hi_t = _bound(w_lo, pd.Index([first])), _bound(w_hi, pd.Index([first])) - pd.Timedelta(minutes=1)
    lo_t, hi_t = max(lo_t, first), min(hi_t, last)
    return (lo_t, hi_t) if lo_t <= hi_t else None


def process_month(numeric_dfs: list, state_dfs: list, user_config: Dict[str, Any], lo, hi,
                  heartbeat_baseline: Optional[Dict[str, Any]] = None,
                  span: Optional[tuple] = None,
                  patterns: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """One month [lo, hi) with WARMUP either side; None if the month has no data.
    `span` = data_span() of the whole period, so outages at the window edges keep their minutes;
    `patterns` = the whole request's reporting patterns, so silent sensors are held as in a
    single pass (without them each month would learn its own)."""
    w_lo, w_hi = pd.Timestamp(lo) - WARMUP, pd.Timestamp(hi) + WARMUP
    num, sta = slice_frames(numeric_dfs, state_dfs, w_lo, w_hi)
    if not num and not sta:
        return None
    loaded = data_loader.load_and_clean_frames(num, sta, user_config, heartbeat_baseline=heartbeat_baseline,
                                               index_span=_window_span(span, w_lo, w_hi),
                                               reporting_patterns=patterns)
    result = pipeline.run_frames_pipeline(loaded, user_config)
    if result is None:
        return None
    trimmed = _trim(result, lo, hi)
    return trimmed if not trimmed["df"].empty else None


def patterns_fingerprint(patterns: Optional[Dict[str, Any]]) -> str:
    """What of the reporting patterns changes a month's result: each sensor's report type and
    hold limit (to the second). Part of the month key."""
    if not patterns:
        return ""
    return json.dumps({k: [v.get("report_type"), round(float(v.get("gap_threshold_sec") or 0))]
                       for k, v in sorted(patterns.items())}, sort_keys=True)


def learning_frames(numeric_dfs: list, state_dfs: list, before) -> tuple[list, list]:
    """Frames the request's reporting patterns are learned from: everything before `before`
    (the first month still in progress), or everything when no month is complete. The month in
    progress changes at every load; learning from it would change the patterns, and with them
    every saved month's key, each time."""
    if before is None:
        return numeric_dfs, state_dfs
    cut = lambda f: f[f["Time"] < _bound(before, f["Time"])] if f is not None and not f.empty else f
    num, sta = [cut(f) for f in numeric_dfs], [cut(f) for f in state_dfs]
    if not any(f is not None and not f.empty for f in num + sta):
        return numeric_dfs, state_dfs
    return num, sta


def month_key(numeric_dfs: list, state_dfs: list, user_config: Dict[str, Any], lo, hi,
              heartbeat_baseline: Optional[Dict[str, Any]], calc_version: str,
              span: Optional[tuple] = None, patterns: Optional[Dict[str, Any]] = None) -> str:
    """Identity of one month's result: its window's data, profile, heartbeat, version and bounds."""
    w_lo, w_hi = pd.Timestamp(lo) - WARMUP, pd.Timestamp(hi) + WARMUP
    num, sta = slice_frames(numeric_dfs, state_dfs, w_lo, w_hi)
    parts = [
        MONTH_FORMAT, calc_version, str(lo), str(hi), str(WARMUP), str(_window_span(span, w_lo, w_hi)),
        _frames_fingerprint(num, sta),
        json.dumps(user_config, sort_keys=True, default=str),
        json.dumps(heartbeat_baseline or {}, sort_keys=True, default=str),
        patterns_fingerprint(patterns),
    ]
    return "month-" + hashlib.sha256("\n".join(parts).encode()).hexdigest()[:32]


class MonthFrames:
    """
    The minute-level engine frame of a month-by-month analysis, kept on disk one
    month at a time (the month results in `store`) instead of as one DataFrame
    for the whole period. Only what is looked at is loaded: the minutes of one run
    (slice), one month at a time for per-run costs, the latest month for debug
    files. Months that are not saved (unfinished, or no store) stay in memory.
    """

    def __init__(self, store: Optional[Any] = None):
        self._store = store
        self._parts: list[dict] = []   # {"lo", "hi" (Timestamps like the index), "key", "df"}
        self.columns: list = []
        self.first = self.last = None
        self.n_rows = 0
        self.days = 0
        self.flags = {"has_flowrate": False, "has_heat_sensor": False}
        self.missing_months: list = []  # saved months since removed from the store

    def add(self, lo, hi, summary: dict, key: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> None:
        """Record one month from its frame_summary(); its frame is kept in memory only when it
        has no saved copy (key None)."""
        if not summary or not summary.get("rows"):
            return
        like = pd.Index([summary["first"]])
        self._parts.append({"lo": _bound(lo, like), "hi": _bound(hi, like), "key": key,
                            "df": df if key is None else None})
        self.columns += [c for c in summary["columns"] if c not in set(self.columns)]
        self.first = summary["first"] if self.first is None else min(self.first, summary["first"])
        self.last = summary["last"] if self.last is None else max(self.last, summary["last"])
        self.n_rows += summary["rows"]
        self.days += summary["days"]
        for flag, value in summary["flags"].items():
            self.flags[flag] = self.flags.get(flag, False) or value

    @property
    def empty(self) -> bool:
        return self.n_rows == 0

    def keys(self) -> list:
        """Store keys of the saved months (for pinning them against pruning)."""
        return [p["key"] for p in self._parts if p["key"]]

    def is_complete(self) -> bool:
        """Every saved month still in the store (false once one has been pruned)."""
        if self._store is None:
            return True
        return all(self._store.exists(p["key"]) for p in self._parts if p["key"])

    def __len__(self) -> int:
        return self.n_rows

    def _load(self, part: dict) -> pd.DataFrame:
        if part["df"] is not None:
            return part["df"]
        stored = self._store.load(part["key"], only=("df",)) if self._store is not None else None
        if not stored or "df" not in stored:
            if part["lo"] not in self.missing_months:
                self.missing_months.append(part["lo"])
            return pd.DataFrame(columns=self.columns)
        return stored["df"]

    def slice(self, start, end) -> pd.DataFrame:
        """Minutes from `start` to `end` inclusive (like df.loc[start:end]), across months if needed."""
        parts = [p for p in self._parts if p["lo"] <= end and p["hi"] > start]
        if not parts:
            return pd.DataFrame(columns=self.columns)
        frames = [self._load(p) for p in parts]
        out = frames[0] if len(frames) == 1 else pd.concat(frames)
        return out.loc[start:end]

    def latest(self) -> pd.DataFrame:
        """The most recent month's minutes (debug files of a long analysis)."""
        return self._load(self._parts[-1]) if self._parts else pd.DataFrame(columns=self.columns)

    def to_frame(self) -> pd.DataFrame:
        """Every month as one DataFrame: tests and short periods only (this is what is avoided)."""
        frames = [self._load(p) for p in self._parts]
        return pd.concat(frames) if frames else pd.DataFrame(columns=self.columns)

    def run_costs_at_current_prices(self, runs: list, tariff_structure, as_of=None) -> list:
        """processing.run_costs_at_current_prices, one month at a time. A run that starts
        late in a month and ends in the next also gets the next month's first minutes."""
        out: list = [None] * len(runs)
        if not runs or not self._parts:
            return out
        starts = pd.DatetimeIndex([r["start"] for r in runs])
        for i, part in enumerate(self._parts):
            chosen = [j for j in range(len(runs)) if part["lo"] <= starts[j] < part["hi"]]
            if not chosen:
                continue
            segment = self._load(part)
            last_end = max(runs[j]["end"] for j in chosen)
            if last_end >= part["hi"] and i + 1 < len(self._parts):
                segment = pd.concat([segment, self._load(self._parts[i + 1]).loc[:last_end]])
            costs = processing.run_costs_at_current_prices(segment, [runs[j] for j in chosen],
                                                           tariff_structure, as_of=as_of)
            for j, cost in zip(chosen, costs):
                out[j] = cost
        return out


def _close_open_runs(frames: "MonthFrames", runs: list, user_config: Dict[str, Any],
                     step: pd.Timedelta = pd.Timedelta(days=7)) -> list:
    """
    Finish runs that were still going at the end of their month's window (flag ``_open_end``):
    the one-day warm-up is not an upper bound on a run (a heating run can last days). Such a
    run is detected again on the saved minutes from just before its start, a week at a time,
    until it ends inside the slice or the data ends, and its metrics are replaced. The minute
    frames are the months' engine output, so this is the same detection a single pass does.
    """
    out = []
    for r in runs:
        if not r.pop("_open_end", False):
            out.append(r)
            continue
        replacement, until = r, r["end"]
        while True:
            until = until + step
            seg = frames.slice(r["start"] - pd.Timedelta(minutes=30), until)
            if seg.empty:
                break
            match = [x for x in processing.detect_runs(seg, user_config) if x["start"] == r["start"]]
            if not match:
                break
            replacement = match[0]
            if replacement["end"] < seg.index[-1] or seg.index[-1] >= frames.last:
                break
        out.append(replacement)
    return out


def merge_provenance_summaries(parts: list) -> dict:
    """baselines.summarize_provenance of the whole period from per-month summaries: a sensor
    absent from a month counts that month's minutes as missing, as in a single pass."""
    parts = [p for p in parts if p]
    month_totals = [max((s["total_minutes"] for s in p.values()), default=0) for p in parts]
    total = sum(month_totals)
    sensors = sorted({s for p in parts for s in p})
    out = {}
    for sensor in sensors:
        observed = sum(p.get(sensor, {}).get("observed_minutes", 0) for p in parts)
        interpolated = sum(p.get(sensor, {}).get("interpolated_minutes", 0) for p in parts)
        held = sum(p.get(sensor, {}).get("held_minutes", 0) for p in parts)
        synthetic = interpolated + held
        out[sensor] = {
            "total_minutes": int(total),
            "observed_minutes": int(observed),
            "interpolated_minutes": int(interpolated),
            "held_minutes": int(held),
            "synthetic_minutes": int(synthetic),
            "missing_minutes": int(total - observed - synthetic),
            "observed_pct": round((observed / total) * 100, 2) if total else 0.0,
            "synthetic_pct": round((synthetic / total) * 100, 2) if total else 0.0,
        }
    return out


def process_period(numeric_dfs: list, state_dfs: list, user_config: Dict[str, Any], start: date, end: date,
                   heartbeat_baseline: Optional[Dict[str, Any]] = None, calc_version: str = "",
                   store: Optional[Any] = None, today: Optional[date] = None,
                   progress: Optional[Progress] = None) -> Optional[Dict[str, Any]]:
    """
    Month-by-month equivalent of data_loader.load_and_clean_frames + pipeline.run_frames_pipeline
    over [start, end). `store` (processed_cache.Store-like: load(key, only) / save(key, obj) -> bool)
    keeps finished months; a month is saved only once its window has ended before `today`.

    The result has the pipeline's keys, except that "df" is a MonthFrames (minutes stay on
    disk per month), "provenance" is None (its summary is kept) and "raw_events" holds only
    the last RECENT_EVENTS of the period (enough for the automatic heartbeat).
    """
    today = today or date.today()
    windows = month_windows(start, end)
    span = data_span(numeric_dfs)
    in_progress = next((lo for lo, hi in windows if (pd.Timestamp(hi) + WARMUP).date() > today), None)
    request_patterns = data_loader.request_reporting_patterns(
        *learning_frames(numeric_dfs, state_dfs, pd.Timestamp(in_progress) if in_progress else None),
        user_config, heartbeat_baseline)
    events_from = pd.Timestamp(end) - RECENT_EVENTS
    frames = MonthFrames(store)
    dailies, runs, stats, provenance, events, unmapped = [], [], [], [], [], set()
    patterns = None
    for i, (lo, hi) in enumerate(windows, start=1):
        label = f"{lo:%b %Y}"
        complete = (pd.Timestamp(hi) + WARMUP).date() <= today
        recent = pd.Timestamp(hi) > events_from
        key, month, saved = None, None, False
        if store is not None and complete:
            key = month_key(numeric_dfs, state_dfs, user_config, lo, hi, heartbeat_baseline, calc_version, span,
                            request_patterns)
            month = store.load(key, only=("daily", "raw_events") if recent else ("daily",))
            saved = month is not None
            if saved and progress:
                progress(i, len(windows), f"{label} from saved results")
        if not saved:
            if progress:
                progress(i, len(windows), f"Processing {label}")
            month = process_month(numeric_dfs, state_dfs, user_config, lo, hi, heartbeat_baseline, span,
                                  request_patterns)
            if key is not None:
                # Never prune a month this analysis has already saved (a long period can span more
                # months than the store keeps).
                saved = store.save(key, month or {}, protect=set(frames.keys()))
        if not month:
            continue
        frames.add(lo, hi, month.get("frame_summary") or frame_summary(month.get("df")),
                   key if saved else None, month.get("df"))
        if month.get("daily") is not None and not month["daily"].empty:
            dailies.append(month["daily"])
        runs += month.get("runs") or []
        stats.append(month.get("global_stats"))
        provenance.append(month.get("provenance_summary"))
        if recent and isinstance(month.get("raw_events"), pd.DataFrame):
            events.append(month["raw_events"])
        unmapped |= set(month.get("unmapped_entities") or [])
        patterns = month.get("patterns") or patterns  # the latest month's reporting behaviour
        month = None  # release this month's minutes before the next one is processed

    if frames.empty:
        return None
    daily = pd.concat(dailies)
    daily = daily[~daily.index.duplicated(keep="first")].sort_index()
    # A sensor silent for a whole month has no column in that month; a single pass counts 0 there.
    counts = [c for c in daily.columns if c.endswith(("_count", "_Count"))]
    daily[counts] = daily[counts].fillna(0)
    runs.sort(key=lambda r: r["start"])
    runs = _close_open_runs(frames, runs, user_config)
    for n, r in enumerate(runs, start=1):  # run ids stay unique and in time order across months
        if "id" in r:
            r["id"] = n
    return {
        "df": frames,
        "runs": runs,
        "daily": daily,
        "global_stats": processing.merge_global_stats(stats),
        "patterns": patterns,
        "raw_history": None,  # pre-physics copy: debug only, too large to keep for long periods
        "raw_events": data_loader._concat_event_parts(events),
        "provenance": None,
        "provenance_summary": merge_provenance_summaries(provenance),
        "unmapped_entities": sorted(unmapped),
        "baselines": heartbeat_baseline or {},
    }


def frame_summary(df: Optional[pd.DataFrame]) -> dict:
    """What MonthFrames needs to know about a month without reading its minutes again."""
    if df is None or df.empty:
        return {}
    heat = pd.to_numeric(df["Heat"], errors="coerce").fillna(0).abs().sum() if "Heat" in df.columns else 0
    return {
        "columns": list(df.columns),
        "first": df.index[0],
        "last": df.index[-1],
        "rows": len(df),
        "days": int(pd.Index(df.index.normalize()).nunique()),
        "flags": {"has_flowrate": bool("FlowRate" in df.columns and df["FlowRate"].notna().any()),
                  "has_heat_sensor": bool(heat > 0)},
    }
