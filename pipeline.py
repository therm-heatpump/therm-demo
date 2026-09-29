# pipeline.py
"""
UI-free processing pipeline for the Grafana/Influx path.

Takes the dict returned by data_loader.load_and_clean_data / load_and_clean_frames
and runs the same steps app.get_processed_data runs for that path: physics
(gatekeepers), run detection, daily stats, diagnostics, heartbeat counts,
energy provenance and global stats.

Used by the parity tool and intended for headless runs (e.g. the add-on's
scheduled sensor job), so the UI and those callers share one sequence.
"""

from typing import Any, Dict, Optional

import processing
import baselines as heartbeat_baselines


def run_frames_pipeline(
    loaded: Optional[Dict[str, Any]],
    user_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Run physics, runs and daily aggregation on a loader result.

    Returns None when the loader found no usable data, otherwise a dict with
    df, runs, daily, global_stats and the loader's pass-through keys
    (patterns, raw_history, raw_events, provenance, provenance_summary,
    unmapped_entities, baselines).
    """
    if not loaded or loaded.get("df") is None or loaded["df"].empty:
        return None

    df = processing.apply_gatekeepers(loaded["df"], user_config)
    runs = processing.detect_runs(df, user_config)

    daily = processing.get_daily_stats(df)
    daily = processing.add_diagnostic_daily_metrics(daily, df, runs, user_config)
    daily = heartbeat_baselines.add_heartbeat_daily_counts(daily, loaded.get("raw_events"))
    daily = heartbeat_baselines.add_energy_provenance(
        daily, df, loaded.get("provenance"), loaded.get("patterns")
    )

    return {
        "df": df,
        "runs": runs,
        "daily": daily,
        "global_stats": processing.compute_global_stats(df),
        "patterns": loaded.get("patterns"),
        "raw_history": loaded.get("raw_history"),
        "raw_events": loaded.get("raw_events"),
        "provenance": loaded.get("provenance"),
        "provenance_summary": loaded.get("provenance_summary", {}),
        "unmapped_entities": loaded.get("unmapped_entities", []),
        "baselines": loaded.get("baselines") or {},
    }
