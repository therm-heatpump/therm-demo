"""Pure, on-demand construction of THERM debug download bundles."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable

import pandas as pd

import processing


SENSOR_DEBUG_COLUMNS = [
    "Power",
    "FlowTemp",
    "ReturnTemp",
    "FlowRate",
    "Freq",
    "DeltaT",
]


def build_debug_download_bundles(
    data: dict,
    *,
    config: dict | None = None,
    capabilities: dict | None = None,
    dataset_source: str = "unknown",
    include_engine_traces: bool = False,
    trace_builder: Callable[[pd.DataFrame], list[dict]] | None = None,
    inspector_summary=None,
    inspector_details=None,
    generated_at=None,
) -> dict:
    """Serialize and compress debug files only when explicitly requested.

    The function has no Streamlit dependency, making the expensive path directly
    benchmarkable and testable. ``data`` is the processed-data cache entry.
    """
    df = data.get("df")
    if hasattr(df, "latest"):
        # Month-by-month analysis (chunked.MonthFrames): the whole period is not held in
        # memory, so the merged CSV covers its most recent month.
        df = df.latest()
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("A non-empty processed dataframe is required")

    generated = pd.Timestamp.now(tz="UTC") if generated_at is None else pd.Timestamp(generated_at)
    if generated.tzinfo is None:
        generated = generated.tz_localize("UTC")
    else:
        generated = generated.tz_convert("UTC")
    timestamp_label = generated.strftime("%Y-%m-%dT%H-%M")

    merged_csv_bytes = df.to_csv(index=True).encode("utf-8")
    raw_history = data.get("raw_history")
    if not isinstance(raw_history, pd.DataFrame) or raw_history.empty:
        raw_history = None
        raw_csv_bytes = None
    else:
        raw_csv_bytes = raw_history.to_csv(index=True).encode("utf-8")

    coverage: dict[str, dict] = {}
    active_stats: dict[str, dict] = {}
    if raw_history is not None:
        total_rows = len(raw_history)
        for column in SENSOR_DEBUG_COLUMNS:
            if column not in raw_history.columns:
                continue
            series = pd.to_numeric(raw_history[column], errors="coerce")
            non_null = int(series.notna().sum())
            non_zero = int((series.notna() & series.ne(0)).sum())
            coverage[column] = {
                "total_rows": int(total_rows),
                "non_null": non_null,
                "non_null_pct": float(100 * non_null / total_rows) if total_rows else 0.0,
                "non_zero": non_zero,
                "non_zero_pct": float(100 * non_zero / total_rows) if total_rows else 0.0,
            }

        if "Power" in raw_history.columns:
            power = pd.to_numeric(raw_history["Power"], errors="coerce")
            present_columns = [column for column in SENSOR_DEBUG_COLUMNS if column in raw_history]
            active_sample = raw_history.loc[power.gt(500), present_columns].head(20)
            for column in active_sample.columns:
                values = pd.to_numeric(active_sample[column], errors="coerce").dropna()
                if not values.empty:
                    active_stats[column] = {
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "mean": float(values.mean()),
                        "count": int(len(values)),
                    }

    mapping = (config.get("mapping") or {}) if isinstance(config, dict) else {}
    if hasattr(inspector_summary, "to_dict"):
        inspector_summary = inspector_summary.to_dict(orient="list")

    engine_traces = []
    if include_engine_traces and trace_builder is not None:
        try:
            engine_traces = trace_builder(df)
        except Exception:
            engine_traces = []

    global_stats = data.get("global_stats")
    if not isinstance(global_stats, dict):
        global_stats = processing.compute_global_stats(df)

    debug_bundle = {
        "generated_at": generated.isoformat(),
        "app_version": processing.CALC_VERSION,
        "dataset_source": dataset_source,
        "config": config,
        "mapping": mapping,
        "global_stats": global_stats,
        "runs": data.get("runs") or [],
        "sensor_coverage": coverage,
        "active_sample_stats": active_stats,
        "capabilities": capabilities or {},
        "inspector_summary": inspector_summary,
        "inspector_details": inspector_details,
        "engine_debug_traces": engine_traces,
    }
    debug_json_bytes = json.dumps(debug_bundle, default=str).encode("utf-8")

    readme_bytes = (
        "THERM Debug Export README\n"
        f"Version: {processing.CALC_VERSION}\n"
        f"Generated at: {generated.isoformat()}\n\n"
        "Files in this archive:\n\n"
        "1) therm_merged_engine.csv\n"
        "   Final engine dataframe after physics, flags, and tariff logic.\n\n"
        "2) therm_debug_bundle.json\n"
        "   Configuration, mapping, global statistics, run summaries, sensor coverage,\n"
        "   capabilities, inspector details, and optional engine traces.\n\n"
        "3) therm_raw_prephysics.csv (full bundle only, when available)\n"
        "   Pre-physics loader/resampler dataframe for ingestion diagnostics.\n"
    ).encode("utf-8")

    def _zip(include_raw: bool) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("therm_readme.txt", readme_bytes)
            archive.writestr("therm_merged_engine.csv", merged_csv_bytes)
            if include_raw and raw_csv_bytes is not None:
                archive.writestr("therm_raw_prephysics.csv", raw_csv_bytes)
            archive.writestr("therm_debug_bundle.json", debug_json_bytes)
        return buffer.getvalue()

    return {
        "timestamp_label": timestamp_label,
        "merged_zip": _zip(include_raw=False),
        "all_zip": _zip(include_raw=True) if raw_csv_bytes is not None else None,
    }
