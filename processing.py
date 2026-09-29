# processing.py

import pandas as pd
import numpy as np
from engine_context import debug_enabled, debug_traces

from config import (
    THRESHOLDS,
    PHYSICS_THRESHOLDS,
    ENGINE_STATE_THRESHOLDS,
    NIGHT_HOURS,
    SENSOR_EXPECTATION_MODE,
    UNSCORED_DQ_MODES,
    CALC_VERSION,
    resolve_fluid_specific_heat_kj,
    resolve_heat_source_mode,
    heat_coefficient_w_per_lpm_k,
    resolve_weather_curve,
)
from utils import safe_div, strip_entity_prefix


# --- DYNAMIC HELPERS ---------------------------------------------------------


def get_active_zone_columns(df: pd.DataFrame) -> list:
    """Return columns that look like heating zone state flags (Zone_XXX)."""
    # Match numbered mapping roles only. The former length check existed to
    # exclude Zone_Config but silently capped discovery at Zone_99.
    return [
        column
        for column in df.columns
        if column.startswith("Zone_") and column.removeprefix("Zone_").isdigit()
    ]


def get_room_columns(df: pd.DataFrame) -> list:
    """Return columns that look like room temperature series (Room_XXX)."""
    return [c for c in df.columns if c.startswith("Room_")]


def get_friendly_name(internal_key, user_config) -> str:
    """
    Robust lookup for friendly names.

    Handles cases where config might be None, empty, or malformed.
    Falls back to the internal key if anything looks off.
    """
    if not isinstance(user_config, dict):
        return str(internal_key)

    mapping = user_config.get("mapping")
    if not isinstance(mapping, dict):
        return str(internal_key)

    val = mapping.get(internal_key, internal_key)
    entity_id = str(val)

    return strip_entity_prefix(entity_id)


def calculate_physics_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate physics-derived metrics that can be inferred cheaply from existing columns.
    Currently this just ensures DeltaT exists.
    """
    d = df.copy()

    # Delta T
    if "DeltaT" not in d.columns and {"FlowTemp", "ReturnTemp"}.issubset(d.columns):
        d["DeltaT"] = d["FlowTemp"] - d["ReturnTemp"]

    return d


# --- DEBUG HELPERS (optional) ------------------------------------------------
def _debug_engine_state(d: pd.DataFrame, label: str = "") -> None:
    """Lightweight debug helper to inspect the internal physics engine state.

    Behaviour:
    - If debug_enabled() is False → no-op.
    - If True → append a structured snapshot into debug_traces().
    - Never writes to the UI directly.
    """
    if not debug_enabled():
        return

    cols = [c for c in ["Power", "Heat", "FlowRate", "DeltaT", "Freq"] if c in d.columns]
    summary: dict[str, float | int | float] = {"rows": int(len(d))}
    for col in cols:
        ser = pd.to_numeric(d[col], errors="coerce").fillna(0)
        summary[f"{col}_nonzero"] = int((ser != 0).sum())
        summary[f"{col}_min"] = float(ser.min())
        summary[f"{col}_max"] = float(ser.max())

    # Append into a session-level trace log instead of writing to UI
    traces = debug_traces()
    if isinstance(traces, list):
        traces.append(
            {
                "type": "engine_state",
                "label": label,
                "summary": summary,
            }
        )
    elif isinstance(traces, dict):
        traces.setdefault("traces", []).append(
            {
                "type": "engine_state",
                "label": label,
                "summary": summary,
            }
        )



# --- HEAT + COP ENGINE -------------------------------------------------------
def _ensure_heat_and_cop(
    d: pd.DataFrame, thresholds: dict | None = None
) -> pd.DataFrame:
    """
    Canonical logic for:
      - Ensuring a 'Heat' channel exists (native or derived from hydraulics)
      - Deriving Power/Heat splits
      - Computing COP_Real and COP_Graph

    This replaces the previously duplicated HEAT OUTPUT blocks and ensures
    we only run this logic once per dataframe.
    """
    if thresholds is None or not isinstance(thresholds, dict):
        thresholds = {}

    # --- HEAT OUTPUT LOGIC ---
    # physics_thresholds["heat_source"] (profile setting):
    #   "sensor"     (default) use a mapped Heat output sensor / heat meter if it
    #                has data, otherwise calculate from hydraulics;
    #   "calculated" always calculate in therm from FlowRate x DeltaT. A mapped
    #                Heat sensor is kept as Heat_Sensor for comparison only and
    #                is never used as a fallback: without FlowRate, heat is
    #                unavailable (profile validation requires the hydraulics).
    has_flow = "FlowRate" in d.columns
    has_heat_col = "Heat" in d.columns
    has_heat_data = False
    heat_mode = resolve_heat_source_mode(thresholds.get("heat_source"))

    if has_heat_col:
        raw_heat = pd.to_numeric(d["Heat"], errors="coerce")
        # Availability before the engine zero-fills Heat, so the DQ tier cannot
        # count manufactured zeros. This is post-loader availability: values the
        # loader filled within its gap limits count, as for the hydraulic inputs.
        d["Heat_Sensor_Available"] = raw_heat.notna()
        # treat NaN as 0 when checking whether we have any signal
        heat_series = raw_heat.fillna(0)
        has_heat_data = heat_series.abs().sum() > 0
        d["Heat"] = heat_series  # normalise type
        if heat_mode == "calculated":
            d["Heat_Sensor"] = raw_heat
            has_heat_data = False

    # Precompute hydraulics series regardless; safe even if missing.
    # IMPORTANT: use Series defaults aligned to the index, not scalars,
    # so that .fillna() is always valid even when the column is absent.
    flow = pd.to_numeric(
        d.get("FlowRate", pd.Series(0, index=d.index)), errors="coerce"
    ).fillna(0)

    delta_t = pd.to_numeric(
        d.get("DeltaT", pd.Series(0, index=d.index)), errors="coerce"
    ).fillna(0)

    freq = pd.to_numeric(
        d.get("Freq", pd.Series(0, index=d.index)), errors="coerce"
    ).fillna(0)

    # Thresholds
    min_flow = thresholds.get("min_flow_rate_lpm", 0)
    min_freq = thresholds.get("min_freq_for_heat", 0)
    min_dt = thresholds.get("min_valid_delta_t", 0)
    max_dt = thresholds.get("max_valid_delta_t", 999)

    # Primary-circuit fluid (profile setting) → W per (L/min × °C)
    fluid_specific_heat_kj = resolve_fluid_specific_heat_kj(
        thresholds.get("fluid_specific_heat_kj")
    )
    heat_coefficient = heat_coefficient_w_per_lpm_k(fluid_specific_heat_kj)

    d.attrs["heat_source_setting"] = heat_mode
    if has_heat_data:
        d.attrs["heat_source"] = "native"
    elif has_flow:
        d.attrs["heat_source"] = "derived_hydraulic"
    else:
        d.attrs["heat_source"] = "unavailable"

    if not has_heat_data:
        if has_flow:
            # Basic physics: Heat [W] = c * m_dot * ΔT
            heat_raw = heat_coefficient * flow * delta_t

            # Gatekeepers
            # UPDATED: remove frequency as a hard gatekeeper
            valid = (
                (flow >= min_flow)
                # (freq >= min_freq)  # disabled as gate
                & (delta_t.abs() >= min_dt)
                & (delta_t.abs() <= max_dt)
            )

            d["Heat"] = 0.0
            d.loc[valid, "Heat"] = heat_raw[valid]
        else:
            # No Heat sensor and no FlowRate → no energy channel
            d["Heat"] = 0.0

    # Optional: debug gatekeepers (captured into engine_debug_traces)
    if debug_enabled():
        payload = {
            "rows_total": int(len(d)),
            "rows_flow_gt0": int((flow > 0).sum()),
            "rows_valid": int(valid.sum()) if has_flow else 0,
            "min_flow_threshold": float(min_flow),
            "min_freq_threshold": float(min_freq),
            "min_dt_threshold": float(min_dt),
            "max_dt_threshold": float(max_dt),
            "fluid_specific_heat_kj": float(fluid_specific_heat_kj),
            "heat_coefficient_w_per_lpm_k": float(heat_coefficient),
        }
        traces = debug_traces()
        if isinstance(traces, list):
            traces.append(
                {
                    "type": "heat_gatekeepers",
                    "label": "Heat gatekeepers",
                    "details": payload,
                }
            )
        elif isinstance(traces, dict):
            traces.setdefault("traces", []).append(
                {
                    "type": "heat_gatekeepers",
                    "label": "Heat gatekeepers",
                    "details": payload,
                }
            )


    # Heat only while the unit runs, plus pump run-on: after the compressor
    # stops the pump keeps circulating for ~1 minute and the water still
    # carries heat the run produced (typically a percent or two of the heat). Run-on
    # minutes are those within coast_down_window_min of an active minute with
    # flow still at or above min_flow_rate_lpm. (Applies to native or derived Heat.)
    if "is_active" in d.columns:
        active = d["is_active"].astype(bool)
        window = int(thresholds.get(
            "coast_down_window_min", ENGINE_STATE_THRESHOLDS.get("coast_down_window_min", 5)
        ) or 0)
        if window > 0 and has_flow:
            recent = active.astype(int).rolling(window + 1, min_periods=1).max().astype(bool)
            run_on = ~active & recent & (flow >= min_flow)
        else:
            run_on = pd.Series(False, index=d.index)
        d["is_run_on"] = run_on
        d.loc[~(active | run_on), "Heat"] = 0.0

    # Normalise Heat and DeltaT again to be safe (prevent NaN cascades)
    if "Heat" in d.columns:
        d["Heat"] = pd.to_numeric(d["Heat"], errors="coerce").fillna(0.0)
    if "DeltaT" in d.columns:
        d["DeltaT"] = pd.to_numeric(d["DeltaT"], errors="coerce").fillna(0.0)

    # ------------------------------------------------------------------
    # Negative Power readings are ignored (treated as 0 W).
    # Some meters read a few watts negative at standby (e.g. ~-2.4 W). This is a meter zero
    # offset: it doesn't vary with wind speed (-2.45 W calm, -2.27 W above
    # 8 m/s over 9.5 months). It isn't consumption, so it must not be
    # counted in energy or cost.
    # ------------------------------------------------------------------
    if "Power" in d.columns:
        d["Power"] = pd.to_numeric(d["Power"], errors="coerce").fillna(0)
        d["Power"] = d["Power"].clip(lower=0)

    # ------------------------------------------------------------------
    # Net heat accounting (EN 14511 / EN 14825 convention).
    # "Heat" is signed: negative when the unit takes heat back out of the
    # water circuit (reverse-cycle defrost, run start/stop transients).
    # All energy, COP and SCOP figures use this NET heat.
    #   Heat_Clean = max(Heat, 0)   → gross heat; charts and per-minute COP only
    #   Heat_Loss  = max(-Heat, 0)  → heat taken back (split by _classify_heat_losses)
    # ------------------------------------------------------------------
    d["Heat_Clean"] = d["Heat"].clip(lower=0)
    d["Heat_Loss"] = (-d["Heat"]).clip(lower=0)

# --- Power & Heat Splits + COP ---
    is_heating = d.get("is_heating", 0).astype(bool)
    is_dhw = d.get("is_DHW", 0).astype(bool)

    d["Power_Heating"] = np.where(is_heating, d["Power"], 0)
    d["Power_DHW"] = np.where(is_dhw, d["Power"], 0)

    # Run-on heat belongs to the run it follows (the valve still shows DHW
    # after a hot-water run, so is_dhw already covers those minutes).
    d["Heat_Heating"] = np.where(_heating_heat_mask(d), d["Heat"], 0)
    d["Heat_DHW"] = np.where(is_dhw, d["Heat"], 0)

    # Per-minute COP is a display series; a single defrost minute would
    # otherwise plot as a large negative spike. Run-on minutes (heat with ~0 W)
    # are left out so they don't plot as huge ratios.
    running = d.get("is_active", pd.Series(1, index=d.index)).astype(bool)
    d["COP_Real"] = safe_div(d["Heat_Clean"].where(running, 0.0), d["Power"])
    d["COP_Graph"] = d["COP_Real"].clip(lower=0, upper=10)

    return d


def _heating_heat_mask(d: pd.DataFrame) -> pd.Series:
    """Minutes whose heat counts as space heating: heating minutes plus pump
    run-on after a heating run (run-on while the valve is on DHW is DHW)."""
    idx = d.index
    is_heating = d.get("is_heating", pd.Series(False, index=idx)).astype(bool)
    is_dhw = d.get("is_DHW", pd.Series(False, index=idx)).astype(bool)
    run_on = d.get("is_run_on", pd.Series(False, index=idx)).astype(bool)
    return is_heating | (run_on & ~is_dhw)


# --- COOLING -----------------------------------------------------------------
def _fill_internal_gaps(mask: pd.Series, max_gap: int) -> pd.Series:
    """
    Set False runs of length <= max_gap to True when they are bounded by True on
    BOTH sides. Leading/trailing False runs and longer gaps are left untouched.
    """
    m = mask.astype(bool)
    if max_gap <= 0 or not m.any():
        return m
    block = (m != m.shift()).cumsum()
    info = pd.DataFrame({"val": m, "block": block}).groupby("block").agg(
        val=("val", "first"), size=("val", "size")
    )
    prev_true = info["val"].shift(1, fill_value=False).astype(bool)
    next_true = info["val"].shift(-1, fill_value=False).astype(bool)
    fill_block = (~info["val"]) & (info["size"] <= max_gap) & prev_true & next_true
    return m | block.map(fill_block).astype(bool)


def _classify_cooling(d: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """
    Flag sustained active cooling and take it out of all heating figures.

    Cooling minutes (is_cooling) are excluded from is_heating, Heat_Heating,
    Power_Heating, Heat_Clean (gross heat) and the heat-loss columns, and are
    reported separately as Power_Cooling and Cooling_Heat_Removed (W, positive).
    """
    if thresholds is None or not isinstance(thresholds, dict):
        thresholds = {}
    idx = d.index
    cool = pd.Series(False, index=idx)

    if {"FlowTemp", "DeltaT"}.issubset(d.columns):
        min_len = int(thresholds.get("cooling_min_minutes", 10))
        dt_min = float(thresholds.get("cooling_dt_c", 0.5))
        max_flow = float(thresholds.get("cooling_max_flow_c", 22.0))
        min_out = float(thresholds.get("cooling_min_outdoor_c", 15.0))
        gap = int(thresholds.get("cooling_gap_tolerance_min", 2) or 0)
        margin = int(thresholds.get("defrost_margin_min", 3) or 0)

        active = d["is_active"].astype(bool)
        dhw = d["is_DHW"].astype(bool)
        dfr = d.get("is_defrost", pd.Series(False, index=idx)).astype(bool)
        if margin > 0:
            dfr = dfr.astype(int).rolling(2 * margin + 1, center=True, min_periods=1).max().astype(bool)
        eligible = active & ~dhw & ~dfr

        flow_t = pd.to_numeric(d["FlowTemp"], errors="coerce")
        dt = pd.to_numeric(d["DeltaT"], errors="coerce")
        cand = eligible & (dt <= -dt_min) & (flow_t < max_flow)
        if "OutdoorTemp" in d.columns:
            cand &= pd.to_numeric(d["OutdoorTemp"], errors="coerce") > min_out

        # Bridge brief blips INSIDE a cooling spell: fill only false runs of at
        # most `gap` minutes that have candidates on both sides (and are still
        # eligible). Spell edges are never extended.
        if gap > 0:
            cand = _fill_internal_gaps(cand, gap) & (cand | eligible)
        spell = (cand != cand.shift(fill_value=False)).cumsum()
        spell_len = cand.groupby(spell).transform("size")
        cool = cand & (spell_len >= min_len)

    d["is_cooling"] = cool
    if cool.any():
        for col in ("Heat_Clean", "Heat_Loss", "Heat_Loss_Defrost", "Heat_Loss_Transient"):
            if col in d.columns:
                d.loc[cool, col] = 0.0
        d["is_heating"] = d["is_heating"].astype(bool) & ~cool
        d["Power_Heating"] = np.where(d["is_heating"], d["Power"], 0)
        d["Heat_Heating"] = np.where(_heating_heat_mask(d) & ~cool, d["Heat"], 0)
    d["Power_Cooling"] = np.where(cool, d["Power"], 0.0)
    d["Cooling_Heat_Removed"] = np.where(cool, (-d["Heat"]).clip(lower=0), 0.0)
    return d


# --- DEFROST + HEAT LOSS ATTRIBUTION -----------------------------------------
def _defrost_signal(d: pd.DataFrame, ignore_codes=None) -> pd.Series:
    """
    Boolean "defrost in progress" from whichever signal is available:
      - Defrost_Is_Active: synthetic, from ha_loader's Modbus decoding
      - Defrost: the mapped defrost sensor. Samsung reports a status code
        (0 = idle, non-zero = defrost stages) or an On/Off label.
    """
    idx = d.index
    if "Defrost_Is_Active" in d.columns:
        s = pd.to_numeric(d["Defrost_Is_Active"], errors="coerce").fillna(0)
        return s > 0
    if "Defrost" in d.columns:
        raw = d["Defrost"]
        num = pd.to_numeric(raw, errors="coerce")
        text_on = raw.astype(str).str.strip().str.lower().isin(
            ["on", "true", "yes", "active", "defrost"]
        )
        ignored = num.isin(list(ignore_codes or []))
        return ((num.fillna(0) > 0) & ~ignored) | text_on
    return pd.Series(False, index=idx)


def _immersion_signal(d: pd.DataFrame) -> pd.Series | None:
    """
    Boolean "immersion on/permitted" from the best available signal, or None:
      - Immersion_Is_On: synthetic, from ha_loader's Modbus decoding (register 85)
      - Immersion_Mode: the mapped immersion sensor (0/1 or On/Off labels)
    """
    for col in ("Immersion_Is_On", "Immersion_Mode"):
        if col in d.columns:
            raw = d[col]
            num = pd.to_numeric(raw, errors="coerce")
            text_on = raw.astype(str).str.strip().str.lower().isin(
                ["on", "true", "yes", "active", "heating"]
            )
            return (num.fillna(0) > 0) | text_on
    return None


def _classify_heat_losses(d: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """
    Identify defrost episodes and split Heat_Loss into defrost vs transient.

    Adds:
      is_defrost          - defrost in progress (episodes capped at defrost_max_minutes)
      Defrost_Start       - first minute of each episode (sum = defrost count)
      Heat_Loss_Defrost   - W taken back within defrost_margin_min of an episode
      Heat_Loss_Transient - all other W taken back (run start/stop, sensor lag)
    """
    if thresholds is None or not isinstance(thresholds, dict):
        thresholds = {}
    max_len = int(thresholds.get("defrost_max_minutes", 15) or 15)
    margin = int(thresholds.get("defrost_margin_min", 3) or 0)

    on = _defrost_signal(d, thresholds.get("defrost_ignore_codes", [7])).astype(bool)
    # Cap episodes: the status is on-change, so a missed "0" can hold it on for hours.
    episode = (on != on.shift(fill_value=False)).cumsum()
    minute_in_episode = on.groupby(episode).cumcount()
    capped = on & (minute_in_episode < max_len)

    d["is_defrost"] = capped
    d["Defrost_Start"] = capped & ~capped.shift(fill_value=False)

    if margin > 0:
        near = (
            capped.astype(int)
            .rolling(2 * margin + 1, center=True, min_periods=1)
            .max()
            .astype(bool)
        )
    else:
        near = capped

    loss = d.get("Heat_Loss", pd.Series(0.0, index=d.index))
    d["Heat_Loss_Defrost"] = np.where(near, loss, 0.0)
    d["Heat_Loss_Transient"] = np.where(near, 0.0, loss)
    return d



# --- TARIFF ENGINE -----------------------------------------------------------


def _parse_tariff_profiles(tariff_structure) -> list:
    """
    Normalise TARIFF_STRUCTURE into a sorted list of profiles:

    [
      {
        "valid_from": date,
        "rules": [
          {"name": str, "start": time, "end": time, "rate": float},
          ...
        ]
      },
      ...
    ]
    """
    # Allow dict forms (flat or day/night) by normalising to a single profile
    if isinstance(tariff_structure, dict):
        try:
            day_rate = float(tariff_structure.get("day_rate", 0.35))
        except Exception:
            day_rate = 0.35
        try:
            night_rate = float(tariff_structure.get("night_rate", day_rate))
        except Exception:
            night_rate = day_rate
        night_start = str(tariff_structure.get("night_start", "00:00"))
        night_end = str(tariff_structure.get("night_end", "07:00"))

        rules = [
            {"name": "Night", "start": night_start, "end": night_end, "rate": night_rate},
            {"name": "Day", "start": night_end, "end": night_start, "rate": day_rate},
        ]

        tariff_structure = [
            {
                "valid_from": tariff_structure.get("valid_from", "1970-01-01"),
                "rules": rules,
            }
        ]

    if not isinstance(tariff_structure, (list, tuple)):
        return []

    profiles: list[dict] = []

    for p in tariff_structure:
        if not isinstance(p, dict):
            continue
        try:
            vf = pd.to_datetime(p.get("valid_from", "1970-01-01")).date()
        except Exception:
            vf = pd.Timestamp.min.date()

        rules = []
        def _parse_hhmm(val: str) -> pd.Timestamp.time:
            # Handle common "24:00" input by clamping to 23:59:59 to avoid wrap errors
            s = str(val)
            if s in ("24:00", "24:00:00"):
                s = "23:59:59"
            return pd.to_datetime(s, errors="coerce").time()
        for r in p.get("rules", []):
            try:
                start = _parse_hhmm(r.get("start", "00:00"))
                end = _parse_hhmm(r.get("end", "00:00"))
            except Exception:
                continue

            try:
                rate = float(r.get("rate", 0.35))
            except Exception:
                rate = 0.35

            rules.append(
                {
                    "name": r.get("name", ""),
                    "start": start,
                    "end": end,
                    "rate": rate,
                }
            )

        profiles.append({"valid_from": vf, "rules": rules})

    profiles = [p for p in profiles if p["rules"]]
    profiles.sort(key=lambda p: p["valid_from"])
    return profiles


def _compute_tariff_series(
    index: pd.DatetimeIndex,
    tariff_structure,
    default_rate: float = 0.35,
) -> pd.Series:
    """
    Vectorised computation of per-timestamp electricity rate based on:

      - TARIFF_STRUCTURE: list of profiles with valid_from + rules
      - Rules with start/end times (supporting overnight wrap)

    Returns a Series aligned to `index`.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return pd.Series(default_rate, index=index)

    profiles = _parse_tariff_profiles(tariff_structure)
    if not profiles:
        return pd.Series(default_rate, index=index)

    result = pd.Series(default_rate, index=index, dtype="float64")
    dates = index.date

    for i, profile in enumerate(profiles):
        vf = profile["valid_from"]
        if i < len(profiles) - 1:
            next_vf = profiles[i + 1]["valid_from"]
            mask_profile = (dates >= vf) & (dates < next_vf)
        else:
            mask_profile = dates >= vf
        if i == 0:
            # Days before the first "valid from" use the earliest prices entered
            # (they used to fall back to a built-in 0.35/kWh).
            mask_profile = mask_profile | (dates < vf)

        if not mask_profile.any():
            continue

        idx_profile = index[mask_profile]
        secs = (
            idx_profile.hour * 3600
            + idx_profile.minute * 60
            + idx_profile.second
        )
        rates = np.full(len(idx_profile), default_rate, dtype="float64")

        for rule in profile["rules"]:
            start_s = (
                rule["start"].hour * 3600
                + rule["start"].minute * 60
                + rule["start"].second
            )
            end_s = (
                rule["end"].hour * 3600
                + rule["end"].minute * 60
                + rule["end"].second
            )

            if start_s < end_s:
                # Normal same-day window
                mask_rule = (secs >= start_s) & (secs < end_s)
            else:
                # Wrap-around window (e.g. 23:00 → 02:00)
                mask_rule = (secs >= start_s) | (secs < end_s)

            if mask_rule.any():
                rates[mask_rule] = rule["rate"]

        result.loc[idx_profile] = rates

    return result


def current_prices(tariff_structure, as_of=None):
    """
    The price period effective on ``as_of`` (today by default), applied to every
    date: "what would this have cost at today's prices". Future-dated changes are
    ignored until they take effect.
    """
    if isinstance(tariff_structure, dict):
        return {k: v for k, v in tariff_structure.items() if k != "valid_from"}
    if not isinstance(tariff_structure, (list, tuple)) or not tariff_structure:
        return tariff_structure
    dated = [p for p in tariff_structure if isinstance(p, dict)]
    if not dated:
        return tariff_structure
    def _from(p):
        ts = pd.to_datetime(p.get("valid_from", "1970-01-01"), errors="coerce")
        return pd.Timestamp("1970-01-01") if pd.isna(ts) else ts

    effective_date = pd.Timestamp.today().date() if as_of is None else pd.to_datetime(as_of).date()
    effective = [p for p in dated if _from(p).date() <= effective_date]
    selected = max(effective, key=_from) if effective else min(dated, key=_from)
    return [{**selected, "valid_from": "1970-01-01"}]


def run_costs_at_current_prices(df: pd.DataFrame, runs: list, tariff_structure, as_of=None) -> list:
    """
    Electricity cost of each run (same minutes and Power as the run's own cost) at
    today's prices (current_prices). Returns one value per run, None without Power.
    """
    if df is None or df.empty or "Power" not in df.columns or not runs:
        return [None] * len(runs or [])
    rate = _compute_tariff_series(df.index, current_prices(tariff_structure, as_of=as_of))
    power = pd.to_numeric(df["Power"], errors="coerce").fillna(0.0)
    cumulative = ((power / 1000.0 / 60.0) * rate).cumsum().to_numpy()
    index = df.index
    out = []
    for r in runs:
        i0 = index.searchsorted(r["start"], side="left")
        i1 = index.searchsorted(r["end"], side="right") - 1
        if i1 < i0:
            out.append(None)
            continue
        out.append(float(cumulative[i1] - (cumulative[i0 - 1] if i0 > 0 else 0.0)))
    return out


def get_rate_for_timestamp(ts, tariff_structure, default_rate: float = 0.35) -> float:
    """
    Convenience helper: rate for a single timestamp.

    Uses the same tariff logic as the vectorised engine.
    """
    if not isinstance(ts, pd.Timestamp):
        ts = pd.to_datetime(ts)

    series = _compute_tariff_series(
        pd.DatetimeIndex([ts]), tariff_structure, default_rate=default_rate
    )
    if series.empty:
        return default_rate
    return float(series.iloc[0])


def apply_weather_fallbacks(d: pd.DataFrame) -> pd.DataFrame:
    """
    Fill gaps in each primary weather column from its secondary source
    (schema_defs.WEATHER_FALLBACKS). A primary reading is never replaced; a
    primary role that is not mapped is taken wholly from the secondary.

    Filled minutes are flagged in <primary>_From_Secondary (0/1) so the Data
    Quality view can score the primary sensor on its own readings, and totalled
    per role in d.attrs["weather_fallback_minutes"]. Expects units already
    converted, so both columns share the base unit.
    """
    from schema_defs import WEATHER_FALLBACKS

    filled: dict[str, int] = {}
    for primary, secondary in WEATHER_FALLBACKS.items():
        if secondary not in d.columns:
            continue
        backup = pd.to_numeric(d[secondary], errors="coerce")
        if primary in d.columns:
            current = pd.to_numeric(d[primary], errors="coerce")
        else:
            current = pd.Series(np.nan, index=d.index)
        gaps = current.isna() & backup.notna()
        d[primary] = current.where(~gaps, backup)
        d[f"{primary}_From_Secondary"] = gaps.astype("int8")
        filled[primary] = int(gaps.sum())
    d.attrs["weather_fallback_minutes"] = filled
    return d


# --- MAIN GATEKEEPER / ENGINE ------------------------------------------------
def apply_gatekeepers(df: pd.DataFrame, user_config: dict | None = None) -> pd.DataFrame:
    """
    Core "engine" that:
    - Normalises physics fields (DeltaT)
    - Classifies activity + DHW/Heating
    - Ensures Heat + splits + COP
    - Derives Zone configuration strings
    - Infers Immersion activity/power
    - Applies tariff to build incremental cost
    """
    d = df.copy()

    # Normalise units up front (honours user-selected units like km/h for wind)
    try:
        if isinstance(user_config, dict):
            units_cfg = user_config.get("units") or {}
            if units_cfg:
                from data_normalizer import convert_units
                d = convert_units(d, units_cfg)
    except Exception:
        # Unit conversion is best-effort; never break the pipeline
        pass

    # Secondary weather sources fill gaps in the primary sensors (after unit
    # conversion, so e.g. a km/h backup is in m/s before it is merged)
    d = apply_weather_fallbacks(d)

    # Merge physics thresholds with any user overrides
    physics_defaults = PHYSICS_THRESHOLDS if isinstance(PHYSICS_THRESHOLDS, dict) else {}
    user_phys = user_config.get("physics_thresholds", {}) if isinstance(user_config, dict) else {}
    thresholds = {**physics_defaults, **user_phys}

    # Merge general thresholds
    general_defaults = THRESHOLDS if isinstance(THRESHOLDS, dict) else {}
    user_thresh = user_config.get("thresholds", {}) if isinstance(user_config, dict) else {}
    general_thresholds = {**general_defaults, **user_thresh}

    # 1. Physics
    d = calculate_physics_metrics(d)
    _debug_engine_state(d, "after physics (DeltaT)")

    # 2. Activity
    # "Power on threshold (W)" from System Setup, stored in physics_thresholds.
    power_on_w = float(
        thresholds.get("power_on_W", ENGINE_STATE_THRESHOLDS.get("power_on_W", 150.0))
    )
    if "Power" not in d.columns:
        raise KeyError(
            "Power column missing after mapping. Please ensure the Power role "
            "is mapped to a valid entity in System Setup."
        )
    d["is_active"] = (d["Power"] > power_on_w).astype(int)

    # 3. Detect DHW Mode (Valve-first, Status-second, DHW_Mode not used for detection)
    idx = d.index
    is_dhw_mask = pd.Series(False, index=idx)

    # Forward-fill sparse DHW/valve indicators (Grafana state feeds are on-change)
    ffill_cols = [
        "Valve_Is_DHW",
        "Valve_Is_Heating",
        "DHW_Status_Is_On",
        "DHW_Active",
    ]
    for col in ffill_cols:
        if col in d.columns:
            # For on-change state signals, keep last known value for up to 12 hours (720 minutes)
            d[col] = d[col].ffill(limit=720)

    # Forward-fill textual ValveMode labels over the same horizon
    if "ValveMode" in d.columns:
        d["ValveMode"] = d["ValveMode"].ffill(limit=720)

    # --- Valve evidence (primary) -------------------------------------------
    valve_dhw_mask = pd.Series(False, index=idx)
    valve_known = pd.Series(False, index=idx)

    if "Valve_Is_DHW" in d.columns:
        v = pd.to_numeric(d["Valve_Is_DHW"], errors="coerce")
        valve_known |= v.notna()
        valve_dhw_mask |= v.fillna(0) > 0.5

    if "Valve_Is_Heating" in d.columns:
        vh = pd.to_numeric(d["Valve_Is_Heating"], errors="coerce")
        valve_known |= vh.notna()

    if "ValveMode" in d.columns:
        valve_str = d["ValveMode"].astype(str).str.strip().str.lower()
        # DHW / hot water circuit (EHS-Sentinel reports the NASA 3-way valve as TANK/ROOM)
        text_dhw = valve_str.str.contains("dhw|hot water|hot_water|tank|cylinder", regex=True)
        # Explicit heating circuit label
        text_heating = valve_str.str.contains("heat|room|space", regex=True)
        valve_dhw_mask |= text_dhw
        valve_known |= text_dhw | text_heating
        # A purely numeric valve feed (Samsung Modbus register 89 / NASA 0x4067 mapped
        # directly): 1 = DHW, 0 = heating. Only when every reading is numeric, so a
        # text feed's back-filled leading zeros are not read as "heating".
        observed = valve_str[d["ValveMode"].notna()]
        numeric_feed = observed.str.fullmatch(r"[01](\.0+)?")
        if len(observed) and numeric_feed.all():
            num = pd.to_numeric(d["ValveMode"], errors="coerce")
            valve_dhw_mask |= num == 1
            valve_known |= num.isin([0, 1])

    # --- Status evidence (secondary) ----------------------------------------
    # Availability is tracked separately from value: a status channel that is
    # present and reporting "off" is information, not "no signal".
    dhw_status_mask = pd.Series(False, index=idx)
    status_known = pd.Series(False, index=idx)

    if "DHW_Status_Is_On" in d.columns:
        s = pd.to_numeric(d["DHW_Status_Is_On"], errors="coerce")
        status_known |= s.notna()
        dhw_status_mask |= s.fillna(0) > 0.5

    if "DHW_Active" in d.columns:
        s = pd.to_numeric(d["DHW_Active"], errors="coerce")
        status_known |= s.notna()
        dhw_status_mask |= s.fillna(0) > 0

    if "DHW_Status" in d.columns:
        status_str = d["DHW_Status"].astype(str).str.strip().str.lower()
        status_known |= d["DHW_Status"].notna() & ~status_str.isin(["nan", "none", "unknown", "unavailable", ""])
        dhw_status_mask |= status_str.str.contains(r"\bon\b|active", regex=True)

    # NOTE: DHW_Mode is intentionally *not* used here for DHW detection.
    # It is a control "aggressiveness" setting (Eco / Standard / Power / Force),
    # not a reliable indicator that a DHW run is actually in progress.

    # --- Combine valve + status into is_DHW ---------------------------------
    # Availability is evaluated per minute, independently from signal value.
    # Where the valve is known it defines the circuit; a known status must also
    # indicate an active DHW demand. Where the valve is unknown, status is the
    # fallback. This preserves explicit non-DHW valve readings (Heating / 0)
    # instead of treating an upload with no positive DHW valve values as if it
    # had no valve information at all.
    is_dhw_mask = (
        valve_known
        & valve_dhw_mask
        & (dhw_status_mask | ~status_known)
    ) | (
        ~valve_known
        & status_known
        & dhw_status_mask
    )

    d["is_DHW"] = is_dhw_mask.astype(bool)
    d["is_heating"] = (d["is_active"] == 1) & (~d["is_DHW"])

    _debug_engine_state(d, "after activity flags")

    # 4. HEAT + SPLITS + COP (single canonical implementation)
    d = _ensure_heat_and_cop(d, thresholds)
    _debug_engine_state(d, "after heat/COP")

    # 4b. Defrost episodes + split of heat taken back (defrost vs transient)
    d = _classify_heat_losses(d, thresholds)

    # 4c. Active cooling: flagged and removed from all heating figures
    d = _classify_cooling(d, thresholds)

    # 5. Zone Config Strings
    zone_cols = get_active_zone_columns(d)
    if zone_cols:
        # Data Type Fix: Ensure all zone columns are numeric
        for z in zone_cols:
            if not pd.api.types.is_numeric_dtype(d[z]):
                # Silent coercion of 'on'/'off'/boolish strings to 1/0
                d[z] = (
                    d[z]
                    .astype(str)
                    .str.lower()
                    .replace({"on": 1, "off": 0, "true": 1, "false": 0})
                )
            d[z] = pd.to_numeric(d[z], errors="coerce").fillna(0)

        d["Active_Zones_Count"] = d[zone_cols].sum(axis=1)
        z_map = {z: get_friendly_name(z, user_config) for z in zone_cols}

        def get_zone_str(row):
            active = [str(z_map.get(z, z)) for z in zone_cols if row[z] > 0]
            return " + ".join(active) if active else "None"

        d["Zone_Config"] = d.apply(get_zone_str, axis=1)
    else:
        d["Active_Zones_Count"] = 0
        d["Zone_Config"] = "None"

    # 6. Immersion
    # Signal priority: Immersion_Is_On (HA Modbus decode) > Immersion_Mode (mapped).
    # On the Samsung Gen 6 the "immersion heater mode" means the element is
    # PERMITTED, not necessarily drawing (e.g. a weekly schedule fires with the
    # tank already hot and the indoor unit stays at ~11 W). So when Indoor_Power
    # is available, the element only counts as on when it is actually drawing.
    immersion_signal = _immersion_signal(d)
    indoor = (
        pd.to_numeric(d["Indoor_Power"], errors="coerce").fillna(0)
        if "Indoor_Power" in d.columns else None
    )
    confirm_w = float(thresholds.get("immersion_confirm_power_w", 500))
    heuristic_w = float(thresholds.get("immersion_heuristic_power_w", 2500))
    if immersion_signal is not None and indoor is not None:
        d["Immersion_Active"] = immersion_signal & (indoor > confirm_w)
        d.attrs["immersion_source"] = "signal+power"
    elif immersion_signal is not None:
        d["Immersion_Active"] = immersion_signal
        d.attrs["immersion_source"] = "signal"
    elif indoor is not None:
        # No immersion signal: only a ~3 kW element exceeds this on the indoor unit
        d["Immersion_Active"] = indoor > heuristic_w
        d.attrs["immersion_source"] = "power_heuristic"
    else:
        d["Immersion_Active"] = False
        d.attrs["immersion_source"] = "none"
    # Metering: does the main Power channel already include the immersion
    # element? If so, daily totals must not add it a second time.
    d.attrs["power_includes_immersion"] = bool(thresholds.get("power_includes_immersion", False))

    if "Indoor_Power" in d.columns:
        d["Immersion_Power"] = np.where(
            d["Immersion_Active"], d["Indoor_Power"], 0
        )
    else:
        # Without Indoor_Power we cannot estimate immersion power reliably.
        # Keep the column but assume 0 W rather than a hard-coded 3 kW.
        d["Immersion_Power"] = 0.0

    # Indoor unit (controls + water pump) for the whole-system (SEPEMO H4)
    # figures: indoor power minus the immersion element. By mode only while
    # that mode runs; idle standby appears in the whole-day total only.
    if "Indoor_Power" in d.columns:
        indoor_net = (
            pd.to_numeric(d["Indoor_Power"], errors="coerce").fillna(0) - d["Immersion_Power"]
        ).clip(lower=0)
        d["Indoor_Net_Power"] = indoor_net
        d["Indoor_Heating_Power"] = np.where(_heating_heat_mask(d), indoor_net, 0.0)
        d["Indoor_DHW_Power"] = np.where(d["is_DHW"].astype(bool), indoor_net, 0.0)

    # 7. Cost (Tariff-aware)
    d["hour"] = d.index.hour

    # Retain legacy "is_night_rate" for any downstream logic/visuals
    d["is_night_rate"] = d["hour"].isin(NIGHT_HOURS)

    # Always use the tariff from the user profile; default to an empty list if absent.
    # The tariff engine handles this gracefully by applying a default flat rate.
    tariff_structure = (
        user_config.get("tariff_structure", [])
        if isinstance(user_config, dict)
        else []
    )

    try:
        d["Current_Rate"] = _compute_tariff_series(d.index, tariff_structure)
    except Exception:
        # Conservative fallback to previous behaviour
        if isinstance(tariff_structure, dict):
            rate_day = tariff_structure.get("day_rate", 0.35)
            rate_night = tariff_structure.get("night_rate", 0.15)
        else:
            rate_day = 0.35
            rate_night = 0.15

        d["Current_Rate"] = np.where(d["is_night_rate"], rate_night, rate_day)

    d["Cost_Inc"] = (d["Power"] / 1000.0 / 60.0) * d["Current_Rate"]
    # What each mode's electricity cost at the price of its own minutes (hot water often runs in a
    # different price band from heating, so a share of the day's cost would be wrong).
    for mode in ("Heating", "DHW"):
        if f"Power_{mode}" in d.columns:
            d[f"Cost_{mode}_Inc"] = (d[f"Power_{mode}"] / 1000.0 / 60.0) * d["Current_Rate"]

    # Final engine debug snapshot
    _debug_engine_state(d, "final engine df")
    # Removed automatic CSV write to disk.
    # Manual downloads via the Streamlit Data Debugger expander are now the intended method.

    return d


# --- GLOBAL STATS (CANONICAL) -----------------------------------------------
def compute_global_stats(df: pd.DataFrame) -> dict:
    """
    Canonical global stats for the whole dataset, based on the processed physics engine.

    - Uses `Heat` and `Power` from the fully-processed dataframe (after apply_gatekeepers),
      so all the existing logic (immersion removal, DHW vs heating, defrost protection)
      has already been applied.
    - If `is_active` is present, we only count periods where the heat pump is actually running
      (plus pump run-on minutes). This makes the COP comparable to per-run COP.
    - Energy is computed as W-minutes -> kWh, consistent with run and daily stats.
    - Heat is NET: heat taken back during defrost and start/stop is subtracted.

    Returns:
        {
            "total_heat_kwh": float,              # net
            "total_elec_kwh": float,
            "global_cop": float,                  # net heat / electricity
            "gross_heat_kwh": float,              # before losses
            "defrost_heat_loss_kwh": float,
            "transient_heat_loss_kwh": float,
            "defrost_count": int,
        }
    """
    empty = {
        "total_heat_kwh": 0.0,
        "total_elec_kwh": 0.0,
        "global_cop": 0.0,
        "gross_heat_kwh": 0.0,
        "defrost_heat_loss_kwh": 0.0,
        "transient_heat_loss_kwh": 0.0,
        "defrost_count": 0,
        "cooling_heat_removed_kwh": 0.0,
        "cooling_elec_kwh": 0.0,
        "cooling_eer": 0.0,
    }
    if df is None or df.empty:
        return empty

    defrost_count = int(df["Defrost_Start"].sum()) if "Defrost_Start" in df.columns else 0
    included = pd.Series(True, index=df.index)

    # Cooling is reported separately and excluded from heating efficiency
    cooling = {"cooling_heat_removed_kwh": 0.0, "cooling_elec_kwh": 0.0, "cooling_eer": 0.0}
    if "is_cooling" in df.columns and df["is_cooling"].any():
        cooling_mask = df["is_cooling"].fillna(False).astype(bool)
        c = df.loc[cooling_mask]
        removed = float(c["Cooling_Heat_Removed"].sum() / 60000.0)
        c_elec = float(pd.to_numeric(c["Power"], errors="coerce").fillna(0).sum() / 60000.0)
        cooling = {
            "cooling_heat_removed_kwh": removed,
            "cooling_elec_kwh": c_elec,
            "cooling_eer": float(safe_div(removed, c_elec)),
        }
        included &= ~cooling_mask

    # Restrict to HP actually running (plus pump run-on, which carries heat
    # the run produced) if we have the engine flag
    if "is_active" in df.columns:
        running = df["is_active"].eq(1)
        if "is_run_on" in df.columns:
            running |= df["is_run_on"].fillna(False).astype(bool)
        included &= running

    # Normalise series
    d = df.loc[included]
    power = pd.to_numeric(d.get("Power", 0), errors="coerce").fillna(0.0)
    heat = pd.to_numeric(d.get("Heat", 0), errors="coerce").fillna(0.0)

    # W-minutes → kWh (60 minutes/hour * 1000 W/kW)
    total_elec_kwh = power.sum() / 60000.0
    total_heat_kwh = heat.sum() / 60000.0
    global_cop = safe_div(total_heat_kwh, total_elec_kwh)

    def _kwh(col: str) -> float:
        if col not in d.columns:
            return 0.0
        return float(pd.to_numeric(d[col], errors="coerce").fillna(0.0).sum() / 60000.0)

    return {
        "total_heat_kwh": float(total_heat_kwh),
        "total_elec_kwh": float(total_elec_kwh),
        "global_cop": float(global_cop),
        "gross_heat_kwh": _kwh("Heat_Clean"),
        "defrost_heat_loss_kwh": _kwh("Heat_Loss_Defrost"),
        "transient_heat_loss_kwh": _kwh("Heat_Loss_Transient"),
        "defrost_count": defrost_count,
        **cooling,
    }


def merge_global_stats(parts: list) -> dict:
    """
    compute_global_stats of consecutive, non-overlapping parts of one dataset
    combined: every total is a minute sum, so parts add up exactly; the ratios
    (COP, EER) are recomputed from the totals, never averaged.
    """
    parts = [p for p in parts if p]
    merged = compute_global_stats(None)
    for p in parts:
        for key, value in p.items():
            if key not in ("global_cop", "cooling_eer"):
                merged[key] = merged.get(key, 0) + value
    merged["global_cop"] = float(safe_div(merged["total_heat_kwh"], merged["total_elec_kwh"]))
    merged["cooling_eer"] = float(safe_div(merged["cooling_heat_removed_kwh"], merged["cooling_elec_kwh"]))
    merged["defrost_count"] = int(merged["defrost_count"])
    return merged


# --- RUN DETECTION -----------------------------------------------------------
def detect_runs(df: pd.DataFrame, user_config: dict | None = None) -> list[dict]:
    """
    Group contiguous active periods into "runs" and derive per-run metrics
    (heat, electricity, COP, hydraulics, zone configuration, room deltas).
    """
    # Safety check on user_config type
    if isinstance(user_config, dict):
        rooms_per_zone = user_config.get("rooms_per_zone", {})
    else:
        rooms_per_zone = {}

    df = df.copy()

    threshold_defaults = THRESHOLDS if isinstance(THRESHOLDS, dict) else {}
    user_thresholds = user_config.get("thresholds", {}) if isinstance(user_config, dict) else {}
    thresholds = {**threshold_defaults, **user_thresholds}

    min_run_duration = thresholds.get("minimum_run_duration_min", 5)
    min_unzoned_run_duration = thresholds.get(
        "min_heating_run_minutes_with_no_zones", max(min_run_duration, 8)
    )
    min_unzoned_heat_kwh = thresholds.get(
        "min_heating_run_heat_kwh_with_no_zones", 0.25
    )

    is_cooling_col = (
        df["is_cooling"].astype(bool) if "is_cooling" in df.columns
        else pd.Series(False, index=df.index)
    )
    df["run_change"] = (
        (df["is_active"].diff().ne(0))
        | (df["is_DHW"].ne(df["is_DHW"].shift()))
        | (is_cooling_col.ne(is_cooling_col.shift()))
    )
    df["run_id"] = df["run_change"].cumsum()

    runs: list[dict] = []
    active_groups = df[df["is_active"] == 1].groupby("run_id")

    zone_cols = get_active_zone_columns(df)
    room_cols = get_room_columns(df)

    for run_id, group in active_groups:
        if len(group) < min_run_duration:
            continue

        # Majority-based DHW classification instead of "any is_DHW"
        is_dhw_series = group.get("is_DHW", pd.Series(False, index=group.index)).astype(bool)
        p_dhw = float(is_dhw_series.mean()) if len(group) > 0 else 0.0

        # Extra context from valve/status if present
        p_valve_dhw = 0.0
        if "Valve_Is_DHW" in group.columns:
            v = pd.to_numeric(group["Valve_Is_DHW"], errors="coerce").fillna(0)
            p_valve_dhw = float((v > 0.5).mean())
        elif "ValveMode" in group.columns:
            vs = group["ValveMode"].astype(str).str.lower()
            p_valve_dhw = float(vs.str.contains("dhw|hot water|hot_water", regex=True).mean())

        p_status_on = 0.0
        if "DHW_Status_Is_On" in group.columns:
            s = pd.to_numeric(group["DHW_Status_Is_On"], errors="coerce").fillna(0)
            p_status_on = float((s > 0.5).mean())
        elif "DHW_Active" in group.columns:
            s = pd.to_numeric(group["DHW_Active"], errors="coerce").fillna(0)
            p_status_on = float((s > 0).mean())
        elif "DHW_Status" in group.columns:
            ss = group["DHW_Status"].astype(str).str.lower()
            p_status_on = float(ss.str.contains("on|active", regex=True).mean())

        run_type = "Heating"
        p_cool = float(group["is_cooling"].astype(bool).mean()) if "is_cooling" in group.columns else 0.0
        # Primary rule: majority of minutes flagged as DHW by engine flags
        if p_dhw >= 0.6:
            run_type = "DHW"
        elif p_cool >= 0.6:
            run_type = "Cooling"
        else:
            # Secondary rule: strong valve+status evidence even if is_DHW is sparse
            if (p_valve_dhw >= 0.6) and (p_status_on >= 0.6):
                run_type = "DHW"

        # Averages
        avg_outdoor = (
            group["OutdoorTemp"].mean() if "OutdoorTemp" in group.columns else 0
        )
        avg_flow = group["FlowTemp"].mean() if "FlowTemp" in group.columns else 0
        avg_flow_rate = (
            group["FlowRate"].mean() if "FlowRate" in group.columns else 0
        )

        # Metrics
        heat_kwh = group["Heat"].sum() / 60000.0  # NET (defrost/transient losses subtracted)

        # Heat accounting breakdown (gross = net + losses)
        def _grp_kwh(col: str) -> float:
            return float(group[col].sum() / 60000.0) if col in group.columns else 0.0

        heat_gross_kwh = _grp_kwh("Heat_Clean")
        defrost_heat_loss_kwh = _grp_kwh("Heat_Loss_Defrost")
        transient_heat_loss_kwh = _grp_kwh("Heat_Loss_Transient")
        defrost_count = int(group["Defrost_Start"].sum()) if "Defrost_Start" in group.columns else 0
        defrost_mins = int(group["is_defrost"].sum()) if "is_defrost" in group.columns else 0
        elec_kwh = group["Power"].sum() / 60000.0
        cop = safe_div(heat_kwh, elec_kwh)

        # Cooling runs: heat_kwh = heat REMOVED and run_cop = EER
        cooling_heat_removed_kwh = _grp_kwh("Cooling_Heat_Removed")
        eer = None
        if run_type == "Cooling":
            heat_kwh = cooling_heat_removed_kwh
            eer = safe_div(heat_kwh, elec_kwh)
            cop = eer

        # Friendly Zones
        active_zones_list = []
        if run_type in ("Heating", "Cooling") and zone_cols:
            for z in zone_cols:
                if (group[z].sum() / len(group)) > 0.1:
                    active_zones_list.append(z)

        friendly_zones = [
            str(get_friendly_name(z, user_config)) for z in active_zones_list
        ]
        dominant_zones_str = (
            " + ".join(friendly_zones)
            if friendly_zones
            else ("DHW" if run_type == "DHW" else "None")
        )

        # Friendly Rooms
        relevant_rooms: list[str] = []
        if run_type in ("Heating", "Cooling") and rooms_per_zone and active_zones_list:
            for z in active_zones_list:
                relevant_rooms.extend(rooms_per_zone.get(z, []))
        relevant_rooms = list(set(relevant_rooms))

        duration_mins = len(group)

        # Heating during DHW (a zone pump running while the heat pump heats the cylinder). Only
        # zone/pump signals can show it: on real data an Indoor_Power proxy flagged 94 DHW
        # runs of which 4 were real (90 false alarms, 39 of 43 missed), and no other signal
        # (outdoor temperature, ΔT, flow, tank energy balance, room temperatures) reached a usable
        # AUC (<= 0.73 at 6 % prevalence). Without zone signals the fields stay None.
        heating_during_dhw_pct = None
        heating_during_dhw_detected = None
        heating_during_dhw_detection_source = "none"
        if run_type == "DHW":
            if zone_cols:
                zone_on_pct = (
                    (group[zone_cols].fillna(0) > 0).any(axis=1).mean()
                    if len(group) > 0
                    else 0.0
                )
                heating_during_dhw_pct = float(zone_on_pct)
                detection_pct_threshold = thresholds.get("heating_during_dhw_detection_pct", 0.15)
                heating_during_dhw_detected = zone_on_pct >= detection_pct_threshold
                heating_during_dhw_detection_source = "zones"

        # DHW Temperature Profile (DHW runs only)
        dhw_temp_start = None
        dhw_temp_end = None
        dhw_rise = None
        if run_type == "DHW" and "DHW_Temp" in group.columns:
            dhw_series = pd.to_numeric(group["DHW_Temp"], errors="coerce").dropna()
            if len(dhw_series) > 0:
                dhw_temp_start = float(dhw_series.iloc[0])
                dhw_temp_end = float(dhw_series.iloc[-1])
                dhw_rise = dhw_temp_end - dhw_temp_start

        # DHW_Mode tracking (capture modal value during run)
        dhw_mode_value = None
        if run_type == "DHW" and "DHW_Mode" in group.columns:
            # Get most common DHW_Mode value during run (mode of the series)
            mode_series = group["DHW_Mode"].dropna()
            if len(mode_series) > 0:
                dhw_mode_value = mode_series.mode().iloc[0] if len(mode_series.mode()) > 0 else None

        # Immersion Detection (boolean flag)
        immersion_kwh = group["Immersion_Power"].sum() / 60000.0
        immersion_mins = int(group["Immersion_Active"].sum())
        # >1 Wh of element draw; or, with a signal but no Indoor_Power to size it,
        # any minute flagged on
        immersion_detected = immersion_kwh > 0.001 or (
            "Indoor_Power" not in group.columns and immersion_mins > 0
        )

        # Return Temperature Stats
        avg_return_temp = None
        min_return_temp = None
        return_temp_range = None
        if "ReturnTemp" in group.columns:
            return_series = pd.to_numeric(group["ReturnTemp"], errors="coerce").dropna()
            if len(return_series) > 0:
                avg_return_temp = float(return_series.mean())
                min_return_temp = float(return_series.min())
                return_temp_range = float(return_series.max() - return_series.min())

        # Compressor Frequency Stats (enhanced)
        min_freq = None
        max_freq = None
        freq_std_dev = None
        if "Freq" in group.columns:
            freq_series = pd.to_numeric(group["Freq"], errors="coerce").dropna()
            if len(freq_series) > 0:
                min_freq = float(freq_series.min())
                max_freq = float(freq_series.max())
                freq_std_dev = float(freq_series.std())

        # ================================================================
        # PHASE 1 QUICK WIN METRICS (5-star additions)
        # ================================================================

        # 1. Short-Cycle Flag
        short_cycle_threshold = thresholds.get("short_cycle_min", 20)
        is_short_cycle = duration_mins < short_cycle_threshold

        # 2. Outdoor Temperature Change During Run
        outdoor_temp_change = None
        outdoor_temp_min = None
        outdoor_temp_max = None
        if "OutdoorTemp" in group.columns:
            outdoor_series = pd.to_numeric(group["OutdoorTemp"], errors="coerce").dropna()
            if len(outdoor_series) > 0:
                outdoor_temp_min = float(outdoor_series.min())
                outdoor_temp_max = float(outdoor_series.max())
                outdoor_temp_change = outdoor_temp_max - outdoor_temp_min

        # 3. Cost per kWh Heat (normalized efficiency metric)
        cost_per_kwh_heat = None
        if "Cost_Inc" in group.columns and heat_kwh > 0:
            run_cost = group["Cost_Inc"].sum()
            cost_per_kwh_heat = safe_div(run_cost, heat_kwh)

        # 4. DHW Stratification Range (mixing/quality indicator)
        dhw_stratification_range = None
        if run_type == "DHW" and "DHW_Temp" in group.columns:
            dhw_series = pd.to_numeric(group["DHW_Temp"], errors="coerce").dropna()
            if len(dhw_series) > 0:
                dhw_stratification_range = float(dhw_series.max() - dhw_series.min())

        # Room Deltas
        room_deltas: dict[str, float] = {}
        for r in room_cols:
            series = group[r].dropna()
            if len(series) > 0:
                start_t = series.iloc[0]
                end_t = series.iloc[-1]
                friendly_r = get_friendly_name(r, user_config)
                room_deltas[friendly_r] = round(end_t - start_t, 2)

        # Filter out zone-less, tiny heating blips (often noise from Grafana ingest)
        if (
            run_type == "Heating"
            and zone_cols
            and not active_zones_list
            and (
                duration_mins < min_unzoned_run_duration
                or heat_kwh < min_unzoned_heat_kwh
            )
        ):
            continue

        runs.append(
            {
                "id": int(run_id),
                "start": group.index[0],
                "end": group.index[-1],
                "duration_mins": duration_mins,
                "run_type": run_type,
                "avg_outdoor": round(avg_outdoor, 1),
                "avg_flow_temp": round(avg_flow, 1),
                "avg_dt": round(group.get("DeltaT", pd.Series(0)).mean(), 1),
                "avg_flow_rate": round(avg_flow_rate, 1),
                "run_cop": round(cop, 2),
                "heat_kwh": heat_kwh,
                "cooling_heat_removed_kwh": cooling_heat_removed_kwh,
                "eer": (round(eer, 2) if eer is not None else None),
                "electricity_kwh": elec_kwh,
                "heat_gross_kwh": heat_gross_kwh,
                "defrost_count": defrost_count,
                "defrost_mins": defrost_mins,
                "defrost_heat_loss_kwh": defrost_heat_loss_kwh,
                "transient_heat_loss_kwh": transient_heat_loss_kwh,
                "active_zones": dominant_zones_str,
                "dominant_zones": dominant_zones_str,
                "room_deltas": room_deltas,
                "relevant_rooms": relevant_rooms,
                "immersion_kwh": immersion_kwh,
                "immersion_mins": immersion_mins,
                "immersion_detected": bool(immersion_detected),
                "heating_during_dhw_pct": heating_during_dhw_pct,
                "heating_during_dhw_detected": (
                    bool(heating_during_dhw_detected)
                    if heating_during_dhw_detected is not None
                    else None
                ),
                "heating_during_dhw_detection_source": heating_during_dhw_detection_source,
                # DHW Temperature Profile
                "dhw_temp_start": dhw_temp_start,
                "dhw_temp_end": dhw_temp_end,
                "dhw_rise": dhw_rise,
                # DHW Mode
                "dhw_mode": dhw_mode_value,
                # Return Temperature Stats
                "avg_return_temp": avg_return_temp,
                "min_return_temp": min_return_temp,
                "return_temp_range": return_temp_range,
                # Compressor Frequency Stats (enhanced)
                "min_freq": min_freq,
                "max_freq": max_freq,
                "freq_std_dev": freq_std_dev,
                # Phase 1 Quick Win Metrics
                "is_short_cycle": bool(is_short_cycle),
                "outdoor_temp_change": outdoor_temp_change,
                "outdoor_temp_min": outdoor_temp_min,
                "outdoor_temp_max": outdoor_temp_max,
                "cost_per_kwh_heat": cost_per_kwh_heat,
                "dhw_stratification_range": dhw_stratification_range,
            }
        )

    return runs



# --- DAILY STATS -------------------------------------------------------------


def get_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate minute-level data into daily metrics.

    Now robust to being called on any dataframe with:
      - a DatetimeIndex, and
      - ideally a 'Power' column (for DQ_Score).

    If some of the internal engine columns are missing (e.g. Power_Heating),
    they are simply omitted from aggregation and the derived kWh fields fall
    back to 0 via .get(...).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("get_daily_stats expects a DatetimeIndex on the input.")

    # Base daily index + row counts (used for DQ_Score)
    if "Power" in df.columns:
        base_counts = df["Power"].resample("D").count().rename("row_count")
    else:
        base_counts = df.resample("D").size().rename("row_count")

    daily = base_counts.to_frame()

    # Core engine aggregates (only include if columns exist)
    base_agg = {
        "Power_Heating": ["sum"],
        "Power_DHW": ["sum"],
        "Heat_Heating": ["sum"],
        "Heat_DHW": ["sum"],
        "Cost_Inc": ["sum"],
        "Cost_Heating_Inc": ["sum"],
        "Cost_DHW_Inc": ["sum"],
        "is_active": ["sum"],
        "Immersion_Power": ["sum"],
        # Indoor unit, net of immersion (whole-system / H4 figures)
        "Indoor_Net_Power": ["sum"],
        "Indoor_Heating_Power": ["sum"],
        "Indoor_DHW_Power": ["sum"],
        "is_DHW": ["sum"],
        "is_heating": ["sum"],
        # Heat accounting (net = gross - defrost loss - transient loss)
        "Heat_Clean": ["sum"],
        "Heat_Loss": ["sum"],
        "Heat_Loss_Defrost": ["sum"],
        "Heat_Loss_Transient": ["sum"],
        "is_defrost": ["sum"],
        "Defrost_Start": ["sum"],
        # Cooling (reported separately, excluded from heating totals)
        "is_cooling": ["sum"],
        "Power_Cooling": ["sum"],
        "Cooling_Heat_Removed": ["sum"],
        # HA retention boundary: minutes before full-resolution data (#22)
        "Before_Complete": ["sum"],
    }

    agg_dict: dict[str, list[str]] = {}
    for col, funcs in base_agg.items():
        if col in df.columns:
            agg_dict[col] = funcs

    # Additional columns: summary stats per column
    exclude = list(base_agg.keys()) + [
        "last_changed",
        "entity_id",
        "state",
        "Zone_Config",
    ]

    for col in df.columns:
        if col in exclude:
            continue
        if col.endswith("_From_Secondary"):
            agg_dict[col] = ["sum"]  # minutes filled from the secondary source
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_dict[col] = ["mean", "min", "max", "count"]
        else:
            agg_dict[col] = ["count"]

    if agg_dict:
        daily_aggs = df.resample("D").agg(agg_dict)
        daily_aggs.columns = [
            "_".join(col).strip() if isinstance(col, tuple) else col
            for col in daily_aggs.columns
        ]
        daily = daily.join(daily_aggs, how="left")

    # Normalise core column names (some may not exist; rename ignores missing)
    for core_col in [
        "Power_Heating",
        "Power_DHW",
        "Heat_Heating",
        "Heat_DHW",
    ]:
        if core_col in daily.columns and f"{core_col}_sum" not in daily.columns:
            daily = daily.rename(columns={core_col: f"{core_col}_sum"})

    rename_map = {
        "Power_Heating_sum": "Electricity_Heating_Wmin",
        "Power_DHW_sum": "Electricity_DHW_Wmin",
        "Heat_Heating_sum": "Heat_Heating_Wmin",
        "Heat_DHW_sum": "Heat_DHW_Wmin",
        "Cost_Inc_sum": "Daily_Cost_Euro",
        "Cost_Heating_Inc_sum": "Heating_Cost",
        "Cost_DHW_Inc_sum": "DHW_Cost",
        "is_active_sum": "Active_Mins",
        "is_DHW_sum": "DHW_Mins",
        "is_heating_sum": "Heating_Mins",
        "Immersion_Power_sum": "Immersion_Wh",
        "OutdoorTemp_mean": "Outdoor_Avg",
        "OutdoorTemp_min": "Outdoor_Min",
        "OutdoorTemp_max": "Outdoor_Max",
        # Environmental averages for long-term trend charts
        "Wind_Speed_mean": "Wind_Avg",
        "Outdoor_Humidity_mean": "Humidity_Avg",
        "Solar_Rad_mean": "Solar_Avg",
    }
    daily = daily.rename(columns=rename_map)
    # Secondary weather sources are merged into the primary columns by
    # apply_weather_fallbacks, so the averages above already include them.

    # kWh conversions (safe even if *_Wmin columns are missing)
    daily["Electricity_Heating_kWh"] = (
        daily.get("Electricity_Heating_Wmin", 0) / 60000.0
    )
    daily["Electricity_DHW_kWh"] = (
        daily.get("Electricity_DHW_Wmin", 0) / 60000.0
    )
    daily["Heat_Heating_kWh"] = daily.get("Heat_Heating_Wmin", 0) / 60000.0
    daily["Heat_DHW_kWh"] = daily.get("Heat_DHW_Wmin", 0) / 60000.0
    daily["Immersion_kWh"] = daily.get("Immersion_Wh", 0) / 60000.0
    if "Indoor_Net_Power_sum" in daily.columns:
        daily["Indoor_Electricity_kWh"] = daily["Indoor_Net_Power_sum"] / 60000.0
        daily["Indoor_Heating_kWh"] = daily.get("Indoor_Heating_Power_sum", 0) / 60000.0
        daily["Indoor_DHW_kWh"] = daily.get("Indoor_DHW_Power_sum", 0) / 60000.0
        daily = daily.drop(columns=[c for c in (
            "Indoor_Net_Power_sum", "Indoor_Heating_Power_sum", "Indoor_DHW_Power_sum"
        ) if c in daily.columns])

    # Immersion is added unless the main Power meter already includes it
    # (profile physics_thresholds["power_includes_immersion"], set in Setup).
    immersion_in_power = bool(df.attrs.get("power_includes_immersion", False))
    daily["Total_Electricity_kWh"] = (
        daily["Electricity_Heating_kWh"]
        + daily["Electricity_DHW_kWh"]
        + (0.0 if immersion_in_power else daily["Immersion_kWh"])
    )
    # NET heat: Heat_Heating/Heat_DHW are signed, so defrost and start/stop
    # losses are already subtracted (EN 14511 / EN 14825 convention).
    daily["Total_Heat_kWh"] = (
        daily["Heat_Heating_kWh"] + daily["Heat_DHW_kWh"]
    )

    # Heat accounting breakdown: Gross = Total (net) + Defrost + Transient.
    # (Named Defrost_Events, not Defrost_Count: view_quality strips "_count"/"_Count"
    # suffixes, so that would collide with the Defrost sensor's Defrost_count.)
    daily["Gross_Heat_kWh"] = daily.get("Heat_Clean_sum", 0) / 60000.0
    daily["Defrost_Heat_Loss_kWh"] = daily.get("Heat_Loss_Defrost_sum", 0) / 60000.0
    daily["Transient_Heat_Loss_kWh"] = daily.get("Heat_Loss_Transient_sum", 0) / 60000.0
    daily["Defrost_Events"] = daily.get("Defrost_Start_sum", 0)
    daily["Defrost_Mins"] = daily.get("is_defrost_sum", 0)
    # Cooling: separate energy + EER; not part of Total_Heat/Total_Electricity
    daily["Cooling_Mins"] = daily.get("is_cooling_sum", 0)
    daily["Electricity_Cooling_kWh"] = daily.get("Power_Cooling_sum", 0) / 60000.0
    daily["Cooling_Heat_Removed_kWh"] = daily.get("Cooling_Heat_Removed_sum", 0) / 60000.0
    daily["Cooling_EER"] = safe_div(daily["Cooling_Heat_Removed_kWh"], daily["Electricity_Cooling_kWh"])
    if "Before_Complete_sum" in daily.columns:
        daily["Incomplete_Mins"] = daily["Before_Complete_sum"]
    daily = daily.drop(
        columns=[
            c for c in (
                "Heat_Clean_sum", "Heat_Loss_sum", "Heat_Loss_Defrost_sum",
                "Heat_Loss_Transient_sum", "Defrost_Start_sum", "is_defrost_sum",
                "is_cooling_sum", "Power_Cooling_sum", "Cooling_Heat_Removed_sum",
                "Before_Complete_sum",
            )
            if c in daily.columns
        ]
    )

    # ---- Efficiency (monitoring convention: a day or period is a COP) ----
    # Heat-pump electricity: immersion is reported separately (Immersion_kWh)
    # and kept out of every COP. If the main Power meter includes the element,
    # its energy is taken back out of the DHW electricity.
    immersion_metered = daily["Immersion_kWh"] if immersion_in_power else 0.0
    daily["Electricity_DHW_HP_kWh"] = (daily["Electricity_DHW_kWh"] - immersion_metered).clip(lower=0)
    daily["HP_Electricity_kWh"] = daily["Electricity_Heating_kWh"] + daily["Electricity_DHW_HP_kWh"]

    def _cop(heat, elec):
        # Empty (not 0) when there is essentially no electricity for the mode
        return (heat / elec).where(elec > 0.01)

    # H2: outdoor unit (the Power meter)
    daily["Daily_COP"] = _cop(daily["Total_Heat_kWh"], daily["HP_Electricity_kWh"])
    daily["Heating_COP"] = _cop(daily["Heat_Heating_kWh"], daily["Electricity_Heating_kWh"])
    daily["DHW_COP"] = _cop(daily["Heat_DHW_kWh"], daily["Electricity_DHW_HP_kWh"])
    # H4: plus the indoor unit (controls + water pump). The whole-day figure
    # includes idle standby; the mode figures only indoor power while that runs.
    if "Indoor_Electricity_kWh" in daily.columns:
        daily["Daily_COP_H4"] = _cop(
            daily["Total_Heat_kWh"], daily["HP_Electricity_kWh"] + daily["Indoor_Electricity_kWh"]
        )
        daily["Heating_COP_H4"] = _cop(
            daily["Heat_Heating_kWh"], daily["Electricity_Heating_kWh"] + daily["Indoor_Heating_kWh"]
        )
        daily["DHW_COP_H4"] = _cop(
            daily["Heat_DHW_kWh"], daily["Electricity_DHW_HP_kWh"] + daily["Indoor_DHW_kWh"]
        )
    # Former name, kept for AI prompts and saved exports: now the heat-pump-only
    # daily COP (before 2026-09-21 it added immersion electricity).
    daily["Global_SCOP"] = safe_div(daily["Total_Heat_kWh"], daily["HP_Electricity_kWh"])

    # Data Quality Score / Tier
    if "row_count" in daily.columns:
        daily["DQ_Score"] = (
            (daily["row_count"] / 1440.0) * 100.0
        ).clip(0, 100)
        # Minutes with a Power reading: the Data Quality view's per-day
        # denominator (handles partial first/last days).
        daily = daily.rename(columns={"row_count": "Recorded_Minutes"})
    else:
        daily["DQ_Score"] = 0

    # Windowed availability counts for the Data Quality view. Sensors that are
    # only expected while the unit runs (heating_active) or during hot water
    # (dhw_active) are counted inside that window, so the count and its
    # denominator (Active_Mins / DHW_Mins) cover the same minutes.
    dq_windows = {"heating_active": "is_active", "dhw_active": "is_DHW"}
    for col in df.columns:
        if col.endswith("_From_Secondary"):
            # Weather primary filled from its secondary source: score the
            # primary sensor on its own readings only.
            primary = col[: -len("_From_Secondary")]
            if primary in df.columns:
                own = df[primary].notna() & ~df[col].astype(bool)
                daily[f"DQ_{primary}_Count"] = own.resample("D").sum()
            continue
        mode = SENSOR_EXPECTATION_MODE.get(col)
        if mode in UNSCORED_DQ_MODES:
            # On-change / event sensors: count state changes per day. Reporting
            # frequency says nothing about their health (silence = unchanged).
            s = df[col]
            prev = s.shift()
            changed = s.notna() & prev.notna() & s.ne(prev)
            daily[f"DQ_{col}_Count"] = changed.resample("D").sum()
            continue
        flag = dq_windows.get(mode)
        if flag and flag in df.columns:
            in_window = df[flag].astype(bool)
            daily[f"DQ_{col}_Count"] = (
                (df[col].notna() & in_window).resample("D").sum()
            )

    # Data Quality tiers (v1.30.2). DQ_Score is Power coverage. Heat input
    # coverage follows the actual heat-calculation path over active minutes.
    #   Gold   = Power coverage >= 90 and heat-input coverage >= 90
    #   Silver = Power >= 90 but heat inputs are incomplete/unavailable
    #   Bronze = Power coverage below 90
    active = df.get("is_active", pd.Series(False, index=df.index)).astype(bool)
    active_mins = active.resample("D").sum().reindex(daily.index, fill_value=0)
    heat_source = df.attrs.get("heat_source")
    if heat_source is None:
        if "Heat" in df.columns:
            heat_source = "native"
        elif {"FlowTemp", "ReturnTemp", "FlowRate"} <= set(df.columns):
            heat_source = "derived_hydraulic"
        else:
            heat_source = "unavailable"
    required_heat_inputs = (
        ["Heat"] if heat_source == "native"
        else ["FlowTemp", "ReturnTemp", "FlowRate"]
        if heat_source == "derived_hydraulic"
        else []
    )
    input_coverage = []
    for col in required_heat_inputs:
        if col not in df.columns:
            input_coverage.append(pd.Series(0.0, index=daily.index))
            continue
        available = df[col].notna()
        if col == "Heat" and "Heat_Sensor_Available" in df.columns:
            # The engine zero-fills Heat; use availability recorded before that.
            available = df["Heat_Sensor_Available"].fillna(False).astype(bool)
        present = (available & active).resample("D").sum().reindex(daily.index, fill_value=0)
        input_coverage.append(safe_div(present, active_mins, default=np.nan) * 100.0)
    if input_coverage:
        daily["Heat_Input_Coverage"] = (
            pd.concat(input_coverage, axis=1).min(axis=1, skipna=True).clip(0, 100)
        )
        heat_input_ok = daily["Heat_Input_Coverage"].fillna(100.0) >= 90
    else:
        daily["Heat_Input_Coverage"] = np.nan
        heat_input_ok = pd.Series(False, index=daily.index)
    daily.attrs["heat_source"] = heat_source
    power_ok = daily["DQ_Score"] >= 90
    daily["DQ_Tier"] = np.select(
        [power_ok & heat_input_ok, power_ok],
        ["Tier 1 (Gold)", "Tier 2 (Silver)"],
        default="Tier 3 (Bronze)",
    )

    return daily


# --- DIAGNOSTIC DAILY METRICS (AI prompt contract) ---------------------------


def resolve_thresholds(user_config=None) -> dict:
    """config.THRESHOLDS with the profile's "thresholds" overrides applied."""
    user = user_config.get("thresholds", {}) if isinstance(user_config, dict) else {}
    return {**(THRESHOLDS if isinstance(THRESHOLDS, dict) else {}), **(user or {})}


def target_flow_series(df: pd.DataFrame, user_config=None):
    """Flow-temperature target per minute and where it came from.

    A mapped Target_Flow sensor wins. Otherwise the profile weather curve
    (config.resolve_weather_curve) is applied to OutdoorTemp. Returns
    (series_or_None, source) with source "sensor", "profile_curve" or
    "unavailable".
    """
    if "Target_Flow" in df.columns:
        target = pd.to_numeric(df["Target_Flow"], errors="coerce")
        if target.notna().any():
            return target, "sensor"
    curve = resolve_weather_curve(user_config)
    if not curve.get("enabled", True) or "OutdoorTemp" not in df.columns:
        return None, "unavailable"
    span = curve["mild_outdoor_c"] - curve["design_outdoor_c"]
    if span == 0:
        return None, "unavailable"
    slope = (curve["mild_flow_c"] - curve["design_flow_c"]) / span
    outdoor = pd.to_numeric(df["OutdoorTemp"], errors="coerce")
    target = curve["design_flow_c"] + slope * (outdoor - curve["design_outdoor_c"])
    return target.clip(curve["min_flow_c"], curve["max_flow_c"]), "profile_curve"


def flow_limit_analysis(df: pd.DataFrame, user_config=None):
    """Return counted, valid-input and eligible-heating masks plus target source."""
    target, source = target_flow_series(df, user_config)
    eligible = df.get("is_heating", pd.Series(False, index=df.index)).astype(bool)
    if "is_defrost" in df.columns:
        eligible &= ~df["is_defrost"].astype(bool)
    if target is None or "FlowTemp" not in df.columns or "is_heating" not in df.columns:
        return None, pd.Series(False, index=df.index), eligible, source
    th = resolve_thresholds(user_config)
    tolerance = float(th.get("flow_limit_tolerance", 2.0))
    min_block = int(th.get("flow_limit_min_duration", 15))
    flow = pd.to_numeric(df["FlowTemp"], errors="coerce")
    valid = eligible & flow.notna() & target.notna()
    limited = valid & target.gt(flow + tolerance)
    block_id = limited.ne(limited.shift(fill_value=False)).cumsum()
    block_len = limited.groupby(block_id).transform("size")
    return limited & block_len.ge(min_block), valid, eligible, source


def flow_limited_minutes(df: pd.DataFrame, user_config=None):
    """Minutes counted as flow-limited, and the flow-target source.

    A heating minute (defrost excluded: flow temperature drops then, which is
    not a flow limit) is limited when the target exceeds FlowTemp by more than
    flow_limit_tolerance; it counts only inside a stretch of at least
    flow_limit_min_duration consecutive limited minutes. Returns
    (bool Series or None, source).
    """
    counted, _valid, _eligible, source = flow_limit_analysis(df, user_config)
    return counted, source


def add_diagnostic_daily_metrics(
    daily: pd.DataFrame,
    df: pd.DataFrame,
    runs: list | None,
    user_config=None,
    run_window_start=None,
    run_window_end=None,
) -> pd.DataFrame:
    """Daily cycling, DHW, tariff and flow-limit metrics named in the AI prompt.

    Restores the metrics of the original THERM design (initial commit),
    using the profile thresholds:

    - Starts / Starts_Heating / Starts_DHW: detected runs by start day.
    - Short_Cycles_Count: heating runs shorter than short_cycle_min;
      Very_Short_Cycles_Count: shorter than very_short_cycle_min.
    - Cycling_Severity_Index = Short_Cycles_Count / Starts_Heating
      (NaN with no heating starts). Flagged above short_cycling_ratio_high.
    - Median_Run_Time_Mins: median heating run length.
    - DHW_SCOP = Heat_DHW_kWh / Electricity_DHW_kWh (NaN below 0.01 kWh).
    - Night_Share_Of_Total_HP_Elec: share of metered heat-pump electricity
      used at the cheapest tariff rate in force that day (NaN for a flat
      tariff day, where the share means nothing).
    - Effective_Avg_Tariff = Daily_Cost_Euro / metered kWh (the same Power
      the cost was computed on).
    - Virtual_FTlim_Time_Mins / Virtual_FTlim_Events: heating minutes
      (excluding defrost) where the flow target exceeds FlowTemp by more than
      flow_limit_tolerance, counted only in stretches of at least
      flow_limit_min_duration minutes. Target from target_flow_series().
    """
    if daily is None or len(daily.index) == 0:
        return daily
    out = daily.copy()
    th = resolve_thresholds(user_config)
    short_min = float(th.get("short_cycle_min", 20))
    very_short_min = float(th.get("very_short_cycle_min", 10))

    # Minutes for which run detection was possible. This makes partial first,
    # last and HA retention-boundary days explicit without discarding their
    # valid runs.
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index):
        window_start = pd.Timestamp(run_window_start) if run_window_start is not None else df.index.min()
        window_end = pd.Timestamp(run_window_end) if run_window_end is not None else df.index.max()
        tz = df.index.tz
        if tz is not None:
            window_start = window_start.tz_localize(tz) if window_start.tzinfo is None else window_start.tz_convert(tz)
            window_end = window_end.tz_localize(tz) if window_end.tzinfo is None else window_end.tz_convert(tz)
        elif window_start.tzinfo is not None:
            window_start = window_start.tz_localize(None)
            window_end = window_end.tz_localize(None)
        detection_mask = pd.Series(
            (df.index >= window_start) & (df.index <= window_end), index=df.index
        )
        out["Run_Detection_Mins"] = detection_mask.resample("D").sum().reindex(out.index, fill_value=0).astype(int)
        expected_mins = []
        for day in out.index:
            next_day = day + pd.DateOffset(days=1)
            expected_mins.append(int((next_day.tz_convert("UTC") - day.tz_convert("UTC")).total_seconds() / 60) if day.tzinfo else 1440)
        out["Run_Detection_Expected_Mins"] = expected_mins
        out.attrs["run_detection_window"] = {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
        }

    def _day(ts):
        return pd.Timestamp(ts).normalize()

    # ---- Cycling (from the same runs the Run Inspector shows) ----
    run_rows = [
        {
            "day": _day(r["start"]),
            "type": r.get("run_type"),
            "mins": float(r.get("duration_mins") or 0),
        }
        for r in (runs or [])
        if r.get("start") is not None
    ]
    runs_df = pd.DataFrame(run_rows, columns=["day", "type", "mins"])
    heating = runs_df[runs_df["type"] == "Heating"]

    def _per_day(frame):
        return frame.groupby("day").size().reindex(out.index, fill_value=0).astype(int)

    out["Starts"] = _per_day(runs_df)
    out["Starts_Heating"] = _per_day(heating)
    out["Starts_DHW"] = _per_day(runs_df[runs_df["type"] == "DHW"])
    out["Short_Cycles_Count"] = _per_day(heating[heating["mins"] < short_min])
    out["Very_Short_Cycles_Count"] = _per_day(heating[heating["mins"] < very_short_min])
    out["Cycling_Severity_Index"] = (
        out["Short_Cycles_Count"] / out["Starts_Heating"].replace(0, np.nan)
    )
    out["Median_Run_Time_Mins"] = (
        heating.groupby("day")["mins"].median().reindex(out.index)
    )
    if run_window_start is not None:
        # HA retention: runs are only detected from the high-resolution window
        # on. Earlier days are unknown, not "0 starts".
        first_day = pd.Timestamp(run_window_start)
        tz = getattr(out.index, "tz", None)
        if tz is not None:
            first_day = (
                first_day.tz_localize(tz) if first_day.tzinfo is None
                else first_day.tz_convert(tz)
            )
        elif first_day.tzinfo is not None:
            first_day = first_day.tz_localize(None)
        first_day = first_day.normalize()
        cycling_cols = [
            "Starts", "Starts_Heating", "Starts_DHW", "Short_Cycles_Count",
            "Very_Short_Cycles_Count", "Cycling_Severity_Index", "Median_Run_Time_Mins",
        ]
        out[cycling_cols] = out[cycling_cols].astype(float)
        out.loc[out["Run_Detection_Mins"].eq(0), cycling_cols] = np.nan

    # ---- DHW efficiency ----
    if {"Heat_DHW_kWh", "Electricity_DHW_kWh"} <= set(out.columns):
        elec = out["Electricity_DHW_kWh"]
        out["DHW_SCOP"] = (out["Heat_DHW_kWh"] / elec).where(elec > 0.01)

    # ---- Tariff usage ----
    if {"Power", "Current_Rate"} <= set(df.columns):
        power = pd.to_numeric(df["Power"], errors="coerce").fillna(0.0)
        rate = pd.to_numeric(df["Current_Rate"], errors="coerce")
        metered_kwh = (power.resample("D").sum() / 60000.0).reindex(out.index)
        day_key = rate.index.normalize()
        day_min = rate.groupby(day_key).transform("min")
        day_rates = rate.groupby(day_key).transform("nunique")
        cheap = rate.le(day_min + 1e-9) & day_rates.gt(1)
        cheap_kwh = (power.where(cheap, 0.0).resample("D").sum() / 60000.0).reindex(out.index)
        multi_rate = (day_rates.gt(1).resample("D").max()).reindex(out.index).fillna(False).astype(bool)
        usable = metered_kwh > 0.1
        out["Metered_Electricity_kWh"] = metered_kwh
        out["Cheap_Rate_Electricity_kWh"] = cheap_kwh.where(multi_rate)
        out["Night_Share_Of_Total_HP_Elec"] = (cheap_kwh / metered_kwh).where(usable & multi_rate)
        if "Daily_Cost_Euro" in out.columns:
            out["Effective_Avg_Tariff"] = (out["Daily_Cost_Euro"] / metered_kwh).where(usable)

    # ---- Flow-limited minutes ----
    counted, valid, eligible, source = flow_limit_analysis(df, user_config)
    out.attrs["flow_target_source"] = source
    if counted is not None:
        starts = counted & ~counted.shift(fill_value=False)
        eligible_daily = eligible.resample("D").sum().reindex(out.index, fill_value=0)
        valid_daily = valid.resample("D").sum().reindex(out.index, fill_value=0)
        coverage = safe_div(valid_daily, eligible_daily, default=np.nan)
        out["FTlim_Input_Coverage"] = coverage
        minutes = (
            counted.resample("D").sum().reindex(out.index, fill_value=0).astype(int)
        )
        events = (
            starts.resample("D").sum().reindex(out.index, fill_value=0).astype(int)
        )
        sufficient = coverage.ge(0.90)
        no_heating = eligible_daily.eq(0)
        out["Virtual_FTlim_Time_Mins"] = minutes.where(sufficient | no_heating)
        out["Virtual_FTlim_Events"] = events.where(sufficient | no_heating)
        out.loc[no_heating, ["Virtual_FTlim_Time_Mins", "Virtual_FTlim_Events"]] = 0
    else:
        eligible_daily = eligible.resample("D").sum().reindex(out.index, fill_value=0)
        no_heating = eligible_daily.eq(0)
        out["FTlim_Input_Coverage"] = np.nan
        out["Virtual_FTlim_Time_Mins"] = np.nan
        out["Virtual_FTlim_Events"] = np.nan
        out.loc[no_heating, ["Virtual_FTlim_Time_Mins", "Virtual_FTlim_Events"]] = 0
    return out
