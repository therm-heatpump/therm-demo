# onboarding.py
"""
The setup wizard: an entry screen (Quick start / Guided / Full), one Setup section per step, and a
Review screen that saves. It draws the same section renderers as the Setup page against the same draft
(setup_sections), so leaving the wizard for the page, or coming back, loses nothing.

State: st.session_state["onb_mode"] is None (no wizard), "entry", "quick", "guided", "full" or "page"
(the user chose the whole Setup page); "onb_step" indexes the route's steps, the last being Review.
"""
from __future__ import annotations

import streamlit as st

import config
import config_manager
import setup_sections as ss
from schema_defs import OPTIONAL_SENSORS, RECOMMENDED_SENSORS, REQUIRED_SENSORS

MODE, STEP = "onb_mode", "onb_step"
ROUTES = ("quick", "guided", "full")
LEVEL_NAMES = {"quick": "Quick start", "guided": "Guided set-up", "full": "Full set-up"}


def _rooms_and_zones(draft: dict, ctx: ss.Ctx) -> None:
    ss.render_rooms(draft, ctx)
    st.markdown("##### Heating zones")
    if ctx.level == "full":
        ss.render_zones(draft, ctx)
    elif ss.zones_on(draft):
        st.caption("This set-up uses zone signals; change them under ⚙️ Setup → Rooms & zones.")
    else:
        st.caption("No heating zones: every room sensor is used for heating runs. Zones with their own "
                   "on/off signal can be added later under ⚙️ Setup → Rooms & zones.")


# id, title, what the step is for, renderer, routes that include it
SECTIONS = [
    ("sensors_core", "Heat pump sensors",
     "The readings everything else is calculated from. Detect has filled in what it found; check each one.",
     ss.render_sensors_core, {"quick", "guided", "full"}),
    ("heat_calc", "How heat is measured",
     "These change every heat and COP figure: the fluid in the primary circuit (water or a glycol mix, about "
     "7 %) and whether the power meter already includes the immersion heater.",
     ss.render_heat_calc, {"guided", "full"}),
    ("hot_water", "Hot water detection",
     "How therm tells hot-water runs from heating runs, which splits energy and COP between the two.",
     ss.render_hot_water, {"guided", "full"}),
    ("costs", "Electricity price & time zone",
     "For the cost figures, and so days and price bands follow your local clock.",
     ss.render_costs, {"guided", "full"}),
    ("rooms_zones", "Rooms & heating zones",
     "Optional: room temperatures to show against heating runs.",
     _rooms_and_zones, {"guided", "full"}),
    ("weather", "Weather sensors",
     "Optional: wind, humidity and sunshine, and a second source to fill gaps.",
     ss.render_weather, {"full"}),
    ("extra_sensors", "Other sensors",
     "Optional readings that add detail.",
     ss.render_extra_sensors, {"full"}),
    ("run_detection", "Run detection",
     "When therm counts the heat pump as running, and what counts as a short cycle. The defaults suit most "
     "systems.",
     ss.render_run_detection, {"full"}),
    ("hydraulics", "Hydraulics & flow limits",
     "Flow thresholds and your weather-compensation curve (for flow-limit detection). The defaults suit most "
     "systems.",
     ss.render_hydraulics, {"full"}),
    ("diagnostics", "Diagnostics thresholds",
     "Expert settings; the defaults suit most systems.",
     ss.render_diagnostics, {"full"}),
    ("notes", "AI context & change log",
     "Optional: notes for AI reports, and a log of changes you make to the heat pump's settings.",
     ss.render_notes, {"full"}),
]


def steps_for(route: str, draft: dict) -> list:
    """The route's steps (Review is added after them). Quick start asks nothing, unless the essential
    sensors are missing: then it opens the heat pump sensors first."""
    if route == "quick":
        return [] if _essentials_mapped(draft) else [SECTIONS[0]]
    return [s for s in SECTIONS if route in s[4]]


def _essentials_mapped(draft: dict) -> bool:
    return all(ss.mapped(draft, r) for r in REQUIRED_SENSORS)


def active(first_run: bool) -> bool:
    """Whether Setup shows the wizard in this run (starting it on a first run)."""
    mode = st.session_state.get(MODE)
    if mode is None and first_run:
        st.session_state[MODE] = mode = "entry"
    return mode not in (None, "page")


def open_page(tab: str | None = None) -> None:
    """Leave the wizard for the whole Setup page (optionally on one tab); the draft is kept."""
    st.session_state[MODE] = "page"
    st.session_state.pop(STEP, None)
    if tab:
        st.session_state.pop("setup_tab", None)
        st.session_state["setup_tab_default"] = tab


def restart() -> None:
    """Run the wizard again, pre-filled from the current draft."""
    st.session_state[MODE] = "entry"
    st.session_state.pop(STEP, None)


def render(draft: dict, ctx: ss.Ctx) -> bool:
    """Draw the current wizard screen. Returns True when the user presses Save on the Review screen."""
    mode = st.session_state.get(MODE)
    if mode == "entry":
        _render_entry(draft)
        return False
    ctx.level = "guided" if mode in ("quick", "guided") else "full"
    steps = steps_for(mode, draft)
    index = min(st.session_state.get(STEP, 0), len(steps))
    total = len(steps) + 1
    title = steps[index][1] if index < len(steps) else "Review"
    st.progress((index + 1) / total, text=f"{LEVEL_NAMES[mode]} · step {index + 1} of {total} · {title}")
    if index < len(steps):
        _render_step(draft, ctx, steps, index)
        return False
    return _render_review(draft, ctx, steps)


def _go(index: int) -> None:
    st.session_state[STEP] = index


def _render_entry(draft: dict) -> None:
    st.markdown("### Set up therm")
    found = (st.session_state.get("autodetect_result") or {})
    if found.get("rows"):
        via = ", ".join(found.get("presets") or []) or "known integrations"
        essentials = ", ".join(REQUIRED_SENSORS[r]["label"] + (" ✔" if ss.mapped(draft, r) else " ✘")
                               for r in REQUIRED_SENSORS)
        st.success(f"Found {len(found['rows'])} heat pump sensors via *{via}* ({essentials}).")
    else:
        st.info("therm couldn't recognise your heat pump's sensors automatically; you'll pick them in the first "
                "step.")
    cards = [
        ("quick", "Quick start", "Use what therm found. Change anything later in ⚙️ Setup."),
        ("guided", "Guided (recommended)", "Five short steps: the settings that change your COP and cost "
                                            "figures most."),
        ("full", "Full set-up", "Every sensor and threshold, one step at a time."),
    ]
    for col, (route, name, promise) in zip(st.columns(3), cards):
        with col, st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(promise)
            if st.button(f"Start {name.split(' (')[0].lower()}", key=f"onb_start_{route}", width="stretch",
                         type="primary" if route == "guided" else "secondary"):
                st.session_state[MODE] = route
                st.session_state[STEP] = 0
                draft["setup_level"] = route
                st.rerun()
    if st.button("I'd rather see the whole Setup page", key="onb_page", type="tertiary"):
        open_page()
        st.rerun()


def _step_problems(step_id: str, draft: dict) -> list[str]:
    """What must be fixed before leaving a step."""
    if step_id == "sensors_core" and not _essentials_mapped(draft):
        missing = [REQUIRED_SENSORS[r]["label"] for r in REQUIRED_SENSORS if not ss.mapped(draft, r)]
        return ["Choose a sensor for: " + ", ".join(missing) + "."]
    if step_id == "costs":
        return list(st.session_state.get(ss.ERRORS_KEY, {}).get("tariff", []))
    return []


def _render_step(draft: dict, ctx: ss.Ctx, steps: list, index: int) -> None:
    step_id, title, purpose, render, _routes = steps[index]
    st.markdown(f"### {title}")
    st.caption(purpose)
    render(draft, ctx)
    st.divider()
    back, nxt, finish = st.columns([1, 1, 2])
    with back:
        if index > 0 and st.button("← Back", key="onb_back", width="stretch"):
            _go(index - 1)
            st.rerun()
    with nxt:
        if st.button("Next →", key="onb_next", type="primary", width="stretch"):
            problems = _step_problems(step_id, draft)
            if problems:
                for p in problems:
                    st.error(p)
            else:
                _go(index + 1)
                st.rerun()
    with finish:
        if index < len(steps) - 1 and st.button("Finish now with defaults for the rest", key="onb_finish",
                                                width="stretch"):
            problems = _step_problems(step_id, draft)
            if problems:
                for p in problems:
                    st.error(p)
            else:
                _go(len(steps))
                st.rerun()


def _worth_checking(draft: dict, profile: dict) -> list[tuple[str, str]]:
    """(message, Setup tab) for settings the user may want to look at later."""
    items = []
    mapping = profile["mapping"]
    if draft.get("_modbus_no_power") and not mapping.get("Power"):
        items.append(("Samsung Modbus (MIM-B19N) doesn't report the heat pump's electricity use: add a separate "
                      "electricity meter and choose it as Heat Pump Power.", "Sensors"))
    if "FlowRate" not in mapping and "Heat" not in mapping:
        items.append(("No flow rate sensor: heat output, and so COP, can't be calculated.", "Sensors"))
    if not any(r in mapping for r in ("DHW_Active", "ValveMode", "DHW_Mode")):
        items.append(("No hot-water status or valve sensor: hot-water runs are recognised from the tank "
                      "temperature only.", "Heat & hot water"))
    rates = [r["rate"] for p in profile["tariff_structure"] for r in p.get("rules", [])]
    if rates == [0.33]:
        items.append(("Electricity price is therm's example (0.33 per kWh): enter yours for real costs.",
                      "Costs & time zone"))
    if draft.get("setup_level") == "quick":
        fluid = {v: k for k, v in config.FLUID_PRESETS.items()}.get(
            profile["physics_thresholds"]["fluid_specific_heat_kj"], "custom")
        items.append((f"Primary circuit fluid assumed: {fluid}.", "Heat & hot water"))
    if draft.get("_new") and "time zone" not in draft.get("_from_home_assistant", {}):
        items.append((f"Time zone and currency are therm's defaults ({profile['timezone']}, "
                      f"{profile['currency_code']}): Home Assistant's settings weren't available.",
                      "Costs & time zone"))
    if profile.get("heating_zones") != ss.ZONES_SIGNALS:
        items.append(("No heating zone signals: heating while the tank is being reheated can't be identified.",
                      "Rooms & zones"))
    return items


def _render_review(draft: dict, ctx: ss.Ctx, steps: list) -> bool:
    profile, problems = ss.build_config(draft)
    ok, validation = config_manager.validate_config(profile)
    problems = problems + ([] if ok else list(validation))
    st.markdown("### Review")
    mapping = profile["mapping"]
    catalogue = {**REQUIRED_SENSORS, **RECOMMENDED_SENSORS, **OPTIONAL_SENSORS}
    sensors = [f"{catalogue[r]['label']}: `{ctx.format_opt(mapping[r])}`" for r in catalogue if r in mapping]
    rooms = sum(1 for r in mapping if r.startswith("Room_"))
    rates = profile["tariff_structure"]
    lines = [
        f"**Profile:** {profile['profile_name']}",
        "**Sensors:** " + ("; ".join(sensors) if sensors else "none"),
        f"**Rooms:** {rooms}" + (" · **zones:** " + ", ".join(z.replace('_', ' ') for z in mapping
                                                           if z.startswith('Zone_'))
                                 if profile.get("heating_zones") == ss.ZONES_SIGNALS else " · no heating zones"),
        f"**Electricity prices:** {len(rates)} price period(s), {profile['currency']} "
        f"({profile['currency_code']}) · **time zone:** {profile['timezone']}"
        + (" (from Home Assistant)" if draft.get("_from_home_assistant") else ""),
    ]
    for line in lines:
        st.markdown(line)

    checks = _worth_checking(draft, profile)
    if checks:
        st.markdown("**Worth checking later**")
        for i, (msg, tab) in enumerate(checks):
            c_msg, c_btn = st.columns([4, 1], vertical_alignment="center")
            c_msg.caption(f"• {msg}")
            if c_btn.button("Open in Setup", key=f"onb_check_{i}", width="stretch"):
                open_page(tab)
                st.rerun()
    for p in problems:
        st.error(p)

    st.divider()
    back, save = st.columns([1, 3])
    with back:
        if steps and st.button("← Back", key="onb_back", width="stretch"):
            _go(len(steps) - 1)
            st.rerun()
        elif not steps and st.button("← Start again", key="onb_restart", width="stretch"):
            restart()
            st.rerun()
    with save:
        return bool(st.button("Save and start", key="onb_save", type="primary", width="stretch",
                              disabled=bool(problems)))
