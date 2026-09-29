# charts.py
"""
One way to show every Plotly chart in therm.

Plotly's toolbar (zoom, pan, download) appears at the top right of a chart while the pointer is over it,
which is where legends used to be: reaching for a legend item (to hide or show a series) brought the
toolbar up over it. Legends therefore sit below the chart, left-aligned, and Plotly makes room for them.
"""
from __future__ import annotations

import streamlit as st

# Below the plot area, clear of its (up to two-line) date labels; Plotly grows the bottom margin to fit.
# Only the vertical position changes: each chart keeps its own legend layout, grouping, alignment and
# colours.
LABEL_CLEARANCE_PX = 48
CONFIG = {"displaylogo": False}


def _plot_height(fig) -> float:
    """Height of the plot area in pixels (Plotly's defaults where the chart sets none)."""
    margin = fig.layout.margin
    return max((fig.layout.height or 450) - (margin.t if margin.t is not None else 100)
               - (margin.b if margin.b is not None else 80), 100)


def place_legend(fig):
    """Move a legend that sits at the top (where the toolbar appears) below the chart; legends a chart
    already places below it are left as they are."""
    legend = fig.layout.legend
    at_top = legend.y is None or legend.y >= 1 or (legend.yanchor == "bottom" and legend.y > 0.9)
    if fig.layout.showlegend is not False and at_top:
        fig.update_layout(legend=dict(yref="paper", yanchor="top",
                                      y=-round(LABEL_CLEARANCE_PX / _plot_height(fig), 3)))
    return fig


def show(fig, key: str, **kwargs) -> None:
    """st.plotly_chart with the legend clear of the toolbar."""
    st.plotly_chart(place_legend(fig), width="stretch", key=key, config=CONFIG, **kwargs)
