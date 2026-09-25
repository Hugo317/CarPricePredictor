"""Shared by every dashboard page: the palette, the Plotly chart style, panels, and loading the notebook CSVs.

The numbers are computed in the notebooks (05, 11, 12, ...) and saved to DASHBOARD_DIR; the pages only read them.
"""

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from carpricepredictor.shared import DASHBOARD_DIR

# palette: dark navy + teal
BG = "#0f172a"
PANEL = "#1e293b"
TEXT = "#e2e8f0"
TEAL = "#14b8a6"
AMBER = "#f59e0b"
ROSE = "#f43f5e"
SLATE = "#64748b"
VIOLET = "#a78bfa"
LIGHT_SLATE = "#94a3b8"  # secondary text, and "everything else" groups
DARK_SLATE = "#334155"  # faint background marks
FONT = "Space Grotesk, sans-serif"

TEMPLATE = go.layout.Template(layout=dict(
    font=dict(family=FONT, color=TEXT, size=13),
    paper_bgcolor=PANEL,  # charts always sit in a panel
    plot_bgcolor=PANEL,
    colorway=[TEAL, AMBER, ROSE, SLATE],
    xaxis=dict(showgrid=False, zeroline=False, linecolor=SLATE, tickcolor=SLATE, automargin=True),
    yaxis=dict(gridcolor="rgba(100, 116, 139, 0.25)", zeroline=False, automargin=True),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=10, b=10),
    hoverlabel=dict(font_family=FONT),
))

# panels (keyed containers) and number cards get the panel colour
STYLE = f"""<style>
[class*="st-key-panel_"] {{ background: {PANEL}; border-radius: 1rem; padding: 1.25rem 1.5rem; }}
[data-testid="stMetric"] {{ background: {PANEL}; }}
</style>"""


def style():
    st.html(STYLE)


def panel(title):
    """A rounded panel box with the title inside: `with panel("Cleaning funnel"): ...`"""
    box = st.container(key="panel_" + re.sub(r"\W+", "_", title.lower()))
    box.markdown(f"**{title}**")
    return box


def chart(fig):
    # set here, not as Plotly's default: Streamlit swaps the default template for its own when it loads.
    # theme=None stops Streamlit from restyling the chart on top of it, but it still paints the page colour
    # behind any chart without its own background, so the backgrounds are set on the figure itself.
    fig.update_layout(template=TEMPLATE, paper_bgcolor=PANEL, plot_bgcolor=PANEL)
    st.plotly_chart(fig, theme=None)


@st.cache_data
def _read(path, modified):
    return pd.read_csv(path)


def load(name, notebook):
    """A CSV from DASHBOARD_DIR; stops the page with a hint when the notebook hasn't been run yet."""
    path = DASHBOARD_DIR / name
    if not path.exists():
        st.warning(f"`data/dashboard/{name}` not found: run `{notebook}` first.")
        st.stop()
    return _read(path, path.stat().st_mtime)  # the timestamp re-reads the file after a notebook re-run
