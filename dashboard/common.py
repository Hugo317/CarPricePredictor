"""Shared by every dashboard page: the palette, the Plotly chart style, panels, and loading the notebook CSVs.

The numbers are computed in the notebooks (05, 11, 12, ...) and saved to DASHBOARD_DIR; the pages only read them.
"""

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from carpricepredictor.shared import DASHBOARD_DIR

# palette: Graphite & Gold (premium automotive, dark)
BG = "#0c0c0e"  # page
PANEL = "#161618"  # panels and cards
LINE = "#2a2a2e"  # borders, gridlines, faint marks
TEXT = "#f2efe8"  # warm white
SUBTEXT = "#a09c94"  # secondary text, and "everything else" groups
GOLD = "#cf9a30"  # the accent: main series, buttons, logo
BLUE = "#5a8fe0"  # second series
RED = "#e05a3a"  # bad, removed, expensive
GREEN = "#3fa88f"  # fourth series; good, cheap
GRAPHITE = "#6b6b73"  # neutral series; average
GREY = "#4a4a50"  # reference marks that aren't real contenders (e.g. the Dummy model)
WHITE = "#ffffff"
FONT = "Space Grotesk, sans-serif"

TEMPLATE = go.layout.Template(layout=dict(
    font=dict(family=FONT, color=TEXT, size=13),
    paper_bgcolor=PANEL,  # charts always sit in a panel
    plot_bgcolor=PANEL,
    colorway=[GOLD, BLUE, RED, GREEN],  # passes the colour-blind check on PANEL (dataviz validator)
    xaxis=dict(showgrid=False, zeroline=False, linecolor=GRAPHITE, tickcolor=GRAPHITE, automargin=True),
    yaxis=dict(gridcolor=LINE, zeroline=False, automargin=True),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=10, b=10),
    hoverlabel=dict(font_family=FONT),
))

# panels (keyed containers) and number cards get the panel colour; Streamlit's own chrome is hidden
STYLE = f"""<style>
[class*="st-key-panel_"] {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 1rem; padding: 1.25rem 1.5rem; }}
[data-testid="stMetric"] {{ background: {PANEL}; }}
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stStatusWidget"],
[data-testid="stDecoration"] {{ display: none; }}
[data-testid^="stBaseButton-primary"], [data-testid^="stBaseButton-primary"] p {{ color: {BG}; font-weight: 700; }}
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


def load(name, notebook, needs=()):
    """A CSV from DASHBOARD_DIR; stops the page with a hint when the notebook hasn't been run (or re-run) yet.

    needs: columns the page uses that older runs of the notebook didn't write.
    """
    path = DASHBOARD_DIR / name
    if not path.exists():
        st.warning(f"`data/dashboard/{name}` not found: run `{notebook}` first.")
        st.stop()
    table = _read(path, path.stat().st_mtime)  # the timestamp re-reads the file after a notebook re-run
    if missing := [c for c in needs if c not in table.columns]:
        st.warning(f"`data/dashboard/{name}` is from an older run (no {', '.join(missing)}): re-run `{notebook}`.")
        st.stop()
    return table
