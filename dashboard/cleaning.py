import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, DARK_SLATE, LIGHT_SLATE, ROSE, SLATE, TEAL, TEXT, VIOLET, chart, load, panel

NOTEBOOK = "11_dashboard_cleaning.ipynb"
GRID = "rgba(100, 116, 139, 0.25)"

st.title("Data cleaning")

steps = load("cleaning_steps.csv", NOTEBOOK)
missing = load("missing_values.csv", NOTEBOOK)
fills = load("cleaning_fills.csv", NOTEBOOK)
sample = load("cleaning_sample.csv", NOTEBOOK)
names = load("model_names.csv", NOTEBOOK)
spellings = load("model_spellings.csv", NOTEBOOK)
examples = load("model_examples.csv", NOTEBOOK)

raw, clean = steps["rows_before"].iloc[0], steps["rows_after"].iloc[-1]

st.caption(f"How {raw:,} raw Craigslist listings became {clean:,} clean ones: what was removed, what was filled, and why.")

cards = st.columns(4)
cards[0].metric("Raw listings", f"{raw:,}", border=True)
cards[1].metric("Clean listings", f"{clean:,}", border=True)
cards[2].metric("Kept", f"{clean / raw:.1%}", border=True)
cards[3].metric("Columns dropped", int(missing["dropped_at"].notna().sum()), border=True)


# ---------- cleaning funnel ----------
st.subheader("Cleaning funnel")
st.caption("Every step that removed listings, top to bottom in the order they run. "
           "Duplicates are by far the biggest cut: the Duplicates page explains them.")

cuts = steps[steps["removed"] > 0]
with panel("Listings removed at each step"):
    fig = go.Figure(go.Waterfall(
        orientation="h",
        y=["raw listings", *cuts["step"], "clean listings"],
        x=[raw, *-cuts["removed"], clean],
        measure=["absolute", *["relative"] * len(cuts), "total"],
        text=[f"{raw:,}", *[f"−{n:,}" for n in cuts["removed"]], f"{clean:,}"],
        textposition="outside",
        textfont=dict(color=TEXT),
        customdata=["Craigslist listings, April–May 2021", *cuts["rule"], "what the model is trained on"],
        hovertemplate="<b>%{y}</b>  %{text}<br>%{customdata}<extra></extra>",
        decreasing=dict(marker=dict(color=ROSE)),
        totals=dict(marker=dict(color=TEAL)),
        connector=dict(line=dict(color=SLATE, width=1)),
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickformat=",", showgrid=True, gridcolor=GRID, range=[0, raw * 1.12])
    fig.update_layout(height=40 * (len(cuts) + 2) + 60, bargap=0.35)
    chart(fig)

with st.expander("How this works"):
    st.markdown("\n".join(
        f"- **{s.step}**: {s.rule}" + (f" (−{s.removed:,})" if s.removed else "")
        for s in steps.itertuples()
    ))


# ---------- what got removed ----------
st.subheader("What got removed")
st.caption("A random 20,000 of the raw listings by year and price. Coloured dots were removed by the step in the "
           "legend: click an entry to hide or show it.")

COLOURS = {"$0 price": TEAL, "before 1980": AMBER, "fake price": ROSE, "bad odometer": VIOLET}
OTHER = "missing values + duplicates"
ZERO = 0.3  # where $0 prices sit on the log axis

plot = sample.dropna(subset=["year"]).assign(
    y=lambda d: d["price"].where(d["price"] > 0, ZERO),
    group=lambda d: d["dropped_at"].where(d["dropped_at"].isin([*COLOURS, "kept"]), OTHER),
    # years are whole numbers: spread each year's dots over ±0.4 so they form a cloud instead of stripes
    x=lambda d: d["year"] + np.random.default_rng(0).uniform(-0.4, 0.4, len(d)),
    hover=lambda d: (d["year"].astype(int).astype(str) + " · $" + d["price"].map("{:,.0f}".format) + " · "
                     + d["odometer"].map(lambda v: "no odometer" if np.isnan(v) else f"{v:,.0f} mi")
                     + "<br>" + d["dropped_at"]),
)
# drawn back to front (kept at the bottom); the legend lists them the other way round
groups = [("kept", DARK_SLATE, 0.8), (OTHER, LIGHT_SLATE, 0.5)] + [(g, c, 0.9) for g, c in COLOURS.items()]
legend_order = [*COLOURS, OTHER, "kept"]

with panel("Raw listings by year and price"):
    fig = go.Figure()
    for group, colour, opacity in groups:
        d = plot[plot["group"] == group]
        fig.add_trace(go.Scattergl(
            x=d["x"], y=d["y"], mode="markers", name=group,
            marker=dict(color=colour, size=5, opacity=opacity),
            text=d["hover"], hovertemplate="%{text}<extra></extra>",
            legendrank=legend_order.index(group),
        ))
    ticks = [ZERO, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7]
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=["$0", "$1", "$10", "$100", "$1k", "$10k", "$100k", "$1M", "$10M"],
                     range=[np.log10(0.2), np.log10(4e7)], gridcolor=GRID)
    fig.update_xaxes(title=dict(text="model year", standoff=12), showgrid=False)
    fig.update_layout(height=560, legend=dict(orientation="h", y=1.08, x=0))
    chart(fig)

with st.expander("How this works"):
    st.markdown(
        "- Price is on a log scale so $1 and $1M fit on one chart. Listings with a $0 price sit on the bottom line.\n"
        "- **fake price**: under $500, under $1,000 for a 2010+ car, over $200k, or a 2021+ car under $5k "
        "(dealers often list the monthly financing price instead of the full price).\n"
        "- **before 1980**: classic cars follow different pricing, so they're out of scope.\n"
        "- **bad odometer**: missing, over 500,000 miles, or under 100 miles on a pre-2020 car.\n"
        "- **missing values + duplicates**: removed for a missing fuel, title, transmission or cylinder count, "
        "or because the same car was posted again. They're spread all over the chart, so they share one colour.\n"
        f"- Listings with no year can't be placed on the chart ({sample['year'].isna().sum()} of the 20,000)."
    )


# ---------- filled and missing values ----------
st.subheader("Filled and missing values")
st.caption("Share of missing values in each raw column before and after cleaning. Most gaps were filled "
           "from similar cars instead of throwing the listing away.")

placeholder = fills.set_index("column")["placeholder"]


def how(r):
    if r.how == "column dropped":
        return f"column dropped ({r.dropped_at})"
    if r.how == "filled":
        text = f"filled {r.filled:,.0f}"
        return text + (f" · {r.to_placeholder:,.0f} → '{placeholder[r.column]}'" if r.to_placeholder else "")
    if r.how == "left missing":
        return "left missing, the model fills it in"
    return r.how  # "rows dropped" or "never missing"


rows = missing.sort_values(["raw_missing", "column"], ascending=[True, False])
kept_cols = rows[rows["how"] != "column dropped"]
dropped_cols = rows[rows["how"] == "column dropped"]

with panel("Missing values per column, raw → clean"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(  # the line between the two dots
        x=[v for r in kept_cols.itertuples() for v in (r.raw_missing, r.clean_missing, None)],
        y=[v for r in kept_cols.itertuples() for v in (r.column, r.column, None)],
        mode="lines", line=dict(color=SLATE, width=2), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=kept_cols["raw_missing"], y=kept_cols["column"], mode="markers", name="raw",
        marker=dict(color=ROSE, size=10), hovertemplate="%{y}: %{x:.1%} missing in the raw data<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=kept_cols["clean_missing"], y=kept_cols["column"], mode="markers", name="clean",
        marker=dict(color=TEAL, size=10), hovertemplate="%{y}: %{x:.1%} missing after cleaning<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dropped_cols["raw_missing"], y=dropped_cols["column"], mode="markers", name="column dropped",
        marker=dict(color=LIGHT_SLATE, size=10, symbol="x-thin", line=dict(width=2, color=LIGHT_SLATE)),
        hovertemplate="%{y}: %{x:.1%} missing, column dropped<extra></extra>",
    ))
    for r in rows.itertuples():
        fig.add_annotation(x=1.01, xref="paper", xanchor="left", y=r.column, text=how(r),
                           showarrow=False, font=dict(color=LIGHT_SLATE, size=12))
    fig.update_xaxes(tickformat=".0%", range=[-0.03, 1.03], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(height=26 * len(rows) + 80, margin=dict(r=320),
                      legend=dict(orientation="h", y=1.04, x=0))
    chart(fig)

with st.expander("How this works"):
    st.markdown("\n".join(f"- **{f.column}**: {f.rule}" for f in fills.itertuples()) + "\n"
                "- **rows dropped**: listings missing a year, fuel, odometer, title or transmission were removed "
                "(too few to be worth guessing).\n"
                "- **left missing**: a few listings have impossible coordinates; they're set to missing and the "
                "model's imputer fills them with the median.")


# ---------- model names ----------
unknown_share = names.loc[names["model_clean"] == "unknown", "listings"].sum() / names["listings"].sum()

st.subheader("Model names")
st.caption(f"The raw model column is free text: {len(names):,} different spellings (each seen 3+ times) were mapped "
           f"to {names['model_clean'].nunique():,} clean models with Claude.")

left, right = st.columns([3, 2])
with left, panel("Clean models with the most raw spellings"):
    bars = spellings.assign(label=spellings["manufacturer"] + " " + spellings["model_clean"])
    fig = go.Figure(go.Bar(
        x=bars["spellings"], y=bars["label"], orientation="h", marker_color=TEAL,
        text=bars["spellings"], textposition="outside", textfont=dict(color=TEXT),
        customdata=bars["listings"],
        hovertemplate="<b>%{y}</b><br>%{x} spellings · %{customdata:,} listings<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, range=[0, bars["spellings"].max() * 1.12])
    fig.update_layout(height=32 * len(bars) + 40, bargap=0.3)
    chart(fig)

with right, panel("Before → after"):
    for (manufacturer, model), g in examples.groupby(["manufacturer", "model_clean"], sort=False):
        n = spellings.set_index(["manufacturer", "model_clean"]).loc[(manufacturer, model), "spellings"]
        st.markdown(f"**{manufacturer} {model}** · {n} spellings  \n"
                    + " · ".join(f"`{raw_name}`" for raw_name in g["model_raw"]) + " …")

with st.expander("How this works"):
    st.markdown(
        "- Every (manufacturer, raw model) pair seen 3+ times was sent to Claude in batches of 100.\n"
        "- The rules in the prompt: lowercase base model only (no trim, cab, engine or year), the same model always "
        "gets exactly the same name, keep the truck class (1500 / 2500 / 3500), collapse to the model family "
        "(328i → 3 series), and answer `unknown` for junk instead of guessing.\n"
        "- Names seen fewer than 3 times became `other`; a missing model became `unknown`.\n"
        f"- {unknown_share:.1%} of the listings ended up as `unknown`."
    )
