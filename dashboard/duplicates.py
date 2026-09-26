import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import BLUE, GOLD, LINE, PANEL, RED, SUBTEXT, TEXT, chart, load, panel

NOTEBOOK = "12_dashboard_duplicates.ipynb"

st.title("Duplicates")

summary = load("dup_summary.csv", NOTEBOOK).set_index("metric")["value"]
evidence = load("dup_evidence.csv", NOTEBOOK)
why = load("dup_why.csv", NOTEBOOK)
posts_per_car = load("dup_posts_per_car.csv", NOTEBOOK)
examples = load("dup_examples.csv", NOTEBOOK, needs=["usual_price"])
text = load("dup_text.csv", NOTEBOOK)
reposts = load("dup_reposts.csv", NOTEBOOK)
wrong = load("dup_wrong.csv", NOTEBOOK)
unsure = load("dup_unsure.csv", NOTEBOOK)

before, removed = int(summary["rows before the dedupe"]), int(summary["removed"])

st.caption(f"The last cleaning step removed {removed:,} listings: the same car posted more than once. "
           "Why that happens, how we know they're the same car, and what's still left.")

cards = st.columns(4)
cards[0].metric("Listings before the dedupe", f"{before:,}", border=True)
cards[1].metric("Removed as duplicates", f"{removed:,}", border=True)
cards[2].metric("Share removed", f"{summary['removed share']:.1%}", border=True)
cards[3].metric("Left after the dedupe", f"{int(summary['kept']):,}", border=True)

with st.expander("What counts as a duplicate"):
    st.markdown(
        "- Two listings are the same when **every column matches except `state`**: price, year, manufacturer, "
        "model, odometer, condition, colour, fuel, title, transmission, drive, type, cylinders.\n"
        "- `state` is left out because the same car is often posted in several states.\n"
        "- The raw listing fields (url, VIN, description, lat / long) are left out too: they differ between "
        "copies of the same car, and they're what's used below to check that the copies really are the same car.\n"
        "- The first listing of each group is kept, the rest are removed."
    )


# ---------- why they exist ----------
st.subheader("Why they exist")
st.caption("Craigslist is one website split into about 400 regional sites, and buyers mostly browse their own. "
           "A seller who wants buyers in more cities posts the car again in each region, and posts again later "
           "to get back to the top of the list.")

cards = st.columns(3)
cards[0].metric("Removed copies posted by dealers", f"{summary['dealer share of removed']:.0%}", border=True)
cards[1].metric("Posted by Carvana alone", f"{summary['carvana share of removed']:.0%}", border=True)
cards[2].metric("Same post scraped twice", f"{int(summary['same post scraped twice']):,}", border=True)

with panel("Removed copies by reason"):
    fig = go.Figure(go.Bar(
        x=why["rows"], y=why["why"], orientation="h", marker_color=GOLD,
        text=[f"{n:,}  ({s:.0%})" for n, s in zip(why["rows"], why["share"])],
        textposition="outside", textfont=dict(color=TEXT),
        customdata=why[["dealer", "carvana", "median_days_apart"]],
        hovertemplate="<b>%{y}</b><br>%{x:,} copies<br>dealers %{customdata[0]:.0%} · Carvana %{customdata[1]:.0%}"
                      "<br>median %{customdata[2]:.1f} days after the kept copy<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickformat=",", showgrid=True, gridcolor=LINE, range=[0, why["rows"].max() * 1.25])
    fig.update_layout(height=45 * len(why) + 70, bargap=0.5)
    chart(fig)

left, right = st.columns(2)
with left, panel("How many times each car was posted"):
    # cars with a real VIN, before the dedupe; the long tail is grouped as 10+
    counts = posts_per_car.assign(bucket=posts_per_car["posts"].clip(upper=10)).groupby("bucket")["cars"].sum()
    labels = [str(p) if p < 10 else "10+" for p in counts.index]
    fig = go.Figure(go.Bar(
        x=labels, y=counts.values, marker_color=GOLD,
        text=[f"{n:,}" for n in counts.values], textposition="outside", textfont=dict(color=TEXT),
        hovertemplate="posted %{x} times: %{y:,} cars<extra></extra>", constraintext="none",
    ))
    fig.update_xaxes(title="posts", type="category")
    fig.update_yaxes(tickformat=",", title="cars", range=[0, counts.max() * 1.12], gridcolor=LINE)
    fig.update_layout(height=360, bargap=0.35)
    chart(fig)

with right, panel("Is the ad copy-pasted?"):
    st.caption("Cars posted 2+ times, by seller: how similar later posts' ads are to the first one.")
    for column, seller in zip(st.columns(len(text)), text.to_dict("records")):
        with column.container(border=True):
            st.markdown(f"**{seller['kind'].capitalize()}**")
            st.markdown(f"<div style='font-size:2rem; line-height:1.2'>{seller['median_similarity']:.0%}</div>"
                        f"<div style='color:{SUBTEXT}'>similar to the first post</div>", unsafe_allow_html=True)
            st.markdown(f"{seller['same ad text']:.0%} identical word for word  \n{seller['cars']:,} cars")
            st.caption(f"similarity from {seller['compared_pairs']:,} compared posts")

with st.expander("How this works"):
    st.markdown(
        "- **Reason**: each removed copy is compared with the listing it matched. The same post id = scraped twice; "
        "the same regional site = re-posted to get back to the top; another regional site = cross-posted.\n"
        "- **Dealer**: craigslist urls have `/ctd/` for dealers and `/cto/` for owners. **Carvana**: the ad mentions Carvana.\n"
        "- **Posts per car**: only cars with a real 17-character VIN can be counted.\n"
        "- **Copy-pasted**: for cars posted 2+ times, the share whose ad text is identical in every post, and the "
        "similarity between the first post and later ones (1.0 = identical) on a sample of 300 cars. Ads are rarely "
        "identical to the character (a local phone number or tracking code changes), but close to 1.0 means copy-pasted."
    )


# ---------- same car? ----------
st.subheader("Are they really the same car?")
st.caption("The dedupe never looked at the VIN or the ad text, so they're an independent check: for each removed "
           "copy, do they match the listing it was matched to?")

with panel("Evidence for each removed copy"):
    colours = [RED if e.startswith("different VIN") else GOLD for e in evidence["evidence"]]
    fig = go.Figure(go.Bar(
        x=evidence["rows"], y=evidence["evidence"], orientation="h", marker_color=colours,
        text=[f"{n:,}  ({s:.1%})" if s >= 0.001 else f"{n:,}  (<0.1%)" for n, s in zip(evidence["rows"], evidence["share"])],
        textposition="outside", textfont=dict(color=TEXT),
        customdata=evidence["dealer"],
        hovertemplate="<b>%{y}</b><br>%{x:,} copies · dealers %{customdata:.0%}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickformat=",", showgrid=True, gridcolor=LINE, range=[0, evidence["rows"].max() * 1.25])
    fig.update_layout(height=45 * len(evidence) + 70, bargap=0.5)
    chart(fig)

with st.expander("The doubtful ones"):
    st.markdown("**Different VIN**: different cars with identical values, almost always a dealer with several "
                "identical cars at the same price. For the model they're the same row anyway, so losing them costs little.")
    st.dataframe(wrong, hide_index=True)
    st.markdown("**No VIN, different ad text**: can't be proven either way from this data. The start of both ads:")
    st.dataframe(unsure, hide_index=True)


# ---------- four real cars ----------
st.subheader("Four real cars")
st.caption("Every post of four cars with a real VIN, over time. Teal = the car's usual price, amber = posted at "
           "a different price: a post like that isn't an exact match, so it slips past the dedupe.")


def car_panel(car, by):
    first = car.iloc[0]
    title = (f"{first['kind'].capitalize()} · {first['car']} · {len(car)} posts in "
             f"{car['region'].nunique()} regions, {car['state'].nunique()} states")
    with panel(title):
        car = car.assign(
            time=pd.to_datetime(car["posted"]),
            hover=(car["region"] + ", " + car["state"] + " · " + car["posted"].str.replace("T", " ")
                   + " · $" + car["price"].map("{:,.0f}".format)
                   + " · " + car["odometer"].map(lambda v: "no odometer" if pd.isna(v) else f"{v:,.0f} mi")
                   + "<br>model typed as: " + car["model_raw"].fillna("")),
        )
        order = list(dict.fromkeys(car.sort_values("time")[by]))  # rows in order of the first post there
        fig = go.Figure()
        for usual, colour, name in [(True, GOLD, "usual price"), (False, BLUE, "different price")]:
            d = car[car["usual_price"] == usual]
            fig.add_trace(go.Scatter(
                x=d["time"], y=d[by], mode="markers", name=name,
                marker=dict(color=colour, size=10, line=dict(color=PANEL, width=2)),
                text=d["hover"], hovertemplate="%{text}<extra></extra>",
            ))
        fig.update_yaxes(categoryorder="array", categoryarray=order, autorange="reversed", gridcolor=LINE)
        fig.update_layout(height=max(22 * len(order), 110) + 110, margin=dict(t=40), showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        chart(fig)
    with st.expander("Posts"):
        st.dataframe(car[["posted", "region", "state", "seller", "price", "usual_price", "odometer", "model_raw",
                          "same_ad_as_first"]], hide_index=True)


cars = [g for _, g in examples.groupby("vin", sort=False)]
columns = st.columns(len(cars) - 1)
for column, car in zip(columns, cars[:-1]):
    with column:
        car_panel(car, "region")
car_panel(cars[-1], "state")  # the most posted car: hundreds of regions, so one row per state


# ---------- what the dedupe misses ----------
st.subheader("What the dedupe misses")
st.caption("A car posted again with a new price or mileage isn't an exact match, so it stays in the data twice.")

cards = st.columns(2)
cards[0].metric("Listings still sharing a VIN", f"{int(summary['rows still sharing a VIN']):,}", border=True)
cards[1].metric("Cars", f"{int(summary['cars still sharing a VIN']):,}", border=True)

with panel("What differs between the copies of those cars"):
    share = reposts.columns[1]
    reposts = reposts.sort_values(share, ascending=False)
    fig = go.Figure(go.Bar(
        x=reposts[share], y=reposts["column"], orientation="h", marker_color=GOLD,
        text=[f"{v:.0%}" for v in reposts[share]], textposition="outside", textfont=dict(color=TEXT),
        hovertemplate="%{y} differs for %{x:.0%} of the cars<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickformat=".0%", range=[0, 1], showgrid=True, gridcolor=LINE)
    fig.update_layout(height=45 * len(reposts) + 70, bargap=0.5)
    chart(fig)


# ---------- why it matters ----------
st.subheader("Why it matters for the model")
st.caption("Copies of the same car can land in both the training and the test set, so the model is scored "
           "partly on cars it has already seen. Keeping one listing per VIN would fix it; it isn't applied yet.")

# from the repost check in 05_final_model.ipynb (iteration 2, validation slice of the training set)
cards = st.columns(3)
cards[0].metric("Validation rows that are reposts", "7.5%", border=True)
cards[1].metric("Error (RMSE, log) on reposts", "0.275", border=True)
cards[2].metric("Error (RMSE, log) on new cars", "0.304", border=True)
