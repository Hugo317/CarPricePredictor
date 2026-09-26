import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from carpricepredictor.shared import MODEL_PATH, cat_features, keyword_cols, num_cols
from common import BG, GRAPHITE, GREEN, PANEL, RED, TEXT, chart, load, panel

NOTEBOOK = "13_dashboard_price_check.ipynb"
BAND = 0.10  # within ±10% of the fair price = average
GAUGE_END = 40  # the dial runs from −40% to +40%; anything further sits at the end
PLACEHOLDERS = {"unknown", "missing", "other"}
KEYWORDS = {
    "one_owner": "one owner", "carfax": "carfax", "warranty": "warranty", "leather": "leather",
    "sunroof": "sunroof", "navigation": "navigation", "turbo": "turbo", "lifted": "lifted",
    "high_trim": "high trim", "diesel_words": "diesel", "needs_work": "needs work",
}
KEYWORD_HELP = {  # the words each flag looks for, from DESCRIPTION_FLAGS in 02_data_cleaning.ipynb
    "high_trim": "platinum, limited, king ranch, denali, lariat, laramie, ltz, rubicon, raptor, ...",
    "diesel_words": "diesel, powerstroke, duramax, cummins",
    "needs_work": "needs work, mechanic special, as-is, for parts, not running, blown engine, ...",
}
VERDICTS = {"cheap": (GREEN, BG), "average": (GRAPHITE, TEXT), "expensive": (RED, TEXT)}  # background, text


@st.cache_resource(show_spinner="Loading the model (first time only)...")
def load_model():
    return joblib.load(MODEL_PATH)


def car_features(car, location, desc_len):
    """The form's car as one row of model input.

    A small copy of add_features (04_modeling.ipynb) and the listing fields (02_data_cleaning.ipynb):
    the model was trained on the 2021 listings, so age counts from 2021.
    """
    age = max(2021 - car["year"], 0)
    if car["fuel"] == "electric":
        cylinders = 0.0
    else:
        cylinders = np.nan if car["cylinders"] == "other" else float(car["cylinders"])
    row = {
        **{c: car[c] for c in cat_features},
        "odometer": car["odometer"],
        "age": age,
        "miles_per_year": car["odometer"] / max(age, 1),
        "log_odometer": np.log1p(car["odometer"]),
        "is_classic": int(car["year"] < 1995),
        "cylinders_num": cylinders,
        "is_dealer": int(car["dealer"]),
        "has_vin": int(car["vin"]),
        "desc_len": desc_len,  # the form doesn't ask for the ad text: the median ad length
        "lat": location["lat"],  # the form asks for the state: its median location
        "long": location["long"],
        **{k: int(car[k]) for k in keyword_cols},
    }
    return pd.DataFrame([row])


def to_catboost(X):
    """Copy of to_catboost in 05_final_model.ipynb, for a model trained on raw categories."""
    X = X[num_cols + cat_features].copy()
    X[cat_features] = X[cat_features].fillna("missing").astype(str)
    return X


def pick(label, column, prefill=None):
    """Dropdown with the column's values, most common first; starts at the pre-fill or the most common real value."""
    values = list(options.loc[options["column"] == column, "value"])
    default = prefill if prefill in values else next(v for v in values if v not in PLACEHOLDERS)
    return st.selectbox(label, values, index=values.index(default))


st.title("Price check")
# inputs in dark navy so what you can change stands out from the panel's labels
st.html(f"""<style>
.st-key-panel_your_car [data-testid="stSelectbox"] > div > div,
.st-key-panel_your_car [data-testid="stNumberInputContainer"] {{ background-color: {BG}; }}
</style>""")
st.caption("Enter a car and its asking price. The model predicts a fair price and says whether the asking price "
           "is cheap, average or expensive. Prices are for the 2021 US Craigslist market.")

options = load("pc_options.csv", NOTEBOOK)
models = load("pc_models.csv", NOTEBOOK)
states = load("pc_states.csv", NOTEBOOK).set_index("state")
defaults = load("pc_defaults.csv", NOTEBOOK).set_index("metric")["value"]

if not MODEL_PATH.exists():
    st.warning("`models/final_model.joblib` not found: run `05_final_model.ipynb` first.")
    st.stop()

left, right = st.columns([3, 2], gap="medium")

with left, panel("Your car"):
    # make and model sit outside the form so picking a model can pre-fill the fields below
    top = st.columns(2)
    makes = sorted(models["manufacturer"].unique())
    manufacturer = top[0].selectbox("Make", makes, index=makes.index("ford"))
    of_make = models[models["manufacturer"] == manufacturer].sort_values("model")
    most_listed = of_make.loc[of_make["listings"].idxmax(), "model"]
    model_names = list(of_make["model"])
    model = top[1].selectbox("Model", model_names, index=model_names.index(most_listed))
    typical = of_make.set_index("model").loc[model]

    with st.form("car", border=False):
        row = st.columns(2)
        year = row[0].number_input("Year", min_value=1980, max_value=2021, value=int(defaults["year"]))
        odometer = row[1].number_input("Mileage (miles)", min_value=0, max_value=500_000, step=1_000,
                                       value=int(round(defaults["odometer"], -3)))
        row = st.columns(2)
        with row[0]:
            condition = pick("Condition", "condition")
        with row[1]:
            title_status = pick("Title", "title_status")
        row = st.columns(2)
        with row[0]:
            fuel = pick("Fuel", "fuel", typical["fuel"])
        with row[1]:
            cylinders = pick("Cylinders", "cylinders", str(typical["cylinders"]))
        row = st.columns(2)
        with row[0]:
            transmission = pick("Transmission", "transmission", typical["transmission"])
        with row[1]:
            drive = pick("Drive", "drive", typical["drive"])
        row = st.columns(2)
        with row[0]:
            body = pick("Type", "type", typical["type"])
        with row[1]:
            paint_color = pick("Paint colour", "paint_color")
        row = st.columns(2)
        with row[0]:
            state = pick("State", "state")
        with row[1]:
            seller = st.radio("Seller", ["private owner", "dealer"], horizontal=True)

        vin = st.checkbox("VIN shown in the ad")
        st.markdown("**The ad mentions**")
        keyword_columns = st.columns(3)
        mentions = {k: keyword_columns[i % 3].checkbox(label, help=KEYWORD_HELP.get(k))
                    for i, (k, label) in enumerate(KEYWORDS.items())}

        asking = st.number_input("Asking price ($)", min_value=100, max_value=500_000, step=100, value=15_000)
        submitted = st.form_submit_button("Check price", type="primary", width="stretch")

if submitted:
    car = {"manufacturer": manufacturer, "model": model, "year": year, "odometer": odometer,
           "condition": condition, "title_status": title_status, "fuel": fuel, "cylinders": cylinders,
           "transmission": transmission, "drive": drive, "type": body, "paint_color": paint_color,
           "state": state, "dealer": seller == "dealer", "vin": vin, **mentions}
    saved = load_model()
    X = car_features(car, states.loc[state], defaults["desc_len"])
    fair = float(np.expm1(saved["model"].predict(to_catboost(X) if saved["use_native"] else X))[0])
    st.session_state["check"] = {"car": f"{year} {manufacturer} {model}", "fair": fair, "asking": asking}

with right, panel("Result"):
    check = st.session_state.get("check")
    if check is None:
        st.markdown("Fill in the car and its asking price, then press **Check price**.")
    else:
        diff = check["asking"] / check["fair"] - 1
        verdict = "cheap" if diff < -BAND else "expensive" if diff > BAND else "average"
        background, ink = VERDICTS[verdict]
        st.markdown(f"<div style='color:{TEXT}'>{check['car']}</div>"
                    f"<div style='display:inline-block; margin:0.5rem 0; padding:0.3rem 1rem; border-radius:999px; "
                    f"background:{background}; color:{ink}; font-weight:700; font-size:1.6rem; letter-spacing:0.05em'>"
                    f"{verdict.upper()}</div>", unsafe_allow_html=True)

        needle = float(np.clip(diff * 100, -GAUGE_END, GAUGE_END))
        fig = go.Figure(go.Indicator(
            mode="gauge", value=needle,
            gauge=dict(
                axis=dict(range=[-GAUGE_END, GAUGE_END], tickvals=[-GAUGE_END, -10, 0, 10, GAUGE_END],
                          ticktext=[f"−{GAUGE_END}%", "−10%", "0", "+10%", f"+{GAUGE_END}%"], tickcolor=GRAPHITE),
                bar=dict(color="rgba(0,0,0,0)", thickness=0),
                steps=[dict(range=[-GAUGE_END, -BAND * 100], color=GREEN), dict(range=[-BAND * 100, BAND * 100], color=GRAPHITE),
                       dict(range=[BAND * 100, GAUGE_END], color=RED)],
                threshold=dict(line=dict(color=TEXT, width=5), thickness=1, value=needle),
                bgcolor=PANEL, borderwidth=0,
            ),
        ))
        fig.update_layout(height=250, margin=dict(t=30, b=10, l=55, r=55))
        chart(fig)

        cards = st.columns(2)
        cards[0].metric("Fair price", f"${check['fair']:,.0f}", border=True,
                        help="What the model predicts for this car in the 2021 market.")
        cards[1].metric("Asking price", f"${check['asking']:,.0f}", delta=f"{diff:+.0%} vs fair price",
                        delta_color="off", delta_arrow="off", border=True)
        st.caption(f"Average = within ±{BAND:.0%} of the fair price: "
                   f"\\${check['fair'] * (1 - BAND):,.0f} to \\${check['fair'] * (1 + BAND):,.0f}.")

with st.expander("How this works"):
    st.markdown(
        f"- The fair price comes from the final CatBoost model (see the Model page), trained on {int(models['listings'].sum()):,} "
        "cleaned 2021 Craigslist listings. Its typical error is about 11%, so a price within ±10% of it is "
        "called average.\n"
        "- Picking a model pre-fills its most common type, drive, fuel, transmission and cylinders; change them if "
        "your car differs.\n"
        "- The model also uses things the form doesn't ask for: the ad's length is set to the median ad "
        f"({int(defaults['desc_len']):,} characters) and the location to the middle of the chosen state.\n"
        "- Age counts from 2021, the year of the data: a 2015 car is treated as 6 years old.\n"
        "- The prices are 2021 prices; used-car prices have moved since."
    )
