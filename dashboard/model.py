import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from common import AMBER, FONT, SLATE, TEAL, TEXT, chart, load, panel

NOTEBOOK = "05_final_model.ipynb"
GRID = "rgba(100, 116, 139, 0.25)"
MODELS = {"CatBoostRegressor": "CatBoost", "LinearRegression (baseline)": "Baseline (linear regression)"}

# readable names for the model's input columns (see num_cols / cat_features in src/carpricepredictor/shared.py)
FEATURES = {
    "age": "age", "log_odometer": "mileage", "miles_per_year": "miles per year", "cylinders_num": "cylinders",
    "is_classic": "classic (before 1995)", "is_dealer": "dealer listing", "has_vin": "VIN shown",
    "desc_len": "description length", "lat": "latitude", "long": "longitude", "needs_work": "ad: needs work",
    "high_trim": "ad: high trim", "lifted": "ad: lifted", "diesel_words": "ad: diesel", "one_owner": "ad: one owner",
    "carfax": "ad: carfax", "warranty": "ad: warranty", "leather": "ad: leather", "sunroof": "ad: sunroof",
    "navigation": "ad: navigation", "turbo": "ad: turbo", "title_status": "title status", "paint_color": "paint colour",
}

st.title("Model")

scores = load("scores.csv", NOTEBOOK).assign(model=lambda d: d["model"].map(MODELS))
preds = load("test_predictions.csv", NOTEBOOK)
importance = load("feature_importance.csv", NOTEBOOK)

test = scores[scores["data"] == "test"].set_index("model")
cat, base = test.loc["CatBoost"], test.loc["Baseline (linear regression)"]

st.caption(f"CatBoost, scored on {len(preds):,} test cars it never saw during training, against a linear "
           "regression baseline. After this test it was retrained on all the data for the price checker.")

cards = st.columns(3)
cards[0].metric("Typical error", f"{cat['median_%_error']:.1f}%", delta=f"baseline: {base['median_%_error']:.1f}%",
                delta_color="off", delta_arrow="off", border=True,
                help="Median % error: half the test cars are predicted closer than this to their real price.")
miss = (preds["pred_catboost"] - preds["price"]).abs()
cards[1].metric("Average miss (up or down)", f"${cat['mae_$']:,.0f}", delta=f"baseline: ${base['mae_$']:,.0f}",
                delta_color="off", delta_arrow="off", border=True,
                help=f"How far the prediction is from the real price on average, whether too high or too low. "
                     f"Not a ± range: {(miss <= cat['mae_$']).mean():.0%} of test cars are closer than this, "
                     f"half are within ${miss.median():,.0f}. Bigger for expensive cars, smaller for cheap ones.")
cards[2].metric("Price differences explained", f"{cat['r2']:.0%}", delta=f"baseline: {base['r2']:.0%}",
                delta_color="off", delta_arrow="off", border=True,
                help="R² on log-price: the share of the differences between car prices the model accounts for.")


# ---------- train vs test ----------
st.subheader("Train vs test")
st.caption("The same scores on the cars the model trained on and on the test cars. A big gap means the model "
           "memorises its training cars: CatBoost does, a lot, but it still beats the baseline clearly on new cars.")

with panel("Scores on the training and the test set"):
    table = scores.rename(columns={"model": "Model", "data": "Data", "rmse_log": "RMSE (log)", "mae_log": "MAE (log)",
                                   "r2": "R²", "mae_$": "MAE ($)", "median_%_error": "Median error (%)"})
    table["MAE ($)"] = table["MAE ($)"].round().astype(int)
    st.dataframe(table, hide_index=True, column_config={
        "RMSE (log)": st.column_config.NumberColumn(format="%.3f"),
        "MAE (log)": st.column_config.NumberColumn(format="%.3f"),
        "R²": st.column_config.NumberColumn(format="%.3f"),
        "MAE ($)": st.column_config.NumberColumn(format="localized"),
        "Median error (%)": st.column_config.NumberColumn(format="%.1f"),
    })

with st.expander("How this works"):
    st.markdown(
        "- The cleaned data is split once: 80% to train on (164,308 cars), 20% held back to test (41,077 cars).\n"
        "- The models predict log(1 + price), so an error of 0.1 in log-price is about 10% off.\n"
        "- **RMSE / MAE (log)**: root mean squared / mean absolute error of log-price. RMSE punishes big misses more.\n"
        "- **R²**: share of the variance of log-price explained (1 = perfect).\n"
        "- **MAE ($)** and **median error (%)**: the same errors turned back into dollars.\n"
        "- The test set is scored once, after all the choices were made on the training set."
    )


# ---------- predictions vs real prices ----------
st.subheader("Predictions vs real prices")
st.caption("Left: each dot is a test car, at its real price and CatBoost's prediction; on the line = a perfect "
           "prediction. Right: how far off the predictions are, for both models.")

left, right = st.columns(2)
with left, panel("Predicted vs real price, 5,000 test cars"):
    s = preds.sample(5_000, random_state=0)
    hover = (s["manufacturer"] + " " + s["model"] + " · " + (2021 - s["age"]).astype(str) + " · "
             + s["odometer"].map("{:,.0f} mi".format) + "<br>real $" + s["price"].map("{:,.0f}".format)
             + " · predicted $" + s["pred_catboost"].map("{:,.0f}".format))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[400, 250_000], y=[400, 250_000], mode="lines", line=dict(color=SLATE, width=1.5),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattergl(x=s["price"], y=s["pred_catboost"], mode="markers", showlegend=False,
                               marker=dict(color=TEAL, size=5, opacity=0.45),
                               text=hover, hovertemplate="%{text}<extra></extra>"))
    ticks = [1e3, 3e3, 1e4, 3e4, 1e5]
    ticktext = ["$1k", "$3k", "$10k", "$30k", "$100k"]
    fig.update_xaxes(type="log", title="real price", tickvals=ticks, ticktext=ticktext, range=[np.log10(400), np.log10(250_000)])
    fig.update_yaxes(type="log", title="predicted price", tickvals=ticks, ticktext=ticktext,
                     range=[np.log10(400), np.log10(250_000)], gridcolor=GRID)
    fig.update_layout(height=460)
    chart(fig)

with right, panel("Error spread, all test cars"):
    # one small histogram per model, same axes, so the shapes compare without overlapping
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=("CatBoost", "Baseline"))
    for col, (column, colour, name) in enumerate([("pred_catboost", TEAL, "CatBoost"),
                                                  ("pred_baseline", AMBER, "Baseline")], start=1):
        error = preds[column] / preds["price"] - 1
        fig.add_trace(go.Histogram(
            x=error.clip(-1, 1), xbins=dict(start=-1, end=1.0001, size=0.04), name=name, showlegend=False,
            marker=dict(color=colour), hovertemplate=name + ": %{y:,} cars<extra></extra>",
        ), row=1, col=col)
        fig.add_vline(x=0, line=dict(color=TEXT, width=1), row=1, col=col)
    fig.update_xaxes(title="prediction vs real price", tickformat="+.0%", range=[-1, 1], tickvals=[-0.5, 0, 0.5])
    fig.update_yaxes(tickformat=",", gridcolor=GRID)
    fig.update_yaxes(title="cars", row=1, col=1)
    fig.update_annotations(font=dict(family=FONT, color=TEXT, size=13))
    fig.update_layout(height=460, bargap=0.05, margin=dict(t=40))
    chart(fig)

far_cat = (preds["pred_catboost"] / preds["price"] - 1).abs().gt(1).mean()
far_base = (preds["pred_baseline"] / preds["price"] - 1).abs().gt(1).mean()
with st.expander("How this works"):
    st.markdown(
        "- **Left**: 5,000 random test cars; both axes on a log scale so cheap and expensive cars fit. "
        "Dots above the line were over-priced by the model, dots below under-priced.\n"
        "- **Right**: prediction ÷ real price − 1 for every test car. 0% = exact, +20% = predicted 20% too high. "
        f"Errors beyond ±100% are counted in the end bars ({far_cat:.1%} of cars for CatBoost, {far_base:.1%} "
        "for the baseline): mostly listings with a strange asking price."
    )


# ---------- feature importance ----------
st.subheader("What the price depends on")
st.caption("How much each input column moves the final model's predictions, as a share of the total (all 31 add "
           "up to 100%). The 15 biggest.")

with panel("Feature importance, top 15"):
    top = importance.head(15).assign(label=lambda d: d["feature"].map(FEATURES).fillna(d["feature"]))
    fig = go.Figure(go.Bar(
        x=top["importance"], y=top["label"], orientation="h", marker_color=TEAL,
        text=[f"{v:.1f}%" for v in top["importance"]], textposition="outside", textfont=dict(color=TEXT),
        customdata=top["feature"], hovertemplate="%{y} (%{customdata}): %{x:.1f}%<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=GRID, range=[0, top["importance"].max() * 1.15])
    fig.update_layout(height=32 * len(top) + 50, bargap=0.35)
    chart(fig)

with st.expander("How this works"):
    st.markdown(
        "- CatBoost's own importance (PredictionValuesChange): how much the predictions change, on average, "
        "when a column's value changes.\n"
        "- Columns that go through one-hot encoding (manufacturer, model, state, ...) are split into many 0/1 columns "
        "inside the model; their importances are summed back into the column they came from.\n"
        "- Importance says what the model *uses*, not cause and effect: age and mileage overlap, so they share credit."
    )
