import streamlit as st

from common import GOLD, SUBTEXT, TEXT, load

# US market size: Cox Automotive's 2021 count, the same year as the Craigslist data
US_USED_SALES_2021 = 40_900_000
SOURCE = ("https://www.carscoops.com/2022/01/an-all-time-record-40-9-million-used-vehicles-were-sold-in-the-u-s-last-year/")

steps = load("cleaning_steps.csv", "11_dashboard_cleaning.ipynb")
scores = load("scores.csv", "05_final_model.ipynb")
raw, clean = steps["rows_before"].iloc[0], steps["rows_after"].iloc[-1]
typical_error = scores.query("model == 'CatBoostRegressor' and data == 'test'")["median_%_error"].iloc[0]

# ---------- hero ----------
st.markdown(
    f"<div style='max-width:48rem; padding:3rem 0 1.5rem'>"
    f"<div style='font-size:3rem; font-weight:700; line-height:1.15; color:{TEXT}'>"
    f"Have you ever thought about selling or buying a used car?</div>"
    f"<p style='font-size:1.25rem; line-height:1.6; color:{SUBTEXT}; margin-top:1.5rem'>"
    f"In 2021, <b style='color:{TEXT}'>{US_USED_SALES_2021 / 1e6:.1f} million</b> used cars were sold in the US: "
    f"about <b style='color:{TEXT}'>{US_USED_SALES_2021 / 365 / 1000:,.0f},000 every day</b>. "
    f"Searching for the right one at the right price is exhausting.</p>"
    f"<p style='font-size:1.25rem; line-height:1.6; color:{TEXT}'>"
    f"This tool helps you find the <b style='color:{GOLD}'>fair price</b>, and tells you whether a car is "
    f"cheap, average or expensive.</p></div>",
    unsafe_allow_html=True,
)
if st.button("Check a price →", type="primary"):
    st.switch_page("price_check.py")
st.caption(f"US sales figure: Cox Automotive, via [Carscoops]({SOURCE}).")

# ---------- numbers ----------
st.write("")
cards = st.columns(3)
cards[0].metric("Craigslist listings analysed", f"{raw:,}", border=True,
                help="Every used-car listing on US Craigslist between April 4 and May 5, 2021.")
cards[1].metric("Clean listings it learned from", f"{clean:,}", border=True,
                help="After removing fake prices, missing values and duplicates: see the Cleaning page.")
cards[2].metric("Typical error", f"{typical_error:.1f}%", border=True,
                help="Half of the test cars are predicted closer than this to their real price: see the Model page.")

# ---------- how it works ----------
st.subheader("How it works")
STEPS = [
    ("Enter your car", "Make, model, year, mileage, condition, and what the ad says."),
    ("Get the fair price", f"A model trained on {clean:,} real listings predicts what a car like yours sells for."),
    ("See the verdict", "Cheap, average or expensive: within ±10% of the fair price counts as average."),
]
for column, (number, (title, text)) in zip(st.columns(3), enumerate(STEPS, start=1)):
    with column.container(border=True):
        st.markdown(f"<div style='font-size:2rem; font-weight:700; color:{GOLD}'>{number}</div>"
                    f"<div style='font-weight:700; margin:0.25rem 0'>{title}</div>"
                    f"<div style='color:{SUBTEXT}'>{text}</div>", unsafe_allow_html=True)

# ---------- disclaimer ----------
st.write("")
st.divider()
st.caption("**Disclaimer:** this tool is for information only and is by no means financial advice. The prices are "
           "estimates from 2021 US Craigslist listings. We are not liable for any sales or purchases you may make.")
