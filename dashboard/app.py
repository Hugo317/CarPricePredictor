"""Fair Car Price dashboard. Run from the repo root: uv run streamlit run dashboard/app.py"""

from pathlib import Path

import streamlit as st

from common import style

HERE = Path(__file__).parent

st.set_page_config(page_title="Fair Car Price", page_icon=":material/directions_car:", layout="wide")
st.logo(str(HERE / "logo.svg"), size="large")
style()

pages = [
    st.Page("home.py", title="Home", default=True),
    st.Page("cleaning.py", title="Cleaning"),
    st.Page("duplicates.py", title="Duplicates"),
    st.Page("model.py", title="Model"),
    st.Page("price_check.py", title="Price check"),
]
st.navigation(pages, position="top").run()
