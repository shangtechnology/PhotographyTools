import streamlit as st
import traceback

st.set_page_config(page_title="Photographer Weather Dashboard (AU)", layout="wide")
st.write("✅ Phase 0: Streamlit started")

# Stage imports so we can see where it dies
try:
    st.write("⏳ Phase 1: importing pandas...")
    import pandas as pd
    st.write("✅ Phase 1 OK")

    st.write("⏳ Phase 2: importing plotly...")
    import plotly.graph_objects as go
    st.write("✅ Phase 2 OK")

    st.write("⏳ Phase 3: importing requests/certifi...")
    import requests
    import certifi
    st.write("✅ Phase 3 OK")

except Exception:
    st.error("❌ Import failed")
    st.code(traceback.format_exc())
    st.stop()

st.write("✅ All staged imports passed")
st.stop()
