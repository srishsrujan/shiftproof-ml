import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

st.set_page_config(
    page_title="ShiftProof",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ShiftProof")
st.write("Loading dashboard...")

try:
    from shiftproof import dashboard
except Exception as exc:
    st.error("ShiftProof dashboard failed to load.")
    st.exception(exc)
    st.stop()