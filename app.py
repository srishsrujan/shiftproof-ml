import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

import streamlit as st

st.set_page_config(
    page_title="ShiftProof",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ShiftProof")

try:
    exec((SRC / "shiftproof" / "dashboard.py").read_text(encoding="utf-8"))
except Exception as exc:
    st.error("Dashboard error")
    st.exception(exc)