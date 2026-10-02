import pandas as pd
import streamlit as st

from shiftproof.config import DATA_DIR, REPORT_DIR
from shiftproof.predictor import Predictor

st.set_page_config(page_title="Placement Readiness", page_icon="🎓", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink: #18302b; --accent: #167d70; --soft: #e9f3ef; }
    h1, h2, h3 { color: var(--ink); }
    [data-testid="stMetric"] { background: var(--soft); color: var(--ink); padding: 14px 16px; border-radius: 6px; }
    [data-testid="stMetric"] * { color: var(--ink) !important; }
    [data-testid="stSidebar"] { border-right: 1px solid #dce6e2; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Placement Readiness Predictor")
st.caption("Estimate placement readiness from academics, skills, projects, coding practice, and internships.")

try:
    predictor = Predictor()
except Exception as exc:
    st.error(f"Model unavailable: {exc}")
    st.stop()

metrics_path = REPORT_DIR / "metrics.csv"
metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
students_path = DATA_DIR / "students.csv"
students = pd.read_csv(students_path) if students_path.exists() else pd.DataFrame()

heldout_f1 = None
if not metrics.empty and "f1" in metrics:
    final_rows = metrics[metrics["model"].astype(str).str.startswith("final:")]
    if not final_rows.empty:
        heldout_f1 = float(final_rows.iloc[0]["f1"])

kpis = st.columns(4)
kpis[0].metric("Student records", f"{len(students):,}" if not students.empty else "Not available")
kpis[1].metric("Placement-ready rate", f"{students['placement_ready'].mean():.1%}" if "placement_ready" in students else "Not available")
kpis[2].metric("Held-out F1", f"{heldout_f1:.3f}" if heldout_f1 is not None else "Not available")
model_labels = {
    "baseline_logistic": "Logistic",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Hist. gradient",
}
kpis[3].metric("Selected model", model_labels.get(predictor.bundle.model_name, predictor.bundle.model_name))

predict_tab, analytics_tab, batch_tab = st.tabs(["Student predictor", "Cohort analytics", "Batch CSV"])

with predict_tab:
    input_col, result_col = st.columns([1.15, 0.85], gap="large")
    with input_col:
        st.subheader("Student profile")
        fields_a, fields_b = st.columns(2)
        with fields_a:
            graduation_year = st.selectbox("Graduation year", [2024, 2025, 2026, 2027, 2028], index=2)
            branch = st.selectbox(
                "Degree branch",
                ["computer_science", "information_technology", "electronics", "mechanical", "civil"],
            )
            cgpa = st.number_input("CGPA (out of 10)", 0.0, 10.0, 7.5, 0.1)
            aptitude = st.number_input("Aptitude score (out of 100)", 0.0, 100.0, 65.0, 1.0)
            technical = st.number_input("Technical skills (out of 100)", 0.0, 100.0, 65.0, 1.0)
        with fields_b:
            communication = st.number_input("Communication (out of 100)", 0.0, 100.0, 65.0, 1.0)
            coding_hours = st.number_input("Coding practice (hours/week)", 0.0, 40.0, 10.0, 1.0)
            projects = st.number_input("Completed projects", 0, 10, 2)
            internships = st.number_input("Completed internships", 0, 5, 1)
            certifications = st.number_input("Certifications", 0, 12, 2)
            backlogs = st.number_input("Active or historical backlogs", 0, 8, 0)

    profile = {
        "graduation_year": graduation_year,
        "branch": branch,
        "cgpa": cgpa,
        "aptitude_score": aptitude,
        "technical_skills_score": technical,
        "communication_score": communication,
        "coding_hours_per_week": coding_hours,
        "projects_completed": projects,
        "internships_completed": internships,
        "certifications_count": certifications,
        "backlogs": backlogs,
    }
    result = predictor.predict_one(profile)
    score = float(result["readiness_score"])

    with result_col:
        st.subheader("Readiness estimate")
        st.metric("Placement readiness", f"{score:.0%}", result["readiness_category"])
        st.progress(score)
        st.caption(f"Model confidence: {result['confidence']:.0%} · {result['model_name'].replace('_', ' ').title()}")
        st.markdown("#### Profile signals")
        st.caption("Signals use transparent profile thresholds; they are not causal model explanations.")
        positive = []
        focus = []
        checks = [
            (cgpa >= 7.5, cgpa < 6.5, "Academic record", "CGPA"),
            (aptitude >= 65, aptitude < 45, "Aptitude", "Aptitude practice"),
            (technical >= 65, technical < 45, "Technical skills", "Technical skills"),
            (communication >= 70, communication < 45, "Communication", "Communication practice"),
            (coding_hours >= 10, coding_hours < 4, "Regular coding practice", "Consistent coding practice"),
            (projects >= 2, projects == 0, "Project experience", "Build a portfolio project"),
            (internships >= 1, internships == 0, "Internship experience", "Seek an internship"),
            (certifications >= 2, False, "Relevant certifications", ""),
            (backlogs == 0, backlogs >= 2, "No backlogs", "Clear outstanding backlogs"),
        ]
        for is_positive, needs_focus, strength, improvement in checks:
            if is_positive:
                positive.append(strength)
            if needs_focus:
                focus.append(improvement)
        if positive:
            st.success("Strengths: " + " · ".join(positive[:3]))
        if focus:
            st.warning("Focus areas: " + " · ".join(focus[:3]))
        if not focus:
            st.info("No major development gaps identified in the entered profile.")

    with st.expander("What-if scenario", expanded=True):
        st.caption("Adjust practical experience and see the model estimate update for this scenario.")
        whatif_cols = st.columns(3)
        with whatif_cols[0]:
            scenario_coding = st.slider("Coding hours per week", 0, 40, int(coding_hours), key="whatif_coding")
        with whatif_cols[1]:
            scenario_projects = st.slider("Projects completed", 0, 10, int(projects), key="whatif_projects")
        with whatif_cols[2]:
            scenario_internships = st.slider("Internships completed", 0, 5, int(internships), key="whatif_internships")
        scenario = profile | {
            "coding_hours_per_week": scenario_coding,
            "projects_completed": scenario_projects,
            "internships_completed": scenario_internships,
        }
        scenario_score = float(predictor.predict_one(scenario)["readiness_score"])
        delta = scenario_score - score
        st.metric("Scenario readiness", f"{scenario_score:.0%}", f"{delta:+.1%} vs. current profile")

with analytics_tab:
    st.subheader("Cohort overview")
    if students.empty:
        st.info("Student cohort data is not available. Run the training command to generate it.")
    else:
        filter_cols = st.columns(2)
        branch_options = ["All branches"] + sorted(students["branch"].dropna().unique().tolist())
        year_options = ["All cohorts"] + sorted(students["graduation_year"].dropna().astype(int).astype(str).unique().tolist())
        selected_branch = filter_cols[0].selectbox("Branch filter", branch_options)
        selected_year = filter_cols[1].selectbox("Graduation cohort", year_options)
        filtered = students.copy()
        if selected_branch != "All branches":
            filtered = filtered[filtered["branch"] == selected_branch]
        if selected_year != "All cohorts":
            filtered = filtered[filtered["graduation_year"].astype(int).astype(str) == selected_year]
        rate = float(filtered["placement_ready"].mean()) if not filtered.empty else 0.0
        chart_cols = st.columns(2)
        chart_cols[0].metric("Students in filter", f"{len(filtered):,}")
        chart_cols[1].metric("Ready in filter", f"{rate:.1%}")
        branch_rates = filtered.groupby("branch")["placement_ready"].mean().sort_values(ascending=False)
        st.markdown("#### Readiness by branch")
        st.bar_chart(branch_rates)
        if not metrics.empty and {"model", "f1", "pr_auc"}.issubset(metrics.columns):
            st.markdown("#### Model comparison")
            comparison = metrics.set_index("model")[["f1", "pr_auc"]]
            st.bar_chart(comparison)
        st.caption("Cohort outcomes are synthetic demonstration data, not employment guarantees.")

with batch_tab:
    st.subheader("Batch predictions")
    st.caption("Upload a CSV with one student per row and the same feature columns as the individual predictor.")
    upload = st.file_uploader("Student profile CSV", type=["csv"])
    if upload is not None:
        try:
            frame = pd.read_csv(upload)
            predictions = predictor.predict_frame(frame)
            st.dataframe(predictions, use_container_width=True, hide_index=True)
            st.download_button(
                "Download predictions",
                predictions.to_csv(index=False),
                file_name="placement_readiness_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")
