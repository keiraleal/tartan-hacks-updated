import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import pandas as pd
from pipeline import run_pipeline
from input_router import route_input

st.title("AI Deployment Decision Engine")

def display_results(results):
    stats = results["basic_stats"]
    failure_pct = int(stats["failure_rate"] * 100)
    risk = "HIGH" if stats["failure_rate"] > 0.4 else "MODERATE" if stats["failure_rate"] > 0.2 else "LOW"
    top_domain = max(stats["domain_failure_rate"], key=stats["domain_failure_rate"].get)

    c1, c2, c3 = st.columns(3)
    c1.metric("Deployment Risk", risk)
    c2.metric("Failure Rate", f"{failure_pct}%")
    c3.metric("Highest Risk Domain", top_domain)

    with st.expander("Pipeline Steps"):
        st.markdown("""
- ✓ Data Ingested
- ✓ Embeddings Generated
- ✓ UMAP Dimensionality Reduction
- ✓ Cluster Analysis (HDBSCAN)
- ✓ Evaluation Metrics Computed
- ✓ Risk Scoring & Ranking
- ✓ Risk Patterns Identified
- ✓ Deployment Recommendations Generated
""")

    with st.expander("Advanced Debug Output"):
        st.write(results)

    st.subheader("Failure Clusters")
    for c in results["overlaps"]["failure"]:
        with st.container():
            if c["risk_level"] == "HIGH":
                st.error(f"**{c['name']}** — {c['action']} (Risk: {c['risk_level']}, Score: {c['risk_score']:.2f})")
            elif c["risk_level"] == "MODERATE":
                st.warning(f"**{c['name']}** — {c['action']} (Risk: {c['risk_level']}, Score: {c['risk_score']:.2f})")
            else:
                st.info(f"**{c['name']}** — {c['action']} (Risk: {c['risk_level']}, Score: {c['risk_score']:.2f})")
            if c["action_why"]:
                st.markdown("Why: " + "; ".join(c["action_why"]))
            if c["controls"]:
                st.markdown("Controls: " + ", ".join(c["controls"]))
            if c["operational_adj"]:
                st.markdown("Operational: " + ", ".join(c["operational_adj"]))
            if c["operational_why"]:
                st.markdown("Why: " + "; ".join(c["operational_why"]))
            if c["confidence_adj"]:
                st.write(c["confidence_adj"])

    st.subheader("Success Clusters")
    for c in results["overlaps"]["success"]:
        st.success(f"**{c['name']}** (Confidence: {c['confidence_score']:.2f})")

    with st.expander("Charts"):
        st.pyplot(results["embeddings_chart"])
        st.pyplot(results["failures_chart"])
        st.pyplot(results["latency_chart"])

col1, col2 = st.columns(2)

with col1:
    file_a = st.file_uploader("Upload CSV A", key="a")
with col2:
    file_b = st.file_uploader("Upload CSV B", key="b")

if file_a and file_b:
    df_a = pd.read_csv(file_a)
    df_a["id"] = range(len(df_a))
    df_b = pd.read_csv(file_b)
    df_b["id"] = range(len(df_b))

    if st.button("Run Deployment Analysis"):
        with st.spinner("Running pipeline..."):
            st.session_state["results_a"] = run_pipeline(df_a)
            st.session_state["results_b"] = run_pipeline(df_b)

    if "results_a" in st.session_state and "results_b" in st.session_state:
        user_input = st.text_input("Test a Model Prompt")
        if user_input:
            msg, score_a, score_b = route_input(user_input, st.session_state["results_a"], st.session_state["results_b"])
            st.markdown(f"**{msg}**")
            st.caption(f"{score_a} · {score_b}")
        st.divider()
        tab1, tab2 = st.tabs(["Model A", "Model B"])
        with tab1:
            display_results(st.session_state["results_a"])
        with tab2:
            display_results(st.session_state["results_b"])
