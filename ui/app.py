import sys
sys.path.append("..")
import streamlit as st
import pandas as pd
from pipeline import run_pipeline

st.title("AI Deployment Decision Engine")

file = st.file_uploader("Upload evaluation CSV")

if file:
    df = pd.read_csv(file)
    df["id"] = range(len(df))
    st.write(df.head())

    if st.button("Run Deployment Analysis"):
        results = run_pipeline(df)
        
        stats = results["basic_stats"]
        failure_pct = int(stats["failure_rate"] * 100)
        risk = "HIGH" if stats["failure_rate"] > 0.4 else "MODERATE" if stats["failure_rate"] > 0.2 else "LOW"
        top_domain = max(stats["domain_failure_rate"], key=stats["domain_failure_rate"].get)
        
        st.markdown(f"""
**Deployment Risk:** {risk}  
**Failure Rate:** {failure_pct}%  
**Highest Risk Domain:** {top_domain}
""")
        
        st.markdown("""
**Pipeline Steps:**
- ✓ Data Ingested
- ✓ Evaluation Metrics Computed
- ✓ Risk Patterns Identified
- ✓ Deployment Recommendations Generated
""")
        
        with st.expander("Advanced Debug Output"):
            st.write(results)
        
        st.subheader("Failure Clusters")
        for c in results["overlaps"]["failure"]:
            st.markdown(f"**{c['name']}** — {c['action']} (Risk: {c['risk_level']}, Score: {c['risk_score']:.2f})")
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
            st.markdown(f"**{c['name']}** (Confidence: {c['confidence_score']:.2f})")

        st.pyplot(results["embeddings_chart"])
        st.pyplot(results["failures_chart"])
        st.pyplot(results["latency_chart"])