"""
ESG Greenwashing Detector — Final Dashboard
=============================================
Combines everything built so far into one interactive demo:
  1. Cross-Sector Comparison  -- your existing 9-report analysis
  2. Analyze a New Report     -- upload a PDF, score it live with v1
  3. v1 vs v2 Comparison      -- see rule-based vs ML predictions side by side

Requires batch_processor.py to be in the same folder (reuses its
extraction/scoring functions so logic isn't duplicated).

Run with:  streamlit run dashboard.py
"""

import os
import pickle
import pandas as pd
import streamlit as st

from batch_processor import (
    extract_text_from_pdf, split_into_sentences, score_sentence
)

st.set_page_config(page_title="ESG Greenwashing Detector", layout="wide")

MODEL_DIR = os.path.join("output", "model")
SECTOR_SUMMARY = os.path.join("output", "esg_summary_by_sector.csv")
REPORT_SUMMARY = os.path.join("output", "esg_summary_by_report.csv")


@st.cache_resource
def load_ml_model():
    """Loads the trained v2 classifier + vectorizer, if available."""
    model_path = os.path.join(MODEL_DIR, "classifier.pkl")
    vec_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        return None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def specificity_risk_label(pct):
    if pct < 40:
        return "🔴 High greenwashing risk", "red"
    elif pct < 65:
        return "🟡 Moderate risk", "orange"
    else:
        return "🟢 Low risk", "green"


st.title("🌱 ESG Greenwashing Detector")
st.caption(
    "Detects vague, unverifiable ESG language vs. specific, checkable claims."
)

tab1, tab2, tab3 = st.tabs([
    "📊 Cross-Sector Comparison", "📄 Analyze a Report", "🤖 v1 vs v2 Comparison"
])

# ---------------------------------------------------------------------------
# TAB 1 — Cross-Sector Comparison (uses your existing batch results)
# ---------------------------------------------------------------------------
with tab1:
    st.header("How specific are ESG reports, by sector?")

    if os.path.exists(SECTOR_SUMMARY):
        sector_df = pd.read_csv(SECTOR_SUMMARY)
        sector_df = sector_df.sort_values("specificity_pct", ascending=False)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(sector_df.set_index("sector")["specificity_pct"])
        with col2:
            st.dataframe(sector_df, use_container_width=True, hide_index=True)

        best = sector_df.iloc[0]
        worst = sector_df.iloc[-1]
        st.info(
            f"**{best['sector']}** is the most specific sector "
            f"({best['specificity_pct']}% concrete language), while "
            f"**{worst['sector']}** is the most vague "
            f"({worst['specificity_pct']}% concrete language). "
            f"Even the best sector backs less than "
            f"{int(best['specificity_pct'] // 10 * 10) + 10}% of its ESG "
            f"claims with verifiable data."
        )
    else:
        st.warning(
            f"Could not find `{SECTOR_SUMMARY}`. "
            f"Run `python batch_processor.py` first to generate it."
        )

    if os.path.exists(REPORT_SUMMARY):
        with st.expander("See breakdown by individual report"):
            report_df = pd.read_csv(REPORT_SUMMARY)
            st.dataframe(report_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 2 — Analyze a new report live (rule-based v1)
# ---------------------------------------------------------------------------
with tab2:
    st.header("Upload a new ESG report to score it")

    uploaded_file = st.file_uploader("Upload a PDF report", type=["pdf"])
    pasted_text = st.text_area(
        "...or paste text directly (useful for testing a specific paragraph)",
        height=150
    )

    if st.button("Analyze"):
        if uploaded_file is not None:
            temp_path = os.path.join("output", "_temp_upload.pdf")
            os.makedirs("output", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            raw_text = extract_text_from_pdf(temp_path)
            os.remove(temp_path)
        elif pasted_text.strip():
            raw_text = pasted_text
        else:
            st.warning("Please upload a PDF or paste some text first.")
            raw_text = None

        if raw_text:
            sentences = split_into_sentences(raw_text)
            if not sentences:
                st.warning(
                    "No ESG-relevant sentences found. This could mean the "
                    "document has no extractable text, or doesn't contain "
                    "ESG-related content."
                )
            else:
                rows = []
                for s in sentences:
                    verdict, score, reason = score_sentence(s)
                    rows.append({"sentence": s, "verdict": verdict, "reason": reason})
                result_df = pd.DataFrame(rows)

                total = len(result_df)
                concrete = (result_df["verdict"] == "Concrete").sum()
                specificity_pct = round((concrete / total) * 100, 1) if total else 0
                risk_label, color = specificity_risk_label(specificity_pct)

                col1, col2, col3 = st.columns(3)
                col1.metric("Sentences analyzed", total)
                col2.metric("Specificity score", f"{specificity_pct}%")
                col3.markdown(f"### {risk_label}")

                verdict_filter = st.multiselect(
                    "Filter by verdict",
                    options=result_df["verdict"].unique().tolist(),
                    default=result_df["verdict"].unique().tolist()
                )
                filtered = result_df[result_df["verdict"].isin(verdict_filter)]
                st.dataframe(filtered, use_container_width=True, hide_index=True)

                csv = filtered.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV", csv,
                    "esg_analysis_results.csv", "text/csv"
                )


# ---------------------------------------------------------------------------
# TAB 3 — v1 (rule-based) vs v2 (ML) side-by-side comparison
# ---------------------------------------------------------------------------
with tab3:
    st.header("Compare rule-based v1 vs ML classifier v2")

    model, vectorizer = load_ml_model()

    if model is None:
        st.warning(
            "No trained ML model found. Run `python train_classifier.py` "
            "first to train and save the v2 model."
        )
    else:
        st.success("ML model (v2) loaded successfully.")
        test_sentence = st.text_input(
            "Type or paste a sentence to compare both models:",
            value="We are committed to a greener future for all."
        )

        if test_sentence:
            v1_verdict, v1_score, v1_reason = score_sentence(test_sentence)
            v2_pred = model.predict(vectorizer.transform([test_sentence]))[0]
            v2_proba = model.predict_proba(vectorizer.transform([test_sentence]))[0]
            v2_confidence = max(v2_proba) * 100

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("v1: Rule-Based")
                st.metric("Verdict", v1_verdict)
                st.caption(f"Reason: {v1_reason}")
            with col2:
                st.subheader("v2: ML Classifier")
                st.metric("Verdict", v2_pred)
                st.caption(f"Confidence: {v2_confidence:.1f}%")

            if v1_verdict == v2_pred:
                st.success("Both models agree.")
            else:
                st.warning(
                    "Models disagree -- a good example to discuss in your "
                    "viva about the trade-offs between approaches."
                )
