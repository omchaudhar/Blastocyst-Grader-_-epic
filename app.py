from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blastocyst_grader.predict import predict_with_checkpoint
from blastocyst_grader.research_sources import RESEARCH_SOURCES
from blastocyst_grader.rule_based import grade_image_rule_based
from blastocyst_grader.taxonomy import EXPANSION_DESCRIPTIONS, ICM_DESCRIPTIONS, TE_DESCRIPTIONS


st.set_page_config(page_title="Blastocyst Grading", page_icon="🔬", layout="wide")

st.title("Human Blastocyst Grading (Research Use)")
st.caption(
    "For research/education only. Not a medical device. "
    "Do not use as a standalone clinical decision tool."
)

with st.sidebar:
    st.header("Inference Mode")
    mode = st.radio(
        "Select grading mode",
        ["Rule-based baseline", "Trained checkpoint"],
        help="Rule-based works without weights; trained mode requires a PyTorch checkpoint.",
    )

    checkpoint_path = None
    if mode == "Trained checkpoint":
        checkpoint_path = st.text_input("Checkpoint path", value="models/best_model.pt")

uploaded = st.file_uploader("Upload a blastocyst image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(image, caption="Input image", use_container_width=True)

    with col2:
        if st.button("Run Grading", type="primary"):
            with st.spinner("Computing grade..."):
                if mode == "Rule-based baseline":
                    result = grade_image_rule_based(image).to_dict()
                else:
                    try:
                        result = predict_with_checkpoint(image, checkpoint_path)
                    except FileNotFoundError as exc:
                        st.error(str(exc))
                        st.stop()

            st.subheader(f"Predicted Grade: {result['grade']}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Expansion", result["expansion"])
            m2.metric("ICM", result["icm"])
            m3.metric("TE", result["te"])
            m4.metric("Confidence", f"{result['confidence']:.2f}")

            st.write(f"Quality band: **{result['quality_band']}**")

            st.markdown("### Interpretation")
            st.write(f"- Expansion: {result['explanation']['expansion']}")
            st.write(f"- ICM: {result['explanation']['icm']}")
            st.write(f"- TE: {result['explanation']['te']}")

            if result["mode"] == "rule_based":
                st.markdown("### Heuristic Features")
                st.dataframe(pd.DataFrame([result["features"]]), use_container_width=True)

with st.expander("Show Grading Taxonomy"):
    st.markdown("#### Expansion")
    st.table(pd.DataFrame([{"Grade": k, "Meaning": v} for k, v in EXPANSION_DESCRIPTIONS.items()]))

    st.markdown("#### Inner Cell Mass (ICM)")
    st.table(pd.DataFrame([{"Grade": k, "Meaning": v} for k, v in ICM_DESCRIPTIONS.items()]))

    st.markdown("#### Trophectoderm (TE)")
    st.table(pd.DataFrame([{"Grade": k, "Meaning": v} for k, v in TE_DESCRIPTIONS.items()]))

with st.expander("Show Research Sources"):
    st.dataframe(pd.DataFrame(RESEARCH_SOURCES), use_container_width=True)
