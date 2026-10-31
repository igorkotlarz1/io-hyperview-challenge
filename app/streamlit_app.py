"""AgroSpectral Analyzer: a Streamlit front end for trained HYPERVIEW models.

Run from the repository root, after `python -m hyperview submit` has saved a model:
    streamlit run app/streamlit_app.py

Original app by @KamiLMusiol, ported to the hyperview package.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow running without installing the package

from hyperview.config import Config  # noqa: E402
from hyperview.data import load_image, save_submission  # noqa: E402
from hyperview.features import build_feature_matrix, extract_features, feature_names  # noqa: E402
from hyperview.models import SoilModel  # noqa: E402

st.set_page_config(page_title="AgroSpectral Analyzer", layout="wide", page_icon="🌱")

BASE_CFG = Config()
MODELS_DIR = BASE_CFG.output_dir / "models"
UNITS = {"P": "mg/100 g (P₂O₅)", "K": "mg/100 g", "Mg": "mg/100 g", "pH": ""}


@st.cache_resource
def load_model(path: str) -> SoilModel:
    return SoilModel.load(BASE_CFG, Path(path))


def read_upload(uploaded) -> np.ma.MaskedArray:
    return load_image(io.BytesIO(uploaded.getvalue()))


def predict_uploads(model: SoilModel, uploads) -> pd.DataFrame:
    rows, progress = [], st.progress(0.0)
    for i, uploaded in enumerate(uploads, start=1):
        try:
            features = extract_features(read_upload(uploaded), model.cfg)
            prediction = model.predict(features[None, :])[0]
            rows.append({"file": uploaded.name, **dict(zip(model.cfg.targets, prediction))})
        except Exception as exc:  # noqa: BLE001 - one bad file should not stop the batch
            st.error(f"{uploaded.name}: {exc}")
        progress.progress(i / len(uploads))
    return pd.DataFrame(rows)


def pdf_report(results: pd.DataFrame) -> bytes | None:
    """PDF table of the results, or None if fpdf2 is not installed."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Soil analysis report (hyperspectral)", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "B", 9)
    widths = [70] + [30] * (len(results.columns) - 1)
    for width, column in zip(widths, results.columns):
        pdf.cell(width, 8, str(column), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for _, row in results.iterrows():
        for width, value in zip(widths, row.values):
            pdf.cell(width, 7, f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value), border=1)
        pdf.ln()
    return bytes(pdf.output())


def farmer_panel(model: SoilModel) -> None:
    st.header("🚜 Soil analysis")
    uploads = st.file_uploader("Upload field patches (.npz)", type=["npz"], accept_multiple_files=True)
    if not uploads or not st.button("Run analysis", type="primary"):
        return
    with st.spinner("Processing images..."):
        results = predict_uploads(model, uploads)
    if results.empty:
        return
    st.success(f"Analysed {len(results)} sample(s)")
    st.dataframe(results.style.format({t: "{:.2f}" for t in model.cfg.targets}), use_container_width=True)
    left, right = st.columns(2)
    left.download_button("Download CSV", results.to_csv(index=False).encode(), "soil_results.csv", "text/csv")
    pdf = pdf_report(results)
    if pdf:
        right.download_button("Download PDF", pdf, "soil_report.pdf", "application/pdf")
    else:
        right.caption("Install fpdf2 for PDF reports")


def scientist_panel(model: SoilModel) -> None:
    st.header("🔬 Scientist panel")
    explore, selection, submission = st.tabs(["Data exploration", "Feature selection", "Batch submission"])
    cfg = model.cfg

    with explore:
        uploaded = st.file_uploader("Upload one patch (.npz)", type=["npz"], key="explore")
        if uploaded:
            cube = read_upload(uploaded)
            st.caption(f"{cube.shape[0]} bands × {cube.shape[1]} × {cube.shape[2]} px, "
                       f"{int((~np.ma.getmaskarray(cube).any(axis=0)).sum())} field pixels")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(cube.data.mean(axis=(1, 2)), "--", label="all pixels")
            ax.plot(np.ma.getdata(cube.mean(axis=(1, 2))), label="field pixels")
            ax.set_xlabel("band")
            ax.set_ylabel("reflectance")
            ax.legend()
            st.pyplot(fig)

    with selection:
        names = feature_names(cfg)
        selected = model.selected_features(names)
        st.write(f"Selector: **{cfg.selector}**, {cfg.n_features} features per target, "
                 f"from groups {', '.join(cfg.feature_groups)} ({len(names)} features)")
        for target, chosen in selected.items():
            with st.expander(f"{target}: {len(chosen)} features", expanded=target == cfg.targets[0]):
                selector = model.pipelines[target].named_steps["select"]
                scores = getattr(getattr(selector, "estimator_", None), "feature_importances_", None)
                if scores is None and hasattr(selector, "ranking_"):
                    scores = 1.0 / selector.ranking_
                if scores is not None:
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.bar(range(len(scores)), scores, width=1.0)
                    ax.set_xlabel("feature index")
                    ax.set_ylabel("importance" if hasattr(selector, "estimator_") else "1 / rank")
                    st.pyplot(fig)
                st.code(", ".join(chosen), language=None)

    with submission:
        st.write(f"Predicts all images in `{cfg.test_images_dir}` and writes a challenge submission.")
        if st.button("Generate submission"):
            with st.spinner("Extracting features and predicting..."):
                X_test, _ = build_feature_matrix(cfg, "test")
                path = save_submission(model.predict(X_test), cfg, f"app_{cfg.model_type}")
            st.success(f"Saved {path}")
            st.download_button("Download CSV", path.read_bytes(), path.name, "text/csv")


def main() -> None:
    st.sidebar.title("🌱 AgroSpectral Analyzer")
    model_files = sorted(MODELS_DIR.glob("*.joblib"))
    if not model_files:
        st.title("AgroSpectral Analyzer")
        st.error(f"No trained model in `{MODELS_DIR}`. Train one first: `python -m hyperview submit`")
        return

    choice = st.sidebar.selectbox("Model", model_files, format_func=lambda p: p.name)
    model = load_model(str(choice))
    st.sidebar.success(f"Loaded **{model.cfg.model_type.upper()}** · {', '.join(model.cfg.targets)}")
    role = st.sidebar.radio("Mode", ["Farmer", "Scientist"])
    farmer_panel(model) if role == "Farmer" else scientist_panel(model)


main()
