import hashlib
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.config import CLASS_TO_INDEX, MODEL_PATH

@st.cache_resource
def load_cached_model():
    from src.config import get_device
    from src.inference import load_model

    return load_model(MODEL_PATH, get_device())

st.set_page_config(page_title="Roadwatch | Pothole analysis", page_icon="◉", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink: #192b2a; --muted: #607371; --paper: #f4f5ef; --line: #d9dfd7; --green: #176b55; --lime: #d8ee8d; }
    .stApp { background: var(--paper); color: var(--ink); font-family: 'Manrope', sans-serif; }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; }
    [data-testid="stHeader"] { background: transparent; }
    .eyebrow { font: 500 11px 'DM Mono', monospace; letter-spacing: 0; color: var(--green); text-transform: uppercase; }
    .masthead { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--line); padding-bottom: 17px; }
    .brand { font-size: 15px; font-weight: 800; letter-spacing: 0; }
    .hero { padding: 46px 0 28px; border-bottom: 1px solid var(--line); background: radial-gradient(ellipse at 84% 5%, #e3ebd8 0%, transparent 42%); }
    .hero h1 { font-size: 54px; line-height: 1.06; letter-spacing: 0; margin: 12px 0 10px; color: var(--ink); }
    .hero p { color: var(--muted); font-size: 15px; margin: 0; }
    .section-label { font: 500 11px 'DM Mono', monospace; color: var(--muted); text-transform: uppercase; letter-spacing: 0; padding-bottom: 11px; border-bottom: 1px solid var(--line); margin: 8px 0 18px; }
    .stButton > button { background: var(--green); color: white; border: 0; border-radius: 4px; min-height: 46px; font-weight: 700; }
    .stButton > button:hover { background: #10533f; color: white; border: 0; }
    [data-testid="stFileUploader"] { background: rgba(255,255,255,.48); border: 1px dashed #aab8ac; border-radius: 4px; padding: 14px; }
    [data-testid="stMetric"] { background: #e8ede3; border-left: 3px solid var(--green); padding: 14px 16px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    .stAlert { border-radius: 4px; }
    @media (max-width: 600px) { .hero h1 { font-size: 36px; } .masthead { align-items: flex-start; gap: 12px; } }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="masthead"><div class="brand">ROADWATCH <span style="color:#176b55">/</span> FIELD VISION</div>'
    '<div class="eyebrow">SURFACE CONDITION · IMAGE ANALYSIS</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<section class="hero"><div class="eyebrow">01 — INSPECTION</div>'
    '<h1>Find the fault.<br>See what the model sees.</h1>'
    '<p>Upload a road image for pothole classification and a Grad-CAM attention map.</p></section>',
    unsafe_allow_html=True,
)
st.caption(
    "The normal class means no pothole is marked in the source mask; some images may still contain cracks."
)

if not MODEL_PATH.exists():
    st.info(
        "The trained model checkpoint is missing. Add the dataset shown below and run `python train.py`."
    )
    with st.expander("Expected dataset layout and training command"):
        st.code(
            "dataset/\n"
            "  train/normal/\n"
            "  train/pothole/\n"
            "  val/normal/\n"
            "  val/pothole/\n\n"
            "python train.py",
            language="text",
        )

uploaded_file = st.file_uploader(
    "Choose a road image", type=("jpg", "jpeg", "png", "webp"),
    help="Use a clear road-surface image in JPG, PNG, or WebP format.",
)

if uploaded_file is not None:
    try:
        uploaded_bytes = uploaded_file.getvalue()
        if not uploaded_bytes:
            st.error("The uploaded file is empty. Choose a road image and try again.")
            st.stop()
        image_key = hashlib.sha256(uploaded_bytes).hexdigest()
        image = Image.open(BytesIO(uploaded_bytes)).convert("RGB")
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="section-label">ORIGINAL IMAGE</div>', unsafe_allow_html=True)
            st.image(image, width="stretch")
        with right:
            st.markdown('<div class="section-label">PREDICTION</div>', unsafe_allow_html=True)
            if MODEL_PATH.exists():
                if st.button("Analyze image", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Analyzing surface…"):
                            from src.gradcam import gradcam_overlay
                            from src.inference import predict_image

                            model = load_cached_model()
                            prediction = predict_image(image, model)
                            overlay = gradcam_overlay(
                                image, model, prediction["class_index"]
                            )
                        st.session_state["analysis"] = {
                            "image_key": image_key,
                            "prediction": prediction,
                            "overlay": overlay,
                        }
                    except FileNotFoundError:
                        st.error("The model checkpoint is missing. Run `python train.py` first.")
                    except OSError as error:
                        if getattr(error, "winerror", None) == 4551:
                            st.warning(
                                "Windows Application Control blocked a PyTorch DLL. Ask your IT administrator to approve this environment."
                            )
                        else:
                            st.error("Could not load the model or image. Check the checkpoint and runtime installation.")
                    except (RuntimeError, ValueError, KeyError, TypeError):
                        st.error("Image analysis failed. Check that the checkpoint is a compatible ResNet18 model.")
                analysis = st.session_state.get("analysis")
                if analysis and analysis["image_key"] == image_key:
                    prediction = analysis["prediction"]
                    pothole_detected = (
                        prediction["class_index"] == CLASS_TO_INDEX["pothole"]
                    )
                    if pothole_detected:
                        st.markdown("### POTHOLE DETECTED")
                        confidence_label = "Pothole Confidence"
                        explanation = (
                            "Highlighted regions indicate the areas that contributed "
                            "to the pothole prediction."
                        )
                    else:
                        st.markdown("### NO POTHOLE IDENTIFIED")
                        confidence_label = "Normal Road Confidence"
                        explanation = (
                            "Highlighted regions indicate the areas considered by the "
                            "model when identifying the road as normal."
                        )
                    st.metric(confidence_label, f"{prediction['confidence']:.1%}")
                    st.markdown('<div class="section-label">XAI / GRAD-CAM EXPLANATION</div>', unsafe_allow_html=True)
                    st.image(analysis["overlay"], width="stretch")
                    st.caption(explanation)
            else:
                st.caption("Train the model before requesting a prediction.")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This file could not be read as an image. Try a JPG, PNG, or WebP file.")

st.markdown("<div style='height:30px'></div><div class='eyebrow'>RESNET18 CLASSIFIER · GRAD-CAM · LOCAL INFERENCE</div>", unsafe_allow_html=True)