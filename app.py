import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from pathlib import Path
from io import BytesIO

# Load model
MODEL_PATH = Path("artifacts/training/model.h5")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Load sample image
SAMPLE_PATH = Path("static/sample_ct.png")  # Make sure you have this image
sample_img = Image.open(SAMPLE_PATH)

# UI setup
st.set_page_config(page_title="Kidney Disease Detector", page_icon="🩺", layout="centered")

st.markdown("<h3 style='text-align:center;'> Kidney Disease Detection</h3>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size:15px'>
Upload a renal image and this model will predict if there is a risk of <strong>Kidney Disease</strong>.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your kidney scan (JPG, PNG)", type=["jpg", "jpeg", "png"])


# Download sample image
st.markdown("#### 🧪 Don't have a scan? Try with this sample CT image:")
col1, col2 = st.columns([1,1])
with col1:
    st.image(sample_img, caption="Sample Kidney CT", width=200)
with col2:
    buf = BytesIO()
    sample_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="⬇️ Download Sample CT",
        data=byte_im,
        file_name="sample_kidney_ct.png",
        mime="image/png"
    )

# Preprocessing function
def preprocess(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = keras_image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return img, arr

st.warning("Tip: This tool works best on kidney scans only")

# Prediction
if uploaded_file:
    image_bytes = uploaded_file.read()
    st.image(uploaded_file, caption="🖼️ Uploaded Image", width=300)

    with st.spinner("🔍 Analyzing image..."):
        pil_img, processed = preprocess(image_bytes)
        preds = model.predict(processed)
        pred_class = np.argmax(preds, axis=1)[0] if preds.shape[-1] > 1 else (preds > 0.5).astype(int)[0][0]

        if pred_class == 0:
            st.success("✅ No signs of Kidney Disease detected.")
        else:
            st.error("🚨 Possible Kidney Disease identified. Please consult a doctor.")

    with st.expander("ℹ️ About this App"):
        st.markdown("""
         This app uses a deep learning model based on **VGG16** architecture with **transfer learning** to detect signs of **kidney disease** from medical images.
        - It classifies uploaded kidney scans as either:
        - ✅ **No Kidney Problems**
        - 🚨 **Likely Diseased**
        - Model Experiments and Tracking done on MLFlow and DagsHub
        - Designed for prototyping and intended to assist medical professionals. Always consult a medical professional for a diagnosis.
        """)
