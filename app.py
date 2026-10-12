
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "florix_ai_best.keras"
LABELS_PATH = BASE_DIR / "models" / "florix_class_names.json"
IMAGE_SIZE = (160, 160)

st.set_page_config(
    page_title="FLORIX AI",
    page_icon="🌿",
    layout="wide",
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

st.title("🌿 FLORIX AI")
st.caption("AI-powered plant disease detection")

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error("Model files not found. Check the models folder.")
    st.stop()

try:
    model = load_model()
    class_names = load_labels()
except Exception as error:
    st.error("FLORIX AI could not load its model or class labels.")
    st.exception(error)
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.image(
                image,
                caption="Uploaded leaf",
                use_container_width=True,
            )

        with right:
            resized_image = image.resize(IMAGE_SIZE)
            image_array = np.asarray(
                resized_image, dtype=np.float32
            )
            image_array = np.expand_dims(image_array, axis=0)

            with st.spinner("Analyzing leaf..."):
                predictions = model.predict(
                    image_array, verbose=0
                )[0]

            if len(predictions) != len(class_names):
                st.error("Model output does not match the class labels.")
                st.stop()

            best_index = int(np.argmax(predictions))
            confidence = float(predictions[best_index]) * 100

            st.subheader("Prediction")
            st.write("**Class:**", class_names[best_index])
            st.metric("Model confidence", f"{confidence:.2f}%")

            st.subheader("Top 3 predictions")
            top_indices = np.argsort(predictions)[-3:][::-1]

            for index in top_indices:
                st.write(
                    f"{class_names[index]} — "
                    f"{predictions[index] * 100:.2f}%"
                )

            if confidence < 60:
                st.warning(
                    "Prediction is uncertain. Try a clear, "
                    "well-lit image with the leaf centered."
                )

    except Exception as error:
        st.error("Unable to analyze this image.")
        st.exception(error)
