import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="COVID-19 Detection App",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Load Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_covid_model.h5")

model = load_model()

# Class Labels
classes = ['Covid', 'Normal', 'Viral Pneumonia']

# -----------------------------
# App Title
# -----------------------------
st.title("🩺 COVID-19 Chest X-ray Detection")
st.markdown(
    "Upload a chest X-ray image and the deep learning model will predict the condition."
)

# -----------------------------
# Sidebar Information
# -----------------------------
st.sidebar.header("About Project")
st.sidebar.info(
    "This Streamlit application uses a Deep Learning model trained on chest X-ray images "
    "to classify patients into:\n\n"
    "• COVID-19\n"
    "• Normal\n"
    "• Viral Pneumonia"
)

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=['jpg', 'jpeg', 'png']
)

# -----------------------------
# Prediction Section
# -----------------------------
if uploaded_file is not None:

    # Open Image
    image = Image.open(uploaded_file).convert('RGB')

    # Display Uploaded Image
    st.image(image, caption='Uploaded X-ray Image', use_container_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction Button
    if st.button("🔍 Predict"):

        with st.spinner("Analyzing X-ray image..."):
            prediction = model.predict(img_array)
            predicted_class = classes[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

        # Display Prediction
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence Score: {confidence:.2f}%")

        # Probability Scores
        st.subheader("Prediction Probabilities")

        for i, class_name in enumerate(classes):
            st.write(f"{class_name}: {prediction[0][i] * 100:.2f}%")
            st.progress(float(prediction[0][i]))

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed using Streamlit and TensorFlow")
