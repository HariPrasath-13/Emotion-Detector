import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np

st.title("🧠 Emotion Detector")
st.write("Upload a face image and I’ll analyze the dominant emotion using DeepFace.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Analyze emotions
    try:
        result = DeepFace.analyze(np.array(image), actions=['emotion'], enforce_detection=False)
        st.success(f"Detected Emotion: **{result[0]['dominant_emotion']}**")
        st.write("Full analysis:", result[0]['emotion'])
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
