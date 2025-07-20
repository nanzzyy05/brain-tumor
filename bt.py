
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
# Load your trained model
model = tf.keras.models.load_model('best_model.h5')  

# Define your class names
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']  

# App title
st.title(":green[🧠 Brain Tumor Classification]")
st.markdown("Upload an MRI image and get the tumor type.")

# Image upload
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Preprocess
    image_resized = image.resize((224, 224))
    image_array = img_to_array(image_resized)
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence_scores = prediction[0]

    # Display result
    st.markdown(f"### 🎯 Predicted Tumor Type: **{predicted_class}**")
    st.markdown("#### 📊 Confidence Scores:")
    for i, score in enumerate(confidence_scores):
        st.write(f"{class_names[i]}: {score:.2%}")
 