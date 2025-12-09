import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ---- 1️⃣ Safer model load ----
model_path = "best_model_finetuned.h5"

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error(f"Model file not found: {model_path}")
    st.stop()  # Stop execution if model is missing

# ---- 2️⃣ Streamlit UI ----
st.title("🌿 Plant Disease Detection")
st.write("Upload an image of a plant leaf to detect disease.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Leaf Image', use_column_width=True)
        
        # Prepare image for prediction
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize if your model was trained on 0-1 images

        # Make prediction
        prediction = model.predict(img_array)

        st.write("Raw model output:", prediction)
        st.write("Prediction shape:", prediction.shape)

        # Safely get predicted class index
        if prediction.ndim == 2:  # output shape (1, num_classes)
            class_idx = int(np.argmax(prediction, axis=1)[0])
        elif prediction.ndim == 1:  # output shape (num_classes,)
            class_idx = int(np.argmax(prediction))
        else:
            st.error("Unexpected prediction shape from model.")
            st.stop()

        # Replace this list with your actual class labels in correct order
        class_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___Healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___Healthy",
    "Corn___Common_rust",
    "Corn___Healthy",
    "Grape___Black_rot",
    "Grape___Healthy"
]


        if class_idx >= len(class_labels):
            st.error(f"Predicted class index {class_idx} exceeds available labels.")
        else:
            predicted_class = class_labels[class_idx]
            st.success(f"Predicted Disease: **{predicted_class}**")

    except Exception as e:
        st.error(f"Error processing image: {e}")

 # ---- Background color + image ----
page_bg_style = """
<style>
.stApp {
background: linear-gradient(to bottom right, #a8e6cf, #dcedc1), 
            url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg_style, unsafe_allow_html=True)
