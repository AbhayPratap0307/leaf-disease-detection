import sys
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model("plant_disease_model.h5")
print("Model loaded successfully!")

# Check if image path is provided
if len(sys.argv) < 2:
    print("Usage: python scripts/predict_leaf.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]  # take from command-line argument

# Build class label mapping from train folder
data_dir = "data/PlantVillage/train"
class_names = sorted(os.listdir(data_dir))  # actual class folders
print("Classes:", class_names)

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]
confidence = np.max(prediction)

print(f"Predicted Class: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2f}")
