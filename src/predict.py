import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('plant_model.h5')

def predict_leaf(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    class_labels = list(model.class_indices.keys())  
    print("Predicted Disease:", class_labels[class_idx])


predict_leaf('../data/PlantVillage/Apple___Apple_scab/0001.jpg')
