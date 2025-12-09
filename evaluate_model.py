from tensorflow.keras.models import load_model
from preprocess import val_generator
import numpy as np
from sklearn.metrics import classification_report

model = load_model('plant_model.h5')

val_generator.reset()
preds = model.predict(val_generator)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes

print(classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))
