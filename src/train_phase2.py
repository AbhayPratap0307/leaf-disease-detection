from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from scripts.preprocess import train_generator, val_generator


import tensorflow as tf

# ------------------ LOAD MODEL TRAINED IN PHASE 1 ------------------
model = load_model("best_model.h5")
print("Loaded model from best_model.h5")

# ------------------ EXTRACT BASE MODEL AUTOMATICALLY ------------------
gap_index = None
for i, layer in enumerate(model.layers):
    if layer.name == "global_average_pooling2d":
        gap_index = i
        break

if gap_index is None:
    raise ValueError("GlobalAveragePooling2D layer not found!")

# The base model ends right before GAP layer
base_model = Model(model.input, model.layers[gap_index - 1].output)
print("Base model extracted. Total layers in base model:", len(base_model.layers))

# ------------------ FINE-TUNE LAST 60 LAYERS ------------------
for layer in base_model.layers[:-60]:
    layer.trainable = False

for layer in base_model.layers[-60:]:
    layer.trainable = True

# Apply trainability to original model layers
for i, layer in enumerate(model.layers[:gap_index - 1]):
    layer.trainable = base_model.layers[i].trainable

print("Trainable layers updated.")

# ------------------ COMPILE ------------------
model.compile(
    optimizer=Adam(learning_rate=3e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------ CALLBACKS ------------------
checkpoint = ModelCheckpoint(
    "best_model_finetuned.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-7
)

# ------------------ TRAIN ------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

model.save("plant_model_finetuned.h5")
print("Fine-tuning complete!")
