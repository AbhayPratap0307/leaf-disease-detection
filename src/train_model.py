from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocess import train_generator, val_generator

# Load pretrained ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# ----------- FINE-TUNING ------------
# Unfreeze last 50 layers of ResNet50
for layer in base_model.layers[:-60]:
    layer.trainable = False

for layer in base_model.layers[-60:]:
    layer.trainable = True
# ------------------------------------

# ----------- CLASSIFIER HEAD ----------
x = GlobalAveragePooling2D()(base_model.output)

x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

predictions = Dense(train_generator.num_classes, activation='softmax')(x)
# ---------------------------------------

model = Model(inputs=base_model.input, outputs=predictions)

# Low LR for fine-tuning
optimizer = Adam(learning_rate=3e-6)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Save best model + early stopping
checkpoint = ModelCheckpoint(
    "best_model.h5",
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

# ----------- TRAIN -------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[checkpoint, earlystop]
)

# Final save
model.save("plant_model.h5")
print("Training complete. Model saved!")
