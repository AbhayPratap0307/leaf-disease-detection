from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image size and batch size
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
SEED = 42

# Path to *train* dataset (NO val folder needed)
TRAIN_DIR = 'data/PlantVillage/train'

# ------------------ DATA AUGMENTATION (STABLE) ------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=8,
    width_shift_range=0.03,
    height_shift_range=0.03,
    shear_range=0.03,
    zoom_range=0.05,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest',
    validation_split=0.2
)

# ------------------ TRAIN GENERATOR ------------------
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED,
    interpolation="bilinear"
)

# ------------------ VALIDATION GENERATOR ------------------
val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,         # MUST be False for stable validation
    seed=SEED,
    interpolation="bilinear"
)

# Print classes detected
print("Classes found:", train_generator.class_indices)
print("Training samples:", train_generator.samples)
print("Validation samples:", val_generator.samples)
