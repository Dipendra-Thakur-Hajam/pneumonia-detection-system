import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Build absolute paths so the script works from any working directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(BASE_DIR, "dataset", "train")
val_dir = os.path.join(BASE_DIR, "dataset", "val")
test_dir = os.path.join(BASE_DIR, "dataset", "test")


# -------------------------
# Data Augmentation (Training)
# -------------------------
# Use ResNet50's preprocess_input instead of rescale=1./255
# This applies the correct ImageNet normalization the backbone expects
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    channel_shift_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation/Test: only apply preprocessing, no augmentation
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# -------------------------
# Data Generators
# -------------------------
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)


# -------------------------
# Class Weights (to fix imbalance)
# -------------------------
# Compute weights inversely proportional to class frequency
# This ensures NORMAL class gets higher weight to compensate for fewer samples
labels = train_generator.classes  # array of 0s and 1s
class_labels = np.unique(labels)
weights = compute_class_weight('balanced', classes=class_labels, y=labels)
class_weights = dict(zip(class_labels.astype(int), weights))

print(f"\n--- Dataset Summary ---")
print(f"Training:   {train_generator.samples} images ({dict(zip(train_generator.class_indices.keys(), [np.sum(labels == v) for v in train_generator.class_indices.values()]))})")
print(f"Validation: {val_generator.samples} images")
print(f"Test:       {test_generator.samples} images")
print(f"Class mapping: {train_generator.class_indices}")
print(f"Class weights: {class_weights}")
print(f"-----------------------\n")