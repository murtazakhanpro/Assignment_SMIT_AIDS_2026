from pathlib import Path
import shutil
import random
from collections import Counter

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

random.seed(42)
tf.random.set_seed(42)

BASE_DIR = Path.cwd()
SPLIT_DIR = BASE_DIR / "data_split"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORED_FOLDERS = {"train", "test", "val", "validation", "data_split"}

# Dataset images locate karein; download nahi hoga
images = []

for file in BASE_DIR.rglob("*"):
    if (
        file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.lower() in IGNORED_FOLDERS for part in file.parts)
    ):
        images.append((file, file.parent.name))

if not images:
    raise FileNotFoundError(
        "Dataset images nahi milin. Dataset ko assignment folder ke andar rakhein."
    )

class_counts = Counter(label for _, label in images)
classes = sorted(class_counts)

if len(classes) != 5:
    print(f"Warning: {len(classes)} classes found: {classes}")
else:
    print("Five classes:", classes)

print("Total images:", len(images))
print("Class distribution:", class_counts)

# Old split remove karein
if SPLIT_DIR.exists():
    shutil.rmtree(SPLIT_DIR)

for split in ["train", "validation", "test"]:
    for class_name in classes:
        (SPLIT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

# 70% train, 15% validation, 15% test
for class_name in classes:
    class_images = [(path, label) for path, label in images if label == class_name]
    random.shuffle(class_images)

    total = len(class_images)
    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    split_data = {
        "train": class_images[:train_end],
        "validation": class_images[train_end:validation_end],
        "test": class_images[validation_end:]
    }

    for split, files in split_data.items():
        for index, (source, _) in enumerate(files):
            destination = (
                SPLIT_DIR / split / class_name / f"{index}_{source.name}"
            )
            shutil.copy2(source, destination)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_dataset(path, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=shuffle,
        seed=42
    ).prefetch(tf.data.AUTOTUNE)

train_ds = load_dataset(SPLIT_DIR / "train", True)
validation_ds = load_dataset(SPLIT_DIR / "validation", False)
test_ds = load_dataset(SPLIT_DIR / "test", False)

class_names = train_ds.class_names
num_classes = len(class_names)

# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),

    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.Rescaling(1 / 255),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(256, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_animal_classifier.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=25,
    callbacks=callbacks
)

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy:.2%}")

model.save("animal_classifier.keras")

# Training graphs
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.show()

# Classification report and confusion matrix
y_true = []
y_pred = []

for images_batch, labels_batch in test_ds:
    predictions = model.predict(images_batch, verbose=0)
    y_true.extend(np.argmax(labels_batch.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()