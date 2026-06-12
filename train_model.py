"""
train_model.py

Train violence detection model using images + videos
"""

import os
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# =============================
# CONFIG
# =============================

DATASET_BASE_PATH = r"C:\Users\Niranjan Manjunath\OneDrive\Desktop\violence-detection-main\dataset\A-Dataset-for-Automatic-Violence-Detection-in-Videos-master\violence-detection-dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mov"}

# =============================
# LABEL
# =============================

def get_label(name):
    name = name.lower()
    if name == "violent":
        return 1
    if name in ["non_violent", "non-violent", "normal"]:
        return 0
    return None

# =============================
# IMAGE LOADER
# =============================

def load_images(folder, label):
    images = []
    labels = []

    for file in folder.rglob("*"):
        if file.suffix.lower() not in IMAGE_EXTS:
            continue

        img = cv2.imread(str(file))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0

        images.append(img)
        labels.append(label)

    return images, labels

# =============================
# VIDEO LOADER
# =============================

def load_video(video_path, label, max_frames=30, frame_interval=10):
    frames = []
    labels = []

    cap = cv2.VideoCapture(str(video_path))
    count = 0
    frame_id = 0

    try:
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, IMG_SIZE)
                frame = frame / 255.0

                frames.append(frame)
                labels.append(label)
                count += 1

            frame_id += 1

    finally:
        cap.release()

    return frames, labels

# =============================
# DATASET BUILDER
# =============================

def build_dataset(path):
    path = Path(path)

    if not path.exists():
        raise Exception("Dataset path not found!")

    class_folders = [f for f in path.iterdir() if f.is_dir()]

    images = []
    labels = []

    total_videos = 0
    total_frames = 0

    print("\n[INFO] Loading dataset...\n")

    for folder in class_folders:
        label = get_label(folder.name)
        if label is None:
            continue

        print(f"Processing folder: {folder.name}")

        # Images
        imgs, lbs = load_images(folder, label)
        images.extend(imgs)
        labels.extend(lbs)

        # Videos
        for file in folder.rglob("*"):
            if file.suffix.lower() in VIDEO_EXTS:
                total_videos += 1
                vid_frames, vid_labels = load_video(file, label)
                total_frames += len(vid_frames)

                images.extend(vid_frames)
                labels.extend(vid_labels)

    print("\n===== DATASET SUMMARY =====")
    print("Total samples:", len(images))
    print("Total videos:", total_videos)
    print("Total frames extracted:", total_frames)

    return np.array(images), np.array(labels)

# =============================
# MODEL
# =============================

def build_model():
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=base.input, outputs=output)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

# =============================
# MAIN
# =============================

def main():
    X, y = build_dataset(DATASET_BASE_PATH)

    if len(X) == 0:
        raise Exception("No data loaded!")

    print("\nSplitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nBuilding model...")
    model = build_model()

    print("\nTraining model...\n")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    model.save("trained_model.h5")
    print("\nModel saved as trained_model.h5")

    print("\nFinal Accuracy:")
    print("Train:", history.history["accuracy"][-1])
    print("Validation:", history.history["val_accuracy"][-1])

# =============================

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()