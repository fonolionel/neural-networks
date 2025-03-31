import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CLASSES = 43  # Adjust based on dataset

def load_dataset(dataset_path):
    """Loads images and labels from dataset folders"""
    images = []
    labels = []

    if not os.path.exists(dataset_path):
        sys.exit(f"Error: Dataset path '{dataset_path}' does not exist.")

    for class_id in range(NUM_CLASSES):
        class_folder = os.path.join(dataset_path, str(class_id))
        if not os.path.exists(class_folder):
            print(f"⚠️ Warning: Folder {class_id} does not exist in dataset.")
            continue

        for image_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, image_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipping unreadable file: {img_path}")
                continue

            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(class_id)

    if len(images) == 0:
        sys.exit("❌ Error: No images found. Check dataset structure!")

    images = np.array(images) / 255.0  # Normalize
    labels = np.array(labels)
    return images, labels

def build_model():
    """Creates a CNN model for traffic sign classification"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train(dataset_path, model_name="model.h5"):
    """Trains and saves the model"""
    print("📥 Loading dataset...")
    X, y = load_dataset(dataset_path)
    y = to_categorical(y, NUM_CLASSES)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"✅ Dataset Loaded! Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Build and train model
    model = build_model()
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

    # Save trained model
    model.save(model_name)
    print(f"✅ Model saved as {model_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python traffic.py <dataset_path> [model_name]")

    dataset_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "model.h5"

    train(dataset_path, model_name)
