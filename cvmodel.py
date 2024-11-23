# pip install numpy pandas opencv-python matplotlib scikit-learn tensorflow

###
### DATA - Load, visualize, split, normalize, one-hot encode
###

import kagglehub
path = kagglehub.dataset_download("nikhilgurav21/nail-disease-detection-dataset")
print("Path to dataset files:", path)

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(data_dir, subset='train'):
    X, y = [], []
    subset_dir = os.path.join(data_dir, subset)
    classes = sorted(os.listdir(subset_dir))
    for label, class_name in enumerate(classes):
        class_path = os.path.join(subset_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (224, 224))
                        X.append(img)
                        y.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    return np.array(X), np.array(y), classes

def visualize_samples(X, y, classes, num_samples=5):
    indices = np.random.choice(len(X), num_samples, replace=False)
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(cv2.cvtColor(X[idx], cv2.COLOR_BGR2RGB))
        plt.title(classes[y[idx]])
        plt.axis('off')
    plt.show()

def split_data(X, y, test_size=0.2, val_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=42)
    val_split = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_split, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_data(X, y, num_classes):
    X_normalized = X / 255.0
    y_encoded = to_categorical(y, num_classes=num_classes)
    return X_normalized, y_encoded

if __name__ == "__main__":
    dataset_dir = "/root/.cache/kagglehub/datasets/nikhilgurav21/nail-disease-detection-dataset/versions/1/data"

    X_train, y_train, classes = load_dataset(dataset_dir, subset='train')
    X_val, y_val, _ = load_dataset(dataset_dir, subset='validation')

    print(f"Loaded {len(X_train)} training images from {len(classes)} classes.")
    print(f"Loaded {len(X_val)} validation images from {len(classes)} classes.")

    visualize_samples(X_train, y_train, classes)

    num_classes = len(classes)
    X_train, y_train = preprocess_data(X_train, y_train, num_classes)
    X_val, y_val = preprocess_data(X_val, y_val, num_classes)

    print("Data preprocessing complete.")

### FOR DEBUG

# import os
#
# print("Dataset directory contents:")
# print(os.listdir(dataset_dir))
# for folder in os.listdir(dataset_dir):
#     folder_path = os.path.join(dataset_dir, folder)
#     if os.path.isdir(folder_path):
#         print(f"Folder: {folder}")
#         print("Contents:", os.listdir(folder_path)[:5])

###


###
### MODEL -
###