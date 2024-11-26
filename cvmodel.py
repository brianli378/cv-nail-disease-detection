# pip install numpy pandas opencv-python matplotlib scikit-learn tensorflow

###
### DATA - Load, visualize, split, normalize, one-hot encode
###

# import kagglehub
# path = kagglehub.dataset_download("nikhilgurav21/nail-disease-detection-dataset")
# print("Path to dataset files:", path)

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
    dataset_dir = "./data"

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

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def build_model(num_classes):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


dataset_dir = "./data"
train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')

X_train, y_train, classes = load_dataset(dataset_dir, subset='train')
X_val, y_val, _ = load_dataset(dataset_dir, subset='validation')


num_classes = len(classes)

X_train, y_train = preprocess_data(X_train, y_train, num_classes)
X_val, y_val = preprocess_data(X_val, y_val, num_classes)


model = build_model(num_classes)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

batch_size = 32
epochs = 10

# training
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    steps_per_epoch=len(X_train) // batch_size,
    verbose=1
)

val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=1)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

model.save("nail_disease_detector.h5")
print("Model saved as nail_disease_detector.h5")

import matplotlib.pyplot as plt

### plots
def plot_training(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss")

    plt.show()

plot_training(history)

###
### Testing with single input - delete when converting to web app
###

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

img_path = "nf.jpg"

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)


prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]

print(f"Predicted Class: {classes[predicted_class]}")

plt.imshow(img)
plt.title(f"Predicted: {classes[predicted_class]}")
plt.axis('off')
plt.show()

###
###
###
