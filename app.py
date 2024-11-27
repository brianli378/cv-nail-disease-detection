from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import io

app = Flask(__name__)
MODEL_PATH = os.path.join("model", "nail_disease_detector.keras")
model = tf.keras.models.load_model(MODEL_PATH)

classes = ["Acral Lentiginous Melanoma", "Blue Finger", "Clubbing", "Healthy Nail", "Onychogryphosis", "Pitting"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/team")
def team():
    return render_template("team.html")

@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("file")
    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        img_bytes = uploaded_file.read()
        img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = classes[np.argmax(predictions)]

        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
