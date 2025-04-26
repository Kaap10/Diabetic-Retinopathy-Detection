import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

MODEL_PATH = os.path.join('model', 'model.h5')
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']  # Adjust as per your dataset

app = Flask(__name__)
CORS(app)

# Load model once at startup
model = load_model(MODEL_PATH)

def preprocess_image(img):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img = image.load_img(file, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = preprocess_image(img)
        preds = model.predict(img_array)
        pred_class = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        result = {
            'class': CLASS_NAMES[pred_class],
            'confidence': confidence
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)