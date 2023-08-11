from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

model_path = "Tumor_classifier_model.h5"
loaded_model = tf.keras.models.load_model(model_path)

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize the pixel values
    return np.expand_dims(image_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = request.files['image']
        image_pil = Image.open(image)
        input_data = preprocess_image(image_pil)
        pred = loaded_model.predict(input_data)
        result = pred.argmax()
        print(int(result))
        return jsonify({'prediction': int(result)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
