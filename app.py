# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from tkinter import Image
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
app=Flask(__name__)
model = load_model('your_model.h5')
@app.route('/')
def home():
    return predict()


def preprocess_image(img):
    img = tf.image.resize(img, [299, 299])
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.sigmoid(predictions[0])
    result=np.argmax(score)
    return result



@app.route('/predict',methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']

    image = Image.open(image)
    # Preprocess the image as needed (e.g., resize to the model's input size and normalize)
    result = preprocess_image(image)
    print(result)

    # Perform inference using your model


    return jsonify({'result': str(result)})

if __name__ == '__main__':
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
