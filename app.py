import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from numpy import loadtxt
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import os
from flask import Flask, jsonify, flash, send_file, request, redirect, url_for
from werkzeug.utils import secure_filename

# Create a directory in a known location to save files to.
app = Flask(__name__)

model = load_model('./model.h5')

app = Flask(__name__, static_folder='build', static_url_path='/')

@app.route('/pic')
def serve_image():
    PEOPLE_FOLDER = os.path.join('instance', 'uploads')
    app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.jpg')
    return send_file(full_filename, mimetype='image/gif')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/upload', methods=['POST'])
def fileUpload():
    uploads_dir = os.path.join(app.instance_path, 'uploads')
    if not os.path.isdir(uploads_dir):
        os.makedirs(uploads_dir)
    profile = request.files['file']
    destination = "/".join([ uploads_dir, 'uploaded.jpg'])
    profile.save(destination)
    test_image = image.load_img('./instance/uploads/uploaded.jpg', target_size=(128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image/255.0
    # result = cnn.predict(test_image)
    result = model.predict_classes(test_image)
    if result[0] == 1:
        prediction = 'camera'
    elif result[0] == 2:
        prediction = 'headphone'
    elif result[0] == 3:
        prediction = 'mobile'
    else:
        prediction = 'baggage'

    print(prediction)

    return jsonify( result = prediction, picture = "/pic" )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    # app.run(host='0.0.0.0', port=port)
    # app.run()
    app.run(host='localhost', port=port)

