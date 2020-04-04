# Import packages
import os
import argparse

from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Command-line parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="model\\covid-vgg.h5",
                help="saved model file name")
args = vars(ap.parse_args())

# Initiate Flask server
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static'

# Used to hold model instance
model: Model

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return redirect("/")

    # Save uploaded file to folder
    file = request.files['file']
    name, path = save_file(file)

    # Load uploaded image and run preprocessing
    image = load_img(path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Run prediction
    result = model.predict(image)
    result = np.argmax(result, axis=1)
    result = "Normal" if result[0] == 1 else "COVID-19"

    return render_template("predict.html", result=result, img_src=url_for('static', filename=name))

def save_file(file):
    extension = file.filename.rsplit('.', 1)[1].lower()
    name = 'image.' + extension
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)

    try:
        os.remove(path)
    except OSError:
        pass
        
    file.save(path)
    return (name, path)

# Main entry point
if __name__ == "__main__":
    # Load saved model
    model = load_model(args['model'])

    # Run web server
    app.run(debug=True)
