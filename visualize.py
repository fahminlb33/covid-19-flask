# Import packages
import os

import argparse
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Model, load_model

# Command-line parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="dataset\\normal\\IM-0115-0001.jpeg",
                help="path to input file")
ap.add_argument("-s", "--stats", type=str, default="model\\layers",
                help="path to output stats files")
ap.add_argument("-m", "--model", type=str, default="model\\covid-vgg.h5",
                help="saved model file name")
args = vars(ap.parse_args())

# Training constant parameters
LAYERS = 20
IMAGES_PER_ROW = 16

# Create output dirs
os.makedirs(os.path.abspath(args['stats']), exist_ok=True)

# Load model
model: Model = load_model(args['model'])

# Load single file
testX = load_img(args['input'], target_size=(224, 224))
testX = img_to_array(testX)
testX = np.expand_dims(testX, axis=0)

# Enumerate layers and it's activation
layer_outputs = [layer.output for layer in model.layers[1:LAYERS]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(testX)

# Enumerate layer names
layer_names = []
for layer in model.layers[1:LAYERS]:
    layer_names.append(layer.name)

# Run layer activation
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // IMAGES_PER_ROW
    display_grid = np.zeros((size * n_cols, IMAGES_PER_ROW * size))

    for col in range(n_cols):
        for row in range(IMAGES_PER_ROW):
            channel_image = layer_activation[0, :, :, col * IMAGES_PER_ROW + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image
                         
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig(os.path.join(args['stats'], "%s.png" % (layer_name)))
