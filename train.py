# Import packages
import os

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

# Command-line parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="dataset",
                help="path to input dataset")
ap.add_argument("-s", "--stats", type=str, default="model",
                help="path to output stats files")
ap.add_argument("-m", "--model", type=str, default="model\\covid-vgg.h5",
                help="output model file name")
args = vars(ap.parse_args())

# Training constant parameters
INIT_LR = 1e-3
EPOCHS = 25
BS = 8

# Create output dirs
print("[INFO] Creating output dirs...")
os.makedirs(os.path.abspath(args['stats']), exist_ok=True)
os.makedirs(os.path.join(os.path.abspath(args['stats']), 'augment'), exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(args['model'])), exist_ok=True)

# Enumerate files
print("[INFO] Loading images...")
data = []
labels = []

# loop over the image paths
for root, _, files in os.walk(args['dataset']):
    for file in files:
        path = os.path.join(root, file)
        img = load_img(path, target_size=(224, 224))

        data.append(img_to_array(img))
        labels.append(os.path.basename(root))

# Convert to NumPy array and rescale data
data = np.array(data) / 255.0
labels = np.array(labels)

# One-hot encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Split data
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, 
                                                stratify=labels, random_state=42)
trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

# Load the VGG16 network
baseModel = VGG16(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Freeze base model
for layer in baseModel.layers:
    layer.trainable = False

# Build model
model = Model(inputs=baseModel.input, outputs=headModel)

# Print model summary
print(model.summary())

# Compile model
print("[INFO] Compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model
print("[INFO] Training...")
H = model.fit(
    trainAug.flow(trainX, trainY, batch_size=BS, 
                  save_to_dir=os.path.join(args['stats'], 'augment')),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# Run predictions
print("[INFO] Evaluating...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

# Show classification report (accuracy, precision, recall, f1-score)
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Compute the confusion matrix
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
df_cm = pd.DataFrame(cm, lb.classes_, lb.classes_)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'

plt.figure(figsize=(10, 7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
plt.tight_layout()
plt.savefig(os.path.join(args['stats'], "confusion-matrix.png"))

# Plot training loss and accuracy
n = EPOCHS
plt.style.use("ggplot")

plt.figure()
plt.plot(np.arange(0, n), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, n), H.history["val_loss"], label="val_loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(os.path.join(args['stats'], "epoch-loss.png"))

plt.figure()
plt.plot(np.arange(0, n), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, n), H.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(os.path.join(args['stats'], "epoch-accuracy.png"))

# Serialize the model to disk
print("[INFO] Saving COVID-19 detector model...")
model.save(args['model'])
