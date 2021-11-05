import tensorflow as tf
from PIL import Image
import numpy as np
import os


def load_image(source):
    img = Image.open(source)
    img = np.array(img)

    return img


files = os.listdir("./tests")
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

model = tf.keras.models.load_model("cifar10_2/", compile=False)

for file in files:
    image = load_image("./tests/" + file)
    image = np.expand_dims(image, 0)

    prediction = model(image)
    print(file + " / Prediction = " + classes[np.argmax(prediction)])
