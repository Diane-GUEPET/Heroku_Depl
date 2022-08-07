# web-app for API image manipulation

# web-app for API image manipulation

from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image

import tensorflow
import keras
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import load_img

import tqdm
import numpy as np
import cv2
import os
import tqdm
from os.path import isfile


app = Flask(__name__)
server=app.server

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_models():
    model = keras.models.load_model('./models/model_Unet.h5')
    return model


model = load_models()


# Constantes
img_size = (512, 512)
num_classes = 8
batch_size = 3
imgaug_multiplier = 1
epochs = 10
patience = 5
alpha = 0.6


# Fonction du Preprocessing


def prepro(img_path):

    x = np.zeros((batch_size * imgaug_multiplier,) +
                 img_size + (3,), dtype="float32")
    for j, path in enumerate(img_path):
        img = image.load_img(path, target_size=(512, 512))
        x[j] = img

    return x

# Fonction de prédiction du masque


def semantic_segmentation(img_path):
    color_map = {
        '0': [0, 0, 0],
        '1': [153, 153, 0],
        '2': [255, 204, 204],
        '3': [255, 0, 127],
        '4': [0, 255, 0],
        '5': [0, 204, 204],
        '6': [255, 0, 0],
        '7': [0, 0, 255]
    }

    img = image.load_img(img_path[0])
    dims = img.size  # Recup des dimensions réelles..
    img = img.resize(img_size)
    dims1 = img.size

    test_seq = prepro(img_path)
    val_preds = model.predict(test_seq)
    mask = np.argmax(val_preds[0], axis=2)
    mask = np.expand_dims(mask, axis=2)
    mask = np.squeeze(mask)
    mask1 = mask.copy()
    mask1 = np.array(mask1, dtype='uint8')
    img1=image.img_to_array(img)


    mask_color = img1.copy()   
    for l in range(dims1[0]):
        for j in range(dims1[1]):
            mask_color[l, j] = color_map[str(mask1[l, j])]
    
    #cv2.addWeighted(mask_color, alpha, mask_color, 1-alpha, 0, mask_color)
    mask_color = cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR)
    mask_color=cv2.resize(mask_color, (dims[0], dims[1]), interpolation=cv2.INTER_LINEAR)
    return (mask_color)

# Fonction permettant d'appliquer le masque


def semantic_segmentation2(img_path):
    color_map = {
        '0': [0, 0, 0],
        '1': [153, 153, 0],
        '2': [255, 204, 204],
        '3': [255, 0, 127],
        '4': [0, 255, 0],
        '5': [0, 204, 204],
        '6': [255, 0, 0],
        '7': [0, 0, 255]
    }

    img = image.load_img(img_path[0])
    dims = img.size  # Recup des dimensions réelles..
    img = img.resize(img_size)
    dims1 = img.size

    test_seq = prepro(img_path)
    val_preds = model.predict(test_seq)
    mask = np.argmax(val_preds[0], axis=2)
    mask = np.expand_dims(mask, axis=2)
    mask = np.squeeze(mask)

    mask1 = mask.copy()
    mask1 = np.array(mask1, dtype='uint8')


    img1 = image.img_to_array(img)
    img_color = img1.copy()
    for l in range(dims1[0]):
        for j in range(dims1[1]):
            img_color[l, j] = color_map[str(mask1[l, j])]
    
    cv2.addWeighted(img1, alpha, img_color, 1-alpha, 0, img_color)
    img_pred = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
    img_pred = cv2.resize(
        img_pred, (dims[0], dims[1]), interpolation=cv2.INTER_LINEAR)
    return(img_pred)

# default access page


@app.route("/")
def main():
    return render_template('index.html')


# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/images/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".png"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file

    destination = "/".join([target, filename])
    print("File saved to:", destination)
    upload.save(destination)

    # forward to processing page
    return render_template("processing.html", image_name=filename)


# Création du masque
@app.route("/predicted_mask", methods=["POST"])
def predicted_mask():
    # retrieve parameters from html form
    #angle = request.form['angle']
    filename = request.form['image']

    # open and process image
    target = os.path.join(APP_ROOT, 'static/images')

    destination = "/".join([target, filename])

    img = Image.open(destination)

    input_dir = "./static/images/"
    input_path = []
    input_path = input_path + sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
    ]
)
    #img = img.rotate(-1*int(angle))
    img = semantic_segmentation(input_path)

    # save and return image
    destination = "/".join([target, 'temp.png'])
    if os.path.isfile(destination):
        os.remove(destination)
    # img.save(destination)
    cv2.imwrite(destination, img)

    return send_image('temp.png')


# flip filename 'vertical' or 'horizontal'
@app.route("/applied_mask", methods=["POST"])
def applied_mask():

    filename = request.form['image']

    # open and process image
    target = os.path.join(APP_ROOT, 'static/images')

    destination = "/".join([target, filename])

    img = Image.open(destination)

    input_dir = "./static/images/"
    input_path = []
    input_path = input_path + sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
    ]
)

    img = semantic_segmentation2(input_path)

    # save and return image
    destination = "/".join([target, 'temp.png'])
    if os.path.isfile(destination):
        os.remove(destination)
    # img.save(destination)
    cv2.imwrite(destination, img)

    return send_image('temp.png')



# retrieve file from 'static/images' directory
@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run(port=8080, debug=True)
