
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'ml_models/quickdraw_cnn_25.h5')
CLASSES_PATH = os.path.join(BASE_DIR, 'ml_models/classes.json')


model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
    classes = json.load(f)
import cv2
import numpy as np
import cv2
import numpy as np


import cv2
import numpy as np

def preprocess_image(path, img_size=28, draw_size=20, pad=4):

    # 1) Oku
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img = clahe.apply(img)


    blur = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)


    img = cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)


    img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)


    _, th_global = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    th_adapt = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=5
    )
    th = cv2.bitwise_or(th_global, th_adapt)


    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_tophat)
    _, tophat_bin = cv2.threshold(
        tophat, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    th = cv2.bitwise_or(th, tophat_bin)


    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    th = cv2.dilate(th, kernel_dil, iterations=2)


    coords = cv2.findNonZero(th)
    if coords is not None:
        x,y,w,h = cv2.boundingRect(coords)
        roi = th[y:y+h, x:x+w]
    else:
        roi = th


    h, w = roi.shape
    scale = draw_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((draw_size, draw_size), dtype=resized.dtype)
    y_off = (draw_size - new_h) // 2
    x_off = (draw_size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized


    pad_top    = pad_left = pad // 2
    pad_bottom = img_size - draw_size - pad_top
    pad_right  = img_size - draw_size - pad_left
    final = cv2.copyMakeBorder(
        canvas,
        pad_top, pad_bottom,
        pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0
    )


    arr = final.astype('float32') / 255.0
    return arr.reshape(1, img_size, img_size, 1)




def predict_file(model, classes, path):


    x = preprocess_image(path)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    return classes[idx]
