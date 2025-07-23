import numpy as np
from matplotlib import pyplot as plt
# main.py en başına ekleyin

import itertools


# ——— 1) Veri Yükleme & Önişleme —————————————

def one_hot(labels, C):
    Y = np.zeros((labels.shape[0], C))
    Y[np.arange(labels.shape[0]), labels] = 1
    return Y

def load_data(classes, data_dir="", img_shape=(28,28), test_ratio=0.2, seed=42):

    np.random.seed(seed)
    X_list, y_list = [], []
    for idx, cls in enumerate(classes):
        data = np.load(f"{data_dir}{cls}.npy")
        if data.ndim == 2:
            data = data.reshape(-1, *img_shape)
        X_list.append(data)
        y_list.append(np.full(data.shape[0], idx))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    X = X.astype(np.float16) / 255.0
    N, H, W = X.shape
    X_flat = X.reshape(N, H*W)

    split = int((1-test_ratio)*N)
    X_train, X_test = X_flat[:split], X_flat[split:]
    y_train, y_test = y[:split], y[split:]

    Y_train = one_hot(y_train, len(classes))
    Y_test  = one_hot(y_test,  len(classes))

    return X_train, Y_train, X_test, Y_test, y_test

# ——— 2) Katman & Kayıp Tanımları ——————————

class Dense:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0/in_dim)
        self.b = np.zeros((1, out_dim))

    def forward(self, A_prev):
        self.A_prev = A_prev
        return A_prev.dot(self.W) + self.b

    def backward(self, dZ, lr):
        dW = self.A_prev.T.dot(dZ)
        db = dZ.sum(axis=0, keepdims=True)
        dA_prev = dZ.dot(self.W.T)
        self.W -= lr * dW
        self.b -= lr * db
        return dA_prev

class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)
    def backward(self, dA):
        return dA * (self.Z > 0)

class SoftmaxCrossEntropy:
    def forward(self, logits, Y_true):
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        self.probs = exps / exps.sum(axis=1, keepdims=True)
        N = logits.shape[0]
        return -np.sum(Y_true * np.log(self.probs + 1e-9)) / N

    def backward(self, Y_true):
        N = Y_true.shape[0]
        return (self.probs - Y_true) / N

# ——— 3) Model Sınıfı ——————————

class SimpleNN:
    def __init__(self, layer_dims, lr=0.01):
        self.lr = lr
        self.layers = []
        for i in range(len(layer_dims)-1):
            self.layers.append(Dense(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims)-2:
                self.layers.append(ReLU())
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, dL):
        grad = dL
        for layer in reversed(self.layers):
            if isinstance(layer, Dense):
                grad = layer.backward(grad, self.lr)
            else:
                grad = layer.backward(grad)

    def train(self,
              X, Y,
              validation_data=None,
              epochs=20,
              batch_size=64):
        N = X.shape[0]

        train_losses = []
        val_losses = []

        for ep in range(1, epochs + 1):
            # --- Training epoch ---
            epoch_loss = 0.0
            for i in range(0, N, batch_size):
                Xb, Yb = X[i:i + batch_size], Y[i:i + batch_size]
                logits = self.forward(Xb)
                loss = self.loss_fn.forward(logits, Yb)
                grad = self.loss_fn.backward(Yb)
                self.backward(grad)
                epoch_loss += loss * Xb.shape[0]

            epoch_loss /= N
            train_losses.append(epoch_loss)


            if validation_data is not None:
                X_val, Y_val = validation_data

                logits_val = self.forward(X_val)
                val_loss = self.loss_fn.forward(logits_val, Y_val)
                val_losses.append(val_loss)
                print(f"Epoch {ep}/{epochs} — train loss: {epoch_loss:.4f} — val loss: {val_loss:.4f}")
            else:
                print(f"Epoch {ep}/{epochs} — train loss: {epoch_loss:.4f}")

        return train_losses, val_losses

    def predict(self, X, batch_size=256):
        preds = []
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size]
            logits = self.forward(xb)
            preds.append(np.argmax(logits, axis=1))
        return np.concatenate(preds)


# ——— 4) Model Kaydet / Yükle ——————————

def save_model(model, filename):

    params = {}
    idx = 0
    for layer in model.layers:
        if isinstance(layer, Dense):
            params[f"W{idx}"] = layer.W
            params[f"b{idx}"] = layer.b
            idx += 1
    np.savez(filename, **params)
    print(f"Model parametreleri '{filename}' olarak kaydedildi.")

def load_model(model, filename):

    data = np.load(filename)
    idx = 0
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.W = data[f"W{idx}"]
            layer.b = data[f"b{idx}"]
            idx += 1
    print(f"Model parametreleri '{filename}' yüklendi.")



def predict_image(model, classes, npy_path, img_shape=(28,28)):

    img = np.load(npy_path)
    if img.ndim == 1:
        img = img.reshape(*img_shape)
    img = img.astype(np.float16) / 255.0
    x = img.reshape(1, -1)
    pred_idx = model.predict(x)[0]
    return classes[pred_idx]


# ——— ekle: 7) Değerlendirme Fonksiyonu ——————————

def evaluate(model, X, y_true, batch_size=64):

    N = X.shape[0]
    preds = []

    for i in range(0, N, batch_size):
        xb = X[i:i+batch_size]
        p = model.predict(xb)
        preds.append(p)
    preds = np.concatenate(preds)
    return np.mean(preds == y_true)

from PIL import Image
import os

from PIL import Image
import numpy as np
import os
def preprocess_image(path, img_shape=(28,28)):

    ext = os.path.splitext(path)[1].lower()

    if ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'):
        img = Image.open(path).convert('L')
        img = img.resize(img_shape, Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float16) / 255.0
    else:
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr.reshape(img_shape)
        arr = arr.astype(np.float16) / 255.0


    arr = 1.0 - arr

    return arr.reshape(1, -1)


def predict_file(model, classes, file_path, img_shape=(28,28)):
    x = preprocess_image(file_path, img_shape)
    idx = model.predict(x)[0]
    return classes[idx]




# 1) 10 sınıf isimleri
classes10 = [
    'ice cream', 'bird', 'car', 'flower', 'clock',
    'bicycle', 'star', 'fish', 'face', 'rabbit'
]

# 2) Model objesini oluştur ve .npz’den yükle
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'ml_models', 'model10.npz')

# katman boyutları 28*28 → 128 → 64 → 10
model10 = SimpleNN(layer_dims=[28*28, 128, 64, len(classes10)], lr=0.01)
load_model(model10, MODEL_PATH)


# --- 5 sınıflı model için sınıf listesi ---
classes2 = ['bird', 'car', 'clock', 'flower', 'ice cream']

# --- model2 objesini oluştur ve .npz’den yükle ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL2_PATH = os.path.join(BASE_DIR, 'ml_models', 'model2.npz')

# 28*28 → 128 → 64 → 5
model2 = SimpleNN(layer_dims=[28*28, 128, 64, len(classes2)], lr=0.01)
load_model(model2, MODEL2_PATH)


# 3) Tek satırlık predict fonksiyonu
def predict_file10(path):

    return predict_file(model10, classes10, path)


def predict_file2(path):

    return predict_file(model2, classes2, path)