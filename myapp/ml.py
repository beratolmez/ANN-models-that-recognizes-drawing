import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import tensorflow as tf
from .ml50 import classes50, preprocess_image50, predict_file50
from .kendimodelimiz import classes10, model10, predict_file10, preprocess_image, model2, classes2, predict_file2


# --- Özel CNN tanımı ---
class CustomCNN(torch.nn.Module):
    def __init__(self, in_channels, conv_layers, fc_layers, num_classes, dropout):
        super().__init__()
        layers, prev_c = [], in_channels
        for out_c in conv_layers:
            layers += [
                torch.nn.Conv2d(prev_c, out_c, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_c),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2)
            ]
            prev_c = out_c
        self.conv = torch.nn.Sequential(*layers)


        n_pool = len(conv_layers)
        feat_size = 256 // (2 ** n_pool)
        prev_f = prev_c * feat_size * feat_size
        fc_mods = []
        for h in fc_layers:
            fc_mods += [
                torch.nn.Linear(prev_f, h),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(dropout)
            ]
            prev_f = h
        fc_mods.append(torch.nn.Linear(prev_f, num_classes))
        self.fc = torch.nn.Sequential(*fc_mods)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- Temel ayarlar ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Ön işleme fonksiyonları ---
torch_preprocess = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),
])
def preprocess_torch(path):
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    img = Image.fromarray((arr*255).astype(np.uint8), mode='L')
    return torch_preprocess(img).unsqueeze(0).to(DEVICE)

def preprocess_tf(path):
    img = Image.open(path).convert('L').resize((28,28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    return arr.reshape(1,28,28,1)

# --- CustomCNN mimari konfigürasyonları ---
ARCH_CONFIG = {
    'quickdraw_customcnn_85_strong': {
        'conv_layers': [32, 64, 128, 256],
        'fc_layers':   [1024, 512, 256],
        'dropout':     0.5
    },
    'quickdraw_customcnn_50_weak': {
        'conv_layers': [16, 32, 64, 128],
        'fc_layers':   [512, 256, 128],
        'dropout':     0.3
    },
    'quickdraw_customcnn_85_weak': {
        'conv_layers': [16, 32, 64, 128],
        'fc_layers':   [512, 256, 128],
        'dropout':     0.3
    }
}

# --- Modelleri ve sınıf listelerini yükleme ---
models_meta = {}
for fn in os.listdir(MODELS_DIR):
    name, ext = os.path.splitext(fn)
    cls_path = os.path.join(MODELS_DIR, f"{name}_classes.json")
    if not os.path.exists(cls_path):
        continue
    classes = json.load(open(cls_path, encoding='utf-8'))
    full_path = os.path.join(MODELS_DIR, fn)

    if ext == '.pth':

        if name in ARCH_CONFIG:
            cfg = ARCH_CONFIG[name]
            net = CustomCNN(
                in_channels=1,
                conv_layers=cfg['conv_layers'],
                fc_layers=cfg['fc_layers'],
                num_classes=len(classes),
                dropout=cfg['dropout']
            )
        elif 'mobilenetv2' in name:
            from torchvision import models
            net = models.mobilenet_v2(weights=None)
            # ilk katmanı 1-kanal yap
            orig = net.features[0][0]
            net.features[0][0] = torch.nn.Conv2d(
                1, orig.out_channels, orig.kernel_size, orig.stride, orig.padding, bias=False
            )
            net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, len(classes))
        else:
            from torchvision.models import resnet18
            net = resnet18(weights=None)
            net.conv1 = torch.nn.Conv2d(
                1, net.conv1.out_channels,
                kernel_size=net.conv1.kernel_size,
                stride=net.conv1.stride,
                padding=net.conv1.padding,
                bias=False
            )
            net.fc = torch.nn.Linear(net.fc.in_features, len(classes))

        net.load_state_dict(torch.load(full_path, map_location=DEVICE))
        net.to(DEVICE).eval()
        models_meta[name] = {
            'type': 'torch', 'model': net,
            'classes': classes, 'preprocess': preprocess_torch
        }

    elif ext == '.h5':

        tf_model = tf.keras.models.load_model(full_path, compile=False)
        models_meta[name] = {
            'type': 'tf', 'model': tf_model,
            'classes': classes, 'preprocess': preprocess_tf
        }

# --- 50-class SimpleCNN modeli entegrasyonu ---
models_meta['quickdraw50'] = {
    'type': 'torch50',
    'model': None,
    'classes': classes50,
    'preprocess': preprocess_image50,
    'predict': lambda path: predict_file50(path)[0]
}
models_meta['model10'] = {
    'type':       'npz',
    'model':      model10,
    'classes':    classes10,
    'preprocess': preprocess_image,
    'predict':    lambda path: predict_file10(path)
}
models_meta['model2'] = {
    'type':      'npz',
    'model':     model2,
    'classes':   classes2,
    'preprocess': preprocess_image,
    'predict':   lambda path: predict_file2(path)
}


available_models = sorted(models_meta.keys())

def predict_file(model_name, image_path):
    meta = models_meta.get(model_name)
    if not meta:
        raise ValueError(f"Model bulunamadı: {model_name}")

    # Özel predict varsa onu kullan
    if 'predict' in meta and callable(meta['predict']):
        return meta['predict'](image_path)

    # Aksi halde Torch / TF akışı
    x = meta['preprocess'](image_path)
    if meta.get('type') == 'torch':
        with torch.no_grad():
            out = meta['model'](x)
            idx = int(out.argmax(dim=1).cpu().item())
    else:
        preds = meta['model'].predict(x, verbose=0)[0]
        idx   = int(np.argmax(preds))

    return meta['classes'][idx]


