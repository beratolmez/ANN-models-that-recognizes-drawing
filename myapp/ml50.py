import os, json, torch
import numpy as np
from PIL import Image
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, img_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        fc_dim = (img_size//8)*(img_size//8)*128
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# --- Paths & device ---
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH50  = os.path.join(BASE_DIR, 'ml_models', 'quickdraw502.pth')
CLASSES50_PATH= os.path.join(BASE_DIR, 'ml_models', 'classes50.json')
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load 50-class names ---
with open(CLASSES50_PATH, 'r', encoding='utf-8') as f:
    classes50 = json.load(f)

# --- Build & load model ---
model50 = SimpleCNN(len(classes50), img_size=64)
state = torch.load(MODEL_PATH50, map_location=DEVICE)
model50.load_state_dict(state)
model50.to(DEVICE)
model50.eval()

from PIL import ImageOps

import numpy as np
import torch
from PIL import Image, ImageOps


def preprocess_image50(path: str) -> torch.Tensor:
    """
    1) Load → grayscale → resize to 64×64
    2) Ensure black background with white drawing
    3) Threshold at 127 → binary {0.0,1.0}
    4) To tensor shape (1,1,64,64) on DEVICE
    """
    # 1) load + resize
    img = Image.open(path).convert('L')
    img = img.resize((64, 64), Image.Resampling.LANCZOS)

    # 2) convert to numpy and ensure black background
    arr = np.array(img, dtype=np.uint8)

    # If most pixels are white, invert the image
    if np.mean(arr) > 127:  # if mostly white
        arr = 255 - arr  # invert to make strokes white on black

    # 3) threshold → float32 0.0 or 1.0 (1.0 for strokes)
    arr_bin = (arr > 127).astype(np.float32)

    # 4) to tensor
    tensor = torch.from_numpy(arr_bin) \
        .unsqueeze(0) \
        .unsqueeze(0) \
        .to(DEVICE)
    return tensor

import torch.nn.functional as F


def predict_file50(path: str):
    """
    Yeni: hem en yüksek puanlı sınıfı hem de tüm softmax olasılıklarını döner.
    """
    x = preprocess_image50(path)  # (1,1,64,64)
    with torch.no_grad():
        logits = model50(x)                         # (1,50)
        probs  = F.softmax(logits, dim=1)[0]        # (50,)
        idx    = int(probs.argmax().cpu())
    return classes50[idx], probs.cpu().numpy()
