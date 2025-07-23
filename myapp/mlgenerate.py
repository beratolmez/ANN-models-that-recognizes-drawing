import os
import json
import random
import numpy as np
import torch
from torch import nn

# ---------------- Ayarlar ----------------
CLASSES = ["butterfly", "house", "star"]
MAX_SEQ_LEN = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SketchGenCond(nn.Module):
    def __init__(self, in_dim=5, hid=512, layers=2,
                 n_cls=len(CLASSES), emb=64, drop=0.3):
        super().__init__()

        self.input_fc = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.LayerNorm(hid),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        self.embed = nn.Embedding(n_cls, emb)

        self.lstm = nn.LSTM(hid + emb, hid, layers,
                            batch_first=True, dropout=drop)

        self.ln = nn.LayerNorm(hid)
        self.drop = nn.Dropout(drop)

        self.fc_xy = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hid//2, 2)
        )
        self.fc_pen = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hid//2, 3)
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, strokes, cls_ids, hidden=None):

        B, T, _ = strokes.size()
        x = self.input_fc(strokes)
        e = self.embed(cls_ids).unsqueeze(1).expand(-1, T, -1)
        out, hidden = self.lstm(torch.cat([x, e], dim=2), hidden)
        out = self.drop(self.ln(out))
        xy = self.fc_xy(out)
        pen = self.fc_pen(out)
        return xy, pen, hidden


def sample_sequence(model, cls_id, max_len=MAX_SEQ_LEN):

    model.eval()
    cls = torch.tensor([cls_id], device=DEVICE)
    hidden = None

    inp = torch.tensor([[[0, 0, 0, 1, 0]]], dtype=torch.float32, device=DEVICE)
    seq = []
    with torch.no_grad():
        for _ in range(max_len - 1):
            xy, pen_logits, hidden = model(inp, cls, hidden)
            last_xy = xy[0, -1]
            dx, dy = last_xy[0].item(), last_xy[1].item()
            probs = torch.softmax(pen_logits[0, -1], dim=0).cpu().numpy()
            pen = np.random.choice(3, p=probs)
            step = [
                dx, dy,
                1 if pen == 0 else 0,
                1 if pen == 1 else 0,
                1 if pen == 2 else 0
            ]
            seq.append(step)
            inp = torch.tensor([[[*step]]], dtype=torch.float32, device=DEVICE)
    model.train()
    return seq
