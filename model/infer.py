import os
import numpy as np
import torch
from .model import LSTMClassifier


def load_model(num_features: int, num_classes: int, weights_path: str):
    model = LSTMClassifier(num_features=num_features, num_classes=num_classes)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model


def predict_sequence(model, seq_np: np.ndarray):
    """seq_np: (T,F) numpy float32
    returns (pred_index, probs_np)
    """
    with torch.no_grad():
        x = torch.from_numpy(seq_np[None, :, :])  # (1,T,F)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return idx, probs
