import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, root_dir: str, labels: list[str], seq_len: int, num_features: int):
        self.root_dir = root_dir
        self.labels = labels
        self.seq_len = seq_len
        self.num_features = num_features
        self.samples = []
        for idx, label in enumerate(labels):
            pattern = os.path.join(root_dir, label, "*.npz")
            for path in glob.glob(pattern):
                self.samples.append((path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        data = np.load(path)["arr_0"].astype(np.float32)  # shape (T,F)
        # pad/trim
        T, F = data.shape
        assert F == self.num_features, f"Expected {self.num_features} got {F} in {path}"
        if T < self.seq_len:
            pad = np.zeros((self.seq_len - T, F), dtype=np.float32)
            data = np.concatenate([pad, data], axis=0)
        elif T > self.seq_len:
            data = data[-self.seq_len:]
        x = torch.from_numpy(data)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
