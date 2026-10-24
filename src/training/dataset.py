import numpy as np
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("x must be [n, features] and match y")
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
