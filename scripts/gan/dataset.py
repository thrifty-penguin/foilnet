import numpy as np
import torch
from torch.utils.data import Dataset

# class to load the airfoil data
# The data is stored in a numpy array with shape (n_samples, n_points, 2)
class AirfoilDataset(Dataset):
    def __init__(self, path):
        self.data = np.load(path).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
