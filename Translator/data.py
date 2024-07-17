import math
import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle=shuffle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return np.squeeze(self.data[idx][0]), np.squeeze(self.data[idx][1])