import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np


class SmartHomeDataset(Dataset):
    def __init__(self, base_path, windown_size, dataset_path, train = True):
        super().__init__()
        self.base_path = base_path
        self.train = train
        self.windown_size = windown_size
        self.data_path = os.path.join(base_path, dataset_path)
        self.data = pd.read_csv(self.data_path)
        self.humidity = np.array(self.data['humidity'])
        self.temperature = np.array(self.data['temperature'])
        self.light = np.array(self.data['light'])
        self.movement = np.array(self.data['movement'])
        self.data_convert = np.concatenate((self.humidity.reshape(-1, 1), self.temperature.reshape(-1, 1), self.light.reshape(-1, 1), self.movement.reshape(-1, 1)), axis=1)
        self.data_convert = torch.tensor(self.data_convert, dtype=torch.float32)
        if self.data_convert.shape[0] % self.windown_size != 0:
            self.data_convert = self.data_convert[:-(self.data_convert.shape[0] % self.windown_size)]
        self.data_convert = self.data_convert.view(-1, self.windown_size, 4)
        if self.train:
            self.label_fan = self.generate_label_for_fan()
            self.label_light = self.generate_label_for_light()
    def generate_label_for_fan(self):
        return torch.mean(F.sigmoid(self.data_convert[:, :, 0] * 0.3 + self.data_convert[:, :, 1] * 0.7 - 40 + self.data_convert[:, :, 3]*10) * 100, dim=1)
    def generate_label_for_light(self):
        return torch.mean(F.sigmoid(-self.data_convert[:, :, 2] * 0.01 + self.data_convert[:, :, 3]*10) * 100, dim=1)
    def __len__(self):
        return self.data_convert.shape[0]
    def __getitem__(self, idx):
        if self.train:
            return self.data_convert[idx], self.label_fan[idx], self.label_light[idx]
        else:
            return self.data_convert[idx]
