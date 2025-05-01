import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms

class FaceRegDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        super().__init__()
        self.mode = mode
        self.anno_dir = cfg['dataset']['anno_dir']
        self.img_dir = cfg['dataset']['img_dir']
        self.img_size = cfg['dataset']['img_size']
        self.transformation = transforms.Compose([
            transforms.Resize((self.img_size[0], self.img_size[1])),
            transforms.ToTensor()
        ])
        self.augmentation = build_augmentation(cfg['dataset']['augmentation']) if mode == 'train' else None
        self.data = pd.read_csv(os.path.join(self.anno_dir, 'train.csv')) if mode == 'train' else pd.read_csv(os.path.join(self.anno_dir, 'val.csv')) if mode == 'val' else pd.read_csv(os.path.join(self.anno_dir, 'test.csv'))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        pass

def build_augmentation(aug_config):
    aug = []
    if aug_config['flip']:
        aug.append(transforms.RandomHorizontalFlip())
    if aug_config['rotation']:
        aug.append(transforms.RandomRotation(aug_config['rotation']))
    if aug_config['color_jitter']:
        aug.append(transforms.ColorJitter(brightness=aug_config['color_jitter']['brightness'],
                                          contrast=aug_config['color_jitter']['contrast'],
                                          saturation=aug_config['color_jitter']['saturation'],
                                          hue=aug_config['color_jitter']['hue']))
    return transforms.Compose(aug) if len(aug) > 0 else None