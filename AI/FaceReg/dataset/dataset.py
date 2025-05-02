from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import random
import numpy as np

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
        self.data = pd.read_csv(os.path.join(self.anno_dir,mode+'.csv'))
        self.id_dict = {
            0: 'KIEN',
            1: 'KIET',
            2: 'LONG',
            3: 'MINH'
        }
        self.name_to_id = {
            'KIEN': 0,
            'KIET': 1,
            'LONG': 2,
            'MINH': 3
        }
        self.num_samples_each_id = {}
        for i in range(4):
            self.num_samples_each_id[self.id_dict[i]] = len(self.data[self.data['id_name'] == self.id_dict[i]])

    def __len__(self):
        return len(self.data)

    def get_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transformation(img)
        if self.augmentation:
            img = self.augmentation(img)
        return img

    def get_item_train(self, idx):
        anchor_img_id = self.data.iloc[idx]['id_name']
        anchor_img_path = os.path.join(self.img_dir, anchor_img_id, f'{self.data.iloc[idx]["id_num"]}.jpg')
        anchor_img = self.get_image(anchor_img_path)

        positive_img_id = self.data.iloc[idx]['id_name']
        positive_id_num = random.choice(
            [id for id in range(self.num_samples_each_id[positive_img_id]) if id != self.data.iloc[idx]['id_num']]
        )
        positive_img_path = os.path.join(self.img_dir, positive_img_id, f'{positive_id_num}.jpg')
        positive_img = self.get_image(positive_img_path)

        negative_img_id = random.choice([id for id in self.id_dict.values() if id != anchor_img_id])
        negative_id_num = random.randint(0, self.num_samples_each_id[negative_img_id] - 1)
        negative_img_path = os.path.join(self.img_dir, negative_img_id, f'{negative_id_num}.jpg')
        negative_img = self.get_image(negative_img_path)

        return anchor_img, positive_img, negative_img, self.name_to_id[anchor_img_id]

    def __getitem__(self, idx):
        if self.mode == 'train':
            anchor_img, positive_img, negative_img, id = self.get_item_train(idx)
            return anchor_img, positive_img, negative_img, id
        else:
            img_id = self.data.iloc[idx]['id_name']
            img_path = os.path.join(self.img_dir, img_id, f'{self.data.iloc[idx]["id_num"]}.jpg')
            img = self.get_image(img_path)
            return img, self.name_to_id[img_id], self.data.iloc[idx]["id_num"]

def build_augmentation(aug_config):
    aug = []
    if aug_config['flip']:
        aug.append(transforms.RandomHorizontalFlip(aug_config['flip']['prob']))
    if aug_config['rotation']:
        aug.append(transforms.RandomRotation(
            degrees=aug_config['rotation']['degrees']
        ))
    if aug_config['color_jitter']:
        aug.append(transforms.ColorJitter(brightness=aug_config['color_jitter']['brightness'],
                                          contrast=aug_config['color_jitter']['contrast'],
                                          saturation=aug_config['color_jitter']['saturation'],
                                          hue=aug_config['color_jitter']['hue']))
    return transforms.Compose(aug) if len(aug) > 0 else None

def build_dataset(cfg, mode='train'):
    return FaceRegDataset(cfg, mode)