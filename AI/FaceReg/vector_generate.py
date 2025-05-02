import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


import yaml
import argparse

parser = argparse.ArgumentParser(description='SmartHome')
parser.add_argument('--cfg', dest='cfg', help='settings of model in yaml format')
args = parser.parse_args()
from torch.utils.data import DataLoader
from dataset import build_dataset
from model import build_model
import numpy as np

def generate_vector_for_train_data(cfg, device):
    # Load the trained model
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg['train']['save_path'], cfg['inference']['checkpoint'])))
    model.eval()

    # Load the training dataset
    train_dataset = build_dataset(cfg, mode='trainval')
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg['inference']['batch_size'],
        shuffle=False,
        num_workers=cfg['inference']['num_workers']
    )

    # Prepare to store embeddings and labels
    embeddings = []
    labels = []
    id_nums = []

    with torch.no_grad():
        for images, label, id_num in train_dataloader:
            images = images.to(device)
            embedding = model(images)  # Generate embedding vectors
            embeddings.append(embedding.cpu())
            labels.append(label)
            id_nums.append(id_num)
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    id_nums = torch.cat(id_nums, dim=0).numpy()
    # Save embeddings and labels to files
    mapping_id_to_name = {
            0: 'KIEN',
            1: 'KIET',
            2: 'LONG',
            3: 'MINH'
    }
    for embedding, label, id_num in zip(embeddings, labels, id_nums):
        print('Save ' + str(embedding.shape) + ' to ' + str(os.path.join(cfg['inference']['vector_database'], mapping_id_to_name[label], str(id_num)+'.npy')))
        np.save(os.path.join(cfg['inference']['vector_database'], mapping_id_to_name[label], str(id_num)+'.npy'), embedding)
def main():
    cfg = yaml.load(open(args.cfg, 'r'), Loader=yaml.Loader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generate_vector_for_train_data(cfg, device)
    print('Generating finished.')
if __name__ == '__main__':
    main()