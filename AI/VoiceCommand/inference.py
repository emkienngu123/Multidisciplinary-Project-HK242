from torch.utils.data import DataLoader
from dataset import build_dataset
from model import build_model
from losses import build_loss
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")
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

class Inferencer:
    def __init__(self, device, cfg):
        self.device = device
        self.cfg = cfg
        self.model = build_model(cfg)
        self.model.load_state_dict(torch.load(os.path.join(cfg['train']['save_path'], cfg['inference']['checkpoint'])))
        self.model.to(self.device)
        self.model.eval()
        self.dataset = build_dataset(cfg, 'test.csv', self.device, False)
        self.dataloader = DataLoader(self.dataset, batch_size=cfg['inference']['batch_size'], shuffle=False, num_workers=cfg['inference']['num_workers'])
        self.crieterion = build_loss(cfg).to(device)
    def inference(self):
        all_voice_preds = []

        with torch.no_grad():
            for i, data in enumerate(self.dataloader):
                data = data.to(self.device)
                output = self.model(data)

                all_voice_preds.append(torch.argmax(output, dim=-1).cpu().numpy())
        all_voice_preds = np.concatenate(all_voice_preds, axis=0)
        return all_voice_preds
def main():
    cfg = yaml.load(open(args.cfg, 'r'), Loader=yaml.Loader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inferencer = Inferencer(device, cfg)
    all_voice_preds = inferencer.inference()
    for i in range(all_voice_preds.shape[0]):
        print(f"Sample {inferencer.dataset.__getitem__(i)}: {all_voice_preds[i]}")
if __name__ == '__main__':
    main()