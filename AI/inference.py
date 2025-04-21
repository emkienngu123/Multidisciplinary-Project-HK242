from dataset import SmartHomeDataset
from model import SmartHome
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from losses import SmartHomeLoss
import numpy as np

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
parser.add_argument('--cfg', dest='cfg', help='settings of detection in yaml format')
args = parser.parse_args()
class Inferencer:
    def __init__(self, device, cfg):
        self.device = device
        self.cfg = cfg
        self.model = SmartHome(
            d_in=cfg['model']['d_in'],
            d_model=cfg['model']['d_model'],
            num_layers=cfg['model']['num_layers'],
            dropout=cfg['model']['dropout'],
            bidirectional=cfg['model']['bidirectional']
        )
        self.model.load_state_dict(torch.load(os.path.join(cfg['train']['save_path'], cfg['inference']['checkpoint'])))
        self.model.to(self.device)
        self.model.eval()
        self.dataset = SmartHomeDataset(cfg['dataset']['base_path'], cfg['dataset']['window_size'], 'testing.csv', False)
        self.dataloader = DataLoader(self.dataset, batch_size=cfg['inference']['batch_size'], shuffle=False, num_workers=cfg['inference']['num_workers'])
        self.crieterion = SmartHomeLoss(
            weight_fan=cfg['loss']['fan_weight'],
            weight_light=cfg['loss']['light_weight'],
            weight_reg_reconstruction=cfg['loss']['reconstruct_regress_weight'],
            weight_cls_reconstruction=cfg['loss']['reconstruct_cls_weight'],
            regress_type=cfg['loss']['regress_type'],
            beta=cfg['loss']['beta']
        ).to(device)
        self.threshold_anomaly = cfg['inference']['threshold_anomaly']
    def inference(self):
        all_fan_preds = []
        all_light_preds = []
        anomaly_samples = []

        with torch.no_grad():
            for i, data in enumerate(self.dataloader):
                data = data.to(self.device)
                fan_pred, light_pred, recon_pred = self.model(data)

                # Collect fan and light predictions
                all_fan_preds.append(fan_pred.cpu().numpy())
                all_light_preds.append(light_pred.cpu().numpy())

                # Calculate reconstruction error
                reconstruct_error, reconstruct_dict = self.crieterion.calculate_only_reconstruct(recon_pred, data)

                # Identify anomaly samples
                anomaly_mask = reconstruct_error > self.threshold_anomaly
                if anomaly_mask.any():
                    anomaly_samples.append((i, data[anomaly_mask].cpu().numpy(), reconstruct_dict))

        # Concatenate all predictions and anomaly samples
        all_fan_preds = np.concatenate(all_fan_preds, axis=0)
        all_light_preds = np.concatenate(all_light_preds, axis=0)

        return all_fan_preds, all_light_preds, anomaly_samples

def main():
    cfg = yaml.load(open(args.cfg, 'r'), Loader=yaml.Loader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inferencer = Inferencer(device, cfg)
    fan_preds, light_preds, anomaly_samples = inferencer.inference()

    print('Fan Predictions:', fan_preds)
    print('Light Predictions:', light_preds)

    print('\nAnomaly Samples:')
    if len(anomaly_samples) == 0:
        print('No anomalies detected.')
    else:
        for idx, (batch_idx, anomaly_data, reconstruct_dict) in enumerate(anomaly_samples):
            print(f'\nAnomaly {idx + 1}:')
            print(f'  Batch Index: {batch_idx}')
            print(f'  Anomaly Data: {anomaly_data}')
            print(f'  Reconstruction Details: {reconstruct_dict}')

    print('\nInference finished.')
if __name__ == '__main__':
    main()
        