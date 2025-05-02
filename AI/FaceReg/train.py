import warnings
warnings.filterwarnings("ignore")

import os
import sys
from trainer import Trainer
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


import yaml
import argparse

parser = argparse.ArgumentParser(description='FaceReg')
parser.add_argument('--cfg', dest='cfg', help='settings of model in yaml format')
args = parser.parse_args()

def main():
    cfg = yaml.load(open(args.cfg, 'r'), Loader=yaml.Loader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(cfg, device)
    trainer.train()
    print('Training finished.')
if __name__ == '__main__':
    main()