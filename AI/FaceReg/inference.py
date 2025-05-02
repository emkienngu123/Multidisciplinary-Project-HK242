from torch.utils.data import DataLoader
from dataset import build_dataset
from model import build_model
from losses import build_loss
from torch.nn.functional import cosine_similarity
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

parser = argparse.ArgumentParser(description='FaceReg')
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
        self.dataset = build_dataset(cfg, 'test')
        self.dataloader = DataLoader(self.dataset, batch_size=cfg['inference']['batch_size'], shuffle=False, num_workers=cfg['inference']['num_workers'])
        self.id_dict = {
            0: 'KIEN',
            1: 'KIET',
            2: 'LONG',
            3: 'MINH'
        }
    def inference(self):
        all_embedding_predicts = []
        name_imgs = []
        with torch.no_grad():
            for i, (data,name_img) in enumerate(self.dataloader):
                data = data.to(self.device)
                output = self.model(data)
                all_embedding_predicts.append(output)
                name_imgs.append(name_img)
        vector_database_path = self.cfg['inference']['vector_database']
        score_all_class = []
        for i in range(4):
            path_embeds = os.listdir(os.path.join(vector_database_path,self.id_dict[i]))
            vector_database = []
            for path in path_embeds:
                embed = np.load(os.path.join(vector_database_path,self.id_dict[i],path))
                vector_database.append(embed)
            vector_database = np.stack(vector_database, axis=0)
            vector_database = torch.as_tensor(vector_database)
            score_with_class_i = []
            for i, pred_embed in enumerate(all_embedding_predicts):
                cosin_i = cosine_similarity(pred_embed.unsqueeze(0),vector_database)
                cosin_i = cosin_i.mean()
                score_with_class_i.append(cosin_i)
            score_with_class_i = torch.stack(score_with_class_i)
            score_all_class.append(score_with_class_i)
        score_all_class = torch.stack(score_all_class)
        max_score, index = torch.max(score_all_class,dim=0)
        finally_pred = []
        for score,index in zip(max_score, index):
            if score < self.cfg['inference']['threshold']:
                finally_pred.append(None)
            else:
                finally_pred.append(self.id_dict[index.item()])
        return name_imgs, finally_pred
            
        
def main():
    cfg = yaml.load(open(args.cfg, 'r'), Loader=yaml.Loader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inferencer = Inferencer(device, cfg)
    name_imgs, predict_ids = inferencer.inference()
    for name_img, predict_id in zip(name_imgs, predict_ids):
        print('Image ' + name_img[0] + ' is ' + str(predict_id))
if __name__ == '__main__':
    main()