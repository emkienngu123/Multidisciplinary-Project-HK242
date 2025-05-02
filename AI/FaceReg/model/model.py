import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
import numpy as np

class FaceReg(nn.Module):
    def __init__(self, cfg):
        super(FaceReg, self).__init__()
        self.cfg = cfg
        self.embedding_size = cfg['model']['embedding_size']

        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        self.mlp = nn.Sequential(
            nn.Linear(576, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        )
    def forward(self, image):
        backbone_feat = self.backbone(image).flatten(1)
        face_embed = self.mlp(backbone_feat)
        return face_embed
def build_model(cfg):
    return FaceReg(cfg)