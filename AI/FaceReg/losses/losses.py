import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, cfg):
        super(TripletLoss, self).__init__()
        self.margin = cfg['loss']['margin']

    def forward(self, anchor, positive, negative):
        # Compute pairwise distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Compute triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
def build_loss(cfg):
    return TripletLoss(cfg)