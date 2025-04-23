import torch.nn as nn


class VoiceCommandLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, outputs, targets):
        return self.loss(outputs, targets.float())
def build_loss(cfg):
    return VoiceCommandLoss(cfg)