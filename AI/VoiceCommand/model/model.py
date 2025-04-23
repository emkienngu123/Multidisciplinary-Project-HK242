import torch
import torch.nn as nn
import torch.functional as F


class VoiceCommand(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_in = cfg['dataset']['transformation']['n_mfcc']
        self.d_model = cfg['model']['d_model']
        self.num_layers = cfg['model']['num_layers']
        self.dropout = cfg['model']['dropout']
        self.bidirectional = cfg['model']['bidirectional']
        self.d_out = cfg['model']['d_out']

        self.feat_extract = nn.LSTM(
            input_size = self.d_in,
            hidden_size= self.d_model,
            num_layers= self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
        self.cls_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, self.d_out),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        x, _ = self.feat_extract(x)
        x = torch.mean(x, dim=1)
        x = self.cls_head(x)
        return x
def build_model(cfg):
    return VoiceCommand(cfg)