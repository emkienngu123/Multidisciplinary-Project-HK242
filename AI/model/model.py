import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools

@functools.lru_cache  # use lru_cache to avoid redundant calculation for dim_t
def get_dim_t(num_pos_feats: int, temperature: int, device: torch.device):
    dim_t = torch.arange(num_pos_feats // 2, dtype=torch.float32, device=device)
    dim_t = temperature**(dim_t * 2 / num_pos_feats)
    return dim_t  # (0, 2, 4, ..., ⌊n/2⌋*2)

def get_sine_pos_embed(
    pos_tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi
):
    """Generate sine position embedding for a position tensor

    :param pos_tensor: shape as (..., 2*n).
    :param num_pos_feats: projected shape for each float in the tensor, defaults to 128
    :param temperature: the temperature used for scaling the position embedding, defaults to 10000
    :param exchange_xy: exchange pos x and pos. For example,
        input tensor is [x, y], the results will be [pos(y), pos(x)], defaults to True
    :return: position embedding with shape (None, n * num_pos_feats)
    """
    dim_t = get_dim_t(num_pos_feats, temperature, pos_tensor.device)
    pos_res = pos_tensor.unsqueeze(-1) * scale / dim_t
    pos_res = torch.stack((pos_res.sin(), pos_res.cos()), dim=-1).flatten(-2)
    pos_res = pos_res.flatten(-2)
    return pos_res

class SmartHome(nn.Module):
    def __init__(self, d_in, d_model, num_layers, dropout, bidirectional=False):
        super().__init__()
        self.d_model = d_model
        self.d_in = d_in
        self.num_layers = num_layers
        self.dropout = dropout
        self.movement_embedding = nn.Embedding(2, d_model//4)
        self.encoding = nn.LSTM(d_model, d_model, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.decoding = nn.LSTM(d_model, d_model, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)

        self.fan_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.light_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_in),
            nn.Sigmoid()
        )

        # Initialize parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # Xavier initialization for weights
            elif 'bias' in name:
                nn.init.zeros_(param)  # Zero initialization for biases

    def forward(self, x):
        movement = x[:, :, 3].long()
        movement = self.movement_embedding(movement)
        x = x[:, :, :3]
        x = get_sine_pos_embed(x, self.d_model//4)
        x = torch.cat((x, movement), dim=-1)
        x, _ = self.encoding(x)
        x, _ = self.decoding(x)

        # Use pooling instead of a learnable query
        pooled_output = torch.mean(x, dim=1)  # Global average pooling over the sequence dimension

        fan = self.fan_head(pooled_output)*100
        light = self.light_head(pooled_output)*100
        reconstruction = self.reconstruction_head(x)
        
        # Avoid in-place operation
        reconstruction_scaled = reconstruction.clone()
        reconstruction_scaled[:, :, :3] = reconstruction[:, :, :3] * 100
        return fan, light, reconstruction_scaled