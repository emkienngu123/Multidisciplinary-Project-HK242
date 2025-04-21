import torch
import torch.nn as nn
import torch.nn.functional as F


class SmartHomeLoss(nn.Module):
    def __init__(self, weight_fan=1.0, weight_light=1.0, weight_reg_reconstruction=1.0, weight_cls_reconstruction=1.0, regress_type='L1', **kwargs):
        super(SmartHomeLoss, self).__init__()
        self.weight_fan = weight_fan
        self.weight_light = weight_light
        self.weight_reg_reconstruction = weight_reg_reconstruction
        self.weight_cls_reconstruction = weight_cls_reconstruction
        if regress_type == 'L1':
            self.fan_loss = nn.L1Loss()
            self.light_loss = nn.L1Loss()
            self.reconstruction_reg_loss = nn.L1Loss()
        elif regress_type == 'L2':
            self.fan_loss = nn.MSELoss()
            self.light_loss = nn.MSELoss()
            self.reconstruction_reg_loss = nn.MSELoss()
        elif regress_type == 'SmoothL1':
            self.fan_loss = nn.SmoothL1Loss(beta=kwargs.get('beta', 1.0))
            self.light_loss = nn.SmoothL1Loss(beta=kwargs.get('beta', 1.0))  
            self.reconstruction_reg_loss = nn.SmoothL1Loss(beta=kwargs.get('beta', 1.0))
        self.reconstruction_cls_loss = nn.CrossEntropyLoss()
    def forward(self, fan, light, reconstruction, fan_target, light_target, reconstruction_target):
        fan_loss = self.fan_loss(fan, fan_target)
        light_loss = self.light_loss(light, light_target)
        reconstruction_reg_loss = self.reconstruction_reg_loss(reconstruction[:, :, :3], reconstruction_target[:, :, :3])
        reconstruction_cls_loss = self.reconstruction_cls_loss(reconstruction[:, :, 3:], reconstruction_target[:, :, 3:].float())
        
        total_loss = (self.weight_fan * fan_loss +
                      self.weight_light * light_loss +
                      self.weight_reg_reconstruction * reconstruction_reg_loss +
                      self.weight_cls_reconstruction * reconstruction_cls_loss)
        
        # Create a loss dictionary for logging
        loss_dict = {
            'fan_loss': fan_loss.item(),
            'light_loss': light_loss.item(),
            'reconstruction_reg_loss': reconstruction_reg_loss.item(),
            'reconstruction_cls_loss': reconstruction_cls_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    def calculate_only_reconstruct(self, reconstruction, reconstruction_target):
        reconstruction_reg_loss = self.reconstruction_reg_loss(reconstruction[:, :, :3], reconstruction_target[:, :, :3])
        reconstruction_cls_loss = self.reconstruction_cls_loss(reconstruction[:, :, 3:], reconstruction_target[:, :, 3:].float())
        
        total_loss = (self.weight_reg_reconstruction * reconstruction_reg_loss +
                      self.weight_cls_reconstruction * reconstruction_cls_loss)
        
        # Create a loss dictionary for logging
        loss_dict = {
            'reconstruction_reg_loss': reconstruction_reg_loss.item(),
            'reconstruction_cls_loss': reconstruction_cls_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


