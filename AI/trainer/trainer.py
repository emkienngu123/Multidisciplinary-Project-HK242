from dataset import SmartHomeDataset
from losses import SmartHomeLoss
from model import SmartHome
from optimizer import get_optim, get_scheduler
import torch
import os
from tqdm import tqdm

class Trainer:
    def __init__(self, device, cfg):
        self.device = device
        self.cfg = cfg
        self.model = SmartHome(
            d_in=cfg['model']['d_in'],
            d_model=cfg['model']['d_model'],
            num_layers=cfg['model']['num_layers'],
            dropout=cfg['model']['dropout'],
            bidirectional=cfg['model']['bidirectional']
        ).to(device)

        self.crieterion = SmartHomeLoss(
            weight_fan=cfg['loss']['fan_weight'],
            weight_light=cfg['loss']['light_weight'],
            weight_reg_reconstruction=cfg['loss']['reg_reconstruction_weight'],
            weight_cls_reconstruction=cfg['loss']['cls_reconstruction_weight'],
            regress_type=cfg['loss']['regress_type'],
            beta=cfg['loss']['beta']
        ).to(device)

        self.optimizer = get_optim(
            model=self.model,
            optim_name=cfg['optimizer']['name'],
            lr=cfg['optimizer']['lr'],
            weight_decay=cfg['optimizer']['weight_decay']
        )

        if cfg['scheduler']['name'] == 'reduce_on_plateau':
            self.scheduler = get_scheduler(
                optimizer=self.optimizer,
                scheduler_name=cfg['scheduler']['name'],
                mode=cfg['scheduler']['mode'],
                factor=cfg['scheduler']['factor'],
                patience=cfg['scheduler']['patience'],
                verbose=cfg['scheduler']['verbose']
            )
        elif cfg['scheduler']['name'] == 'cosine':
            self.scheduler = get_scheduler(
                optimizer=self.optimizer,
                scheduler_name=cfg['scheduler']['name'],
                T_max=cfg['scheduler']['T_max'],
                eta_min=cfg['scheduler']['eta_min'],
                last_epoch=cfg['scheduler']['last_epoch']
            )
        elif cfg['scheduler']['name'] == 'step':
            self.scheduler = get_scheduler(
                optimizer=self.optimizer,
                scheduler_name=cfg['scheduler']['name'],
                step_size=cfg['scheduler']['step_size'],
                gamma=cfg['scheduler']['gamma']
            )
        elif cfg['scheduler']['name'] == 'multi_step':
            self.scheduler = get_scheduler(
                optimizer=self.optimizer,
                scheduler_name=cfg['scheduler']['name'],
                milestones=cfg['scheduler']['milestones'],
                gamma=cfg['scheduler']['gamma']
            )
        elif cfg['scheduler']['name'] == 'exponential':
            self.scheduler = get_scheduler(
                optimizer=self.optimizer,
                scheduler_name=cfg['scheduler']['name'],
                gamma=cfg['scheduler']['gamma']
            )
        else:
            self.scheduler = None

        self.train_dataset = SmartHomeDataset(
            base_path=cfg['dataset']['base_path'],
            windown_size=cfg['dataset']['window_size'],
            dataset_path='training.csv',
            train=True
        )
        self.val_dataset = SmartHomeDataset(
            base_path=cfg['dataset']['base_path'],
            windown_size=cfg['dataset']['window_size'],
            dataset_path='validation.csv',
            train=True
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=True,
            num_workers=cfg['train']['num_workers']
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=False,
            num_workers=cfg['train']['num_workers']
        )
        self.epoch = cfg['train']['epoch']
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_model = None
        self.save_path = cfg['train']['save_path']
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loss_logs = {'fan_loss': 0, 'light_loss': 0, 'reconstruction_reg_loss': 0, 'reconstruction_cls_loss': 0}

        for batch_idx, (data, label_fan, label_light) in enumerate(tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}/{self.epoch}")):
            data = data.to(self.device)
            label_fan = label_fan.to(self.device)
            label_light = label_light.to(self.device)
            self.optimizer.zero_grad()
            fan_pred, light_pred, recon_pred = self.model(data)

            # Get total loss and loss dictionary
            loss, loss_dict = self.crieterion(fan_pred, light_pred, recon_pred, label_fan, label_light, data)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            for key in loss_logs:
                loss_logs[key] += loss_dict[key]

        avg_loss = total_loss / len(self.train_dataloader)
        for key in loss_logs:
            loss_logs[key] /= len(self.train_dataloader)

        print(f'Epoch [{epoch}/{self.epoch}], Loss: {avg_loss:.4f}, Loss Breakdown: {loss_logs}')
        return avg_loss, loss_logs

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        loss_logs = {'fan_loss': 0, 'light_loss': 0, 'reconstruction_reg_loss': 0, 'reconstruction_cls_loss': 0}

        with torch.no_grad():
            for batch_idx, (data, label_fan, label_light) in enumerate(tqdm(self.val_dataloader, desc=f"Validating Epoch {epoch}/{self.epoch}")):
                data = data.to(self.device)
                label_fan = label_fan.to(self.device)
                label_light = label_light.to(self.device)
                fan_pred, light_pred, recon_pred = self.model(data)

                # Get total loss and loss dictionary
                loss, loss_dict = self.crieterion(fan_pred, light_pred, recon_pred, label_fan, label_light, data)
                total_loss += loss.item()
                for key in loss_logs:
                    loss_logs[key] += loss_dict[key]

        avg_loss = total_loss / len(self.val_dataloader)
        for key in loss_logs:
            loss_logs[key] /= len(self.val_dataloader)

        print(f'Validation Loss: {avg_loss:.4f}, Loss Breakdown: {loss_logs}')
        return avg_loss, loss_logs

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f'epoch_{epoch}.pth'))
        print(f'Model saved at epoch {epoch}', os.path.join(self.save_path, f'epoch_{epoch}.pth'))

    def save_best_epoch(self):
        if self.best_model is not None:
            torch.save(self.best_model, os.path.join(self.save_path, f'best_epoch_{self.best_epoch}.pth'))
            print(f'Best model saved at epoch {self.best_epoch}')

    def train(self):
        for epoch in range(1, self.epoch + 1):
            train_loss, train_loss_logs = self.train_one_epoch(epoch)
            val_loss, val_loss_logs = self.validate(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_model = self.model.state_dict()
                self.save_model(epoch)
        print(f'Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}')
        self.save_best_epoch()