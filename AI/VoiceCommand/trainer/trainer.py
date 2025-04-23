from torch.utils.data import DataLoader
from dataset import build_dataset
from model import build_model
from losses import build_loss
from optimizer import get_optim, get_scheduler
import os
import torch

class Trainer:
    def __init__(self, device, cfg):
        self.model = build_model(cfg).to(device)
        self.device = device
        self.cfg = cfg
        self.criterion = build_loss(cfg).to(device)
        

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
        self.train_dataset = build_dataset(cfg, anno_file='train.csv', device=device, training=True)
        self.val_dataset = build_dataset(cfg, anno_file='val.csv', device=device, training=True)
        self.train_dataset = DataLoader(
            dataset=self.train_dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=True,
            num_workers=cfg['train']['num_workers'],
            pin_memory=True
        )
        self.val_dataset = DataLoader(
            dataset=self.val_dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=False,
            num_workers=cfg['train']['num_workers'],
            pin_memory=True
        )
        self.epoch = cfg['train']['epoch']
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_model = None
        self.save_path = cfg['train']['save_path']
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for i, (inputs, labels) in enumerate(self.train_dataset):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_dataset)
        print(f'Epoch [{epoch}/{self.epoch}], Loss: {avg_loss:.4f}')
        return avg_loss
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        accuracy = 0.0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.val_dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                accuracy += (predicted == labels).sum().item()
        avg_loss = total_loss / len(self.val_dataset)
        accuracy = accuracy / len(self.val_dataset.dataset)
        print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        return avg_loss, accuracy
    def save_model(self, epoch):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f'model_epoch_{epoch}.pth'))
        print(f'Model saved at epoch {epoch}')
    def save_best_model(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.best_model, os.path.join(self.save_path, 'best_model.pth'))
        print('Best model saved')
    def train(self):
        for epoch in range(self.epoch):
            train_loss = self.train_one_epoch(epoch)
            val_loss, accuracy = self.validate(epoch)

            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_epoch = epoch
                self.best_model = self.model.state_dict()
                self.save_best_model()

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            if (epoch + 1) % 5 == 0:
                self.save_model(epoch)
        print(f'Training completed. Best accuracy: {self.best_acc:.4f} at epoch {self.best_epoch}')
        self.save_best_model()
        self.remove_non_best_checkpoints()
    def remove_non_best_checkpoints(self):
        for filename in os.listdir(self.save_path):
            if filename.startswith('model_epoch_') and filename != f'model_epoch_{self.best_epoch}.pth':
                os.remove(os.path.join(self.save_path, filename))
                print(f'Removed non-best checkpoint: {filename}')