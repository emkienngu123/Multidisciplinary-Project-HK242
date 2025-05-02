from torch.utils.data import DataLoader
from dataset import build_dataset
from model import build_model
from losses import build_loss
from optimizer import get_optim, get_scheduler
import os
import torch
from tqdm import tqdm
import glob
from torch.nn.functional import cosine_similarity



class Trainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.model = build_model(cfg).to(device)
        self.crietion = build_loss(cfg).to(device)


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

        self.train_dataset = build_dataset(cfg, 'train')
        self.val_dataset = build_dataset(cfg, 'val')
        self.train_dataset = DataLoader(
            dataset=self.train_dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=True,
            num_workers=cfg['train']['num_workers']
        )
        self.val_dataset = DataLoader(
            dataset=self.val_dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=False,
            num_workers=cfg['train']['num_workers']
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

        for batch_idx, (anchor_img, positive_img, negative_img, id) in enumerate(tqdm(self.train_dataset, desc=f"Training Epoch {epoch}/{self.epoch}")):
            anchor_img, positive_img, negative_img = anchor_img, positive_img, negative_img

            self.optimizer.zero_grad()
            anchor_output = self.model(anchor_img)
            positive_output = self.model(positive_img)
            negative_output = self.model(negative_img)

            loss = self.crietion(anchor_output, positive_output, negative_output)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_dataset)
        print(f'Epoch [{epoch}/{self.epoch}], Loss: {avg_loss:.4f}')
        return avg_loss

    def validate(self, epoch):
        self.model.eval()

        # Step 1: Generate embeddings for the training set
        train_embeddings = []
        train_labels = []
        with torch.no_grad():
            for batch_idx, (images, _, _, labels) in enumerate(tqdm(self.train_dataset, desc="Generating Training Embeddings")):
                images = images
                embeddings = self.model(images)
                train_embeddings.append(embeddings)
                train_labels.append(labels)

        train_embeddings = torch.cat(train_embeddings, dim=0)  # Combine all embeddings
        train_labels = torch.cat(train_labels, dim=0)

        # Step 2: Validate using cosine similarity
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (val_images, val_labels) in enumerate(tqdm(self.val_dataset, desc=f"Validating Epoch {epoch}/{self.epoch}")):
                val_images = val_images.to(self.device)
                val_labels = val_labels.to(self.device)
                val_embeddings = self.model(val_images)
                for val_embedding, val_label in zip(val_embeddings, val_labels):
                    # Compute cosine similarity with all training embeddings
                    similarities = cosine_similarity(val_embedding.unsqueeze(0), train_embeddings)
                    # Group similarities by person and average them
                    person_scores = {}
                    for sim, label in zip(similarities, train_labels):
                        if label.item() not in person_scores:
                            person_scores[label.item()] = []
                        person_scores[label.item()].append(sim.item())

                    person_avg_scores = {person: sum(scores) / len(scores) for person, scores in person_scores.items()}

                    # Predict the person with the highest average score
                    predicted_person = max(person_avg_scores, key=person_avg_scores.get)

                    if predicted_person == val_label.item():
                        total_correct += 1
                    total_samples += 1

        accuracy = total_correct / total_samples * 100
        print(f"Validation Accuracy: {accuracy:.2f}%")
        return accuracy

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f'epoch_{epoch}.pth'))
        print(f'Model saved at epoch {epoch}', os.path.join(self.save_path, f'epoch_{epoch}.pth'))

    def save_best_epoch(self):
        if self.best_model is not None:
            torch.save(self.best_model, os.path.join(self.save_path, f'best_epoch_{self.best_epoch}.pth'))
            print(f'Best model saved at epoch {self.best_epoch}')

    def remove_non_best_checkpoints(self):
        """Remove all saved checkpoints except the best checkpoint."""
        checkpoint_files = glob.glob(os.path.join(self.save_path, 'epoch_*.pth'))
        best_checkpoint = os.path.join(self.save_path, f'best_epoch_{self.best_epoch}.pth')

        for checkpoint in checkpoint_files:
            if checkpoint != best_checkpoint:
                os.remove(checkpoint)
                print(f'Removed checkpoint: {checkpoint}')

    def train(self):
        for epoch in range(1, self.epoch + 1):
            train_loss = self.train_one_epoch(epoch)
            val_accuracy = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            # Save the best model based on validation accuracy
            if val_accuracy > self.best_acc:
                self.best_acc = val_accuracy
                self.best_epoch = epoch
                self.best_model = self.model.state_dict()
                self.save_model(epoch)

        print(f'Best validation accuracy: {self.best_acc:.2f}% at epoch {self.best_epoch}')
        self.save_best_epoch()
        self.remove_non_best_checkpoints()