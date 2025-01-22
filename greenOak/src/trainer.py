import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Dict, List
import wandb
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
try:
    from .model import TestCaseGenerator
    from .dataset import TestCaseDataset
except ImportError:
    from model import TestCaseGenerator
    from dataset import TestCaseDataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class TestCaseTrainer:
    def __init__(self,
                 model: TestCaseGenerator,
                 train_dataset: TestCaseDataset,
                 val_dataset: TestCaseDataset,
                 learning_rate: float = 2e-5,
                 batch_size: int = 8,
                 num_epochs: int = 3,
                 warmup_steps: int = 0,
                 logging_steps: int = 100,
                 save_steps: int = 1000,
                 output_dir: str = "models"):
        
        self.model = model
        self.device = model.device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(DataLoader(train_dataset, batch_size=batch_size, shuffle=True)) * num_epochs
        )
        
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {
            'precision': [],
            'recall': [],
            'f1': []
        }

    def train(self):
        """Train the model"""
        wandb.init(project="greenoak-test-generator")
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            metrics = self.train_epoch()
            self.train_losses.append(metrics['loss'])
            self.val_losses.append(metrics['val_loss'])
            
            # Update metrics history
            self.metrics_history['precision'].append(0)
            self.metrics_history['recall'].append(0)
            self.metrics_history['f1'].append(0)
            
            # Save best model
            if metrics['val_loss'] < best_val_loss:
                best_val_loss = metrics['val_loss']
                self.save_checkpoint('best')
            
            # Generate and save visualizations
            self.generate_visualizations(epoch)
            
            # Log epoch metrics
            wandb.log({
                'epoch': epoch,
                'train_loss': metrics['loss'],
                'val_loss': metrics['val_loss'],
                'val_precision': 0,
                'val_recall': 0,
                'val_f1': 0
            })

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Create DataLoader
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Training loop
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.model.device)
            attention_mask = batch['attention_mask'].to(self.model.device)
            labels = batch['labels'].to(self.model.device)
            
            # Forward pass
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Update progress bar
            num_batches += 1
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Evaluate on validation set
        val_metrics = self.evaluate()
        
        metrics = {
            'loss': total_loss / num_batches,
            'val_loss': val_metrics['loss']
        }
        
        return metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches
        }

    def generate_visualizations(self, epoch: int):
        """Generate training visualizations"""
        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # Loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(vis_dir / f'loss_curves_epoch_{epoch}.png')
        plt.close()
        
        # Metrics over time
        plt.figure(figsize=(12, 6))
        for metric, values in self.metrics_history.items():
            plt.plot(values, label=metric.capitalize())
        plt.title('Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(vis_dir / f'metrics_epoch_{epoch}.png')
        plt.close()
        
        # Learning rate schedule
        plt.figure(figsize=(10, 6))
        lrs = [self.scheduler.get_lr()[0] for _ in range(len(DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)))]
        plt.plot(lrs)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.savefig(vis_dir / f'lr_schedule_epoch_{epoch}.png')
        plt.close()

    def log_metrics(self, step: int, loss: float, outputs: Dict):
        """Log metrics to wandb"""
        wandb.log({
            'step': step,
            'loss': loss,
            'learning_rate': self.scheduler.get_lr()[0]
        })

    def save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / f'checkpoint-{step}'
        checkpoint_dir.mkdir(exist_ok=True)
        self.model.save_model(str(checkpoint_dir))
        
        # Save optimizer and scheduler
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, checkpoint_dir / 'optimizer.pt')
