"""
Advanced Training System for Neural Decoders
Includes modern training techniques and evaluation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")
from tqdm import tqdm
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import time


@dataclass
class TrainingConfig:
    """Configuration for advanced training"""
    # Model
    model_type: str = "ensemble"
    num_channels: int = 32
    num_classes: int = 4
    window_size: int = 64
    
    # Training
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Advanced techniques
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    
    # Learning rate schedule
    scheduler_type: str = "cosine"  # cosine, step, exponential, cyclic
    warmup_epochs: int = 5
    
    # Regularization
    dropout_rate: float = 0.5
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    
    # Data augmentation
    augment_data: bool = True
    noise_level: float = 0.1
    time_shift_range: int = 5
    
    # Evaluation
    eval_every_n_epochs: int = 1
    save_best_only: bool = True
    early_stopping_patience: int = 20
    
    # Logging
    use_wandb: bool = True
    project_name: str = "neural-decoder"
    experiment_name: str = "advanced-training"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True


class AdvancedNeuralDataset(Dataset):
    """Advanced dataset with augmentation and preprocessing"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 config: TrainingConfig, transform: Optional[Callable] = None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.config = config
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        # Apply augmentations
        if self.config.augment_data and self.transform:
            x = self.transform(x)
        
        return x, y


class DataAugmentation:
    """Neural signal augmentation techniques"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to neural signal"""
        # Add Gaussian noise
        if self.config.noise_level > 0:
            noise = torch.randn_like(x) * self.config.noise_level
            x = x + noise
        
        # Time shift
        if self.config.time_shift_range > 0:
            shift = np.random.randint(-self.config.time_shift_range, 
                                     self.config.time_shift_range)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)
        
        # Channel dropout (simulate missing channels)
        if np.random.random() < 0.1:  # 10% chance
            num_drop = np.random.randint(1, 3)
            drop_channels = np.random.choice(x.size(0), num_drop, replace=False)
            x[drop_channels] = 0
        
        # Amplitude scaling
        scale = np.random.uniform(0.8, 1.2)
        x = x * scale
        
        return x


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing"""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prb = F.log_softmax(pred, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1).mean()
        return loss


class AdvancedTrainer:
    """Advanced training system with modern techniques"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model.to(config.device)
        self.config = config
        
        # Loss function
        if config.label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(config.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
        self.epochs_without_improvement = 0
        
        # Initialize wandb
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.project_name,
                name=config.experiment_name,
                config=config.__dict__
            )
            wandb.watch(self.model)
        elif config.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Continuing without experiment tracking.")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with different options"""
        # Separate parameters for different learning rates
        params = [
            {'params': self.model.parameters(), 'lr': self.config.learning_rate}
        ]
        
        # You could add parameter groups with different LRs here
        
        optimizer = optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        return optimizer
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        if self.config.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs,
                eta_min=1e-6
            )
        elif self.config.scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler_type == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        elif self.config.scheduler_type == "cyclic":
            scheduler = optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=1e-4,
                max_lr=self.config.learning_rate,
                step_size_up=10,
                mode='triangular2'
            )
        else:
            scheduler = None
        
        # Add warmup
        if self.config.warmup_epochs > 0 and scheduler:
            scheduler = GradualWarmupScheduler(
                self.optimizer,
                multiplier=1,
                total_epoch=self.config.warmup_epochs,
                after_scheduler=scheduler
            )
        
        return scheduler
    
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor, 
                   alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Mixup data augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.config.device), target.to(self.config.device)
            
            # Mixup augmentation
            if self.config.mixup_alpha > 0 and np.random.random() < 0.5:
                data, target_a, target_b, lam = self.mixup_data(
                    data, target, self.config.mixup_alpha
                )
                
            # Mixed precision training
            if self.config.use_mixed_precision:
                with autocast():
                    output = self.model(data)
                    if self.config.mixup_alpha > 0 and 'target_a' in locals():
                        loss = lam * self.criterion(output, target_a) + \
                               (1 - lam) * self.criterion(output, target_b)
                    else:
                        loss = self.criterion(output, target)
            else:
                output = self.model(data)
                if self.config.mixup_alpha > 0 and 'target_a' in locals():
                    loss = lam * self.criterion(output, target_a) + \
                           (1 - lam) * self.criterion(output, target_b)
                else:
                    loss = self.criterion(output, target)
            
            # Gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clipping
                    )
                
                if self.config.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            _, predicted = output.max(1)
            total += target.size(0)
            
            if self.config.mixup_alpha > 0 and 'target_a' in locals():
                correct += (lam * predicted.eq(target_a).sum().item() + 
                           (1 - lam) * predicted.eq(target_b).sum().item())
            else:
                correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Evaluating", leave=False):
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                if self.config.use_mixed_precision:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate additional metrics
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Per-class accuracy
        class_correct = list(0. for i in range(self.config.num_classes))
        class_total = list(0. for i in range(self.config.num_classes))
        
        for t, p in zip(all_targets, all_preds):
            class_correct[t] += (t == p)
            class_total[t] += 1
        
        class_accuracies = [100 * c / t if t > 0 else 0 
                           for c, t in zip(class_correct, class_total)]
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'class_accuracies': class_accuracies,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        print(f"Training {self.config.model_type} model on {self.config.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            self.train_accs.append(train_metrics['accuracy'])
            
            # Validation
            if (epoch + 1) % self.config.eval_every_n_epochs == 0:
                val_metrics = self.evaluate(val_loader)
                self.val_losses.append(val_metrics['loss'])
                self.val_accs.append(val_metrics['accuracy'])
                
                print(f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.2f}%")
                print(f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.2f}%")
                print(f"Class Accuracies: {val_metrics['class_accuracies']}")
                
                # Log to wandb
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_metrics['loss'],
                        'train_acc': train_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                        'val_acc': val_metrics['accuracy'],
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
                    
                    # Log confusion matrix
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(val_metrics['confusion_matrix'], 
                               annot=True, fmt='d', cmap='Blues')
                    plt.title('Confusion Matrix')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    wandb.log({"confusion_matrix": wandb.Image(plt)})
                    plt.close()
                
                # Save best model
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.epochs_without_improvement = 0
                    if self.config.save_best_only:
                        self.save_checkpoint(epoch, val_metrics, is_best=True)
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.CyclicLR):
                    self.scheduler.step()
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if not self.config.save_best_only and (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, train_metrics, is_best=False)
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Final evaluation
        self.plot_training_history()
        
        if self.config.use_wandb:
            wandb.finish()
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'best_val_acc': self.best_val_acc
        }
        
        save_path = Path("checkpoints")
        save_path.mkdir(exist_ok=True)
        
        if is_best:
            torch.save(checkpoint, save_path / "best_model.pth")
        else:
            torch.save(checkpoint, save_path / f"checkpoint_epoch_{epoch + 1}.pth")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(np.arange(0, len(self.val_losses)) * self.config.eval_every_n_epochs, 
                self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(np.arange(0, len(self.val_accs)) * self.config.eval_every_n_epochs,
                self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({"training_history": wandb.Image(plt)})
        
        plt.show()


class GradualWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Gradually warm-up learning rate"""
    
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = self.base_lrs
                    self.finished = True
                return self.after_scheduler.get_lr()
            return self.base_lrs
        
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / 
                self.total_epoch + 1.) for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


# Example usage
if __name__ == "__main__":
    from .advanced_data_generation import AdvancedNeuralDataGenerator, NeuralDataConfig
    from .advanced_models import build_advanced_model
    
    # Generate data
    data_config = NeuralDataConfig(
        num_channels=32,
        num_classes=4,
        sampling_rate=1000,
        trial_length=2.0
    )
    
    generator = AdvancedNeuralDataGenerator(data_config)
    train_data, train_labels = generator.generate_dataset(num_trials=1000)
    val_data, val_labels = generator.generate_dataset(num_trials=200)
    
    # Training configuration
    train_config = TrainingConfig(
        model_type="ensemble",
        num_channels=32,
        num_classes=4,
        window_size=2000,  # 2 seconds at 1000 Hz
        epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        use_wandb=False  # Set to True if you have wandb account
    )
    
    # Create datasets
    augmentation = DataAugmentation(train_config)
    train_dataset = AdvancedNeuralDataset(train_data, train_labels, train_config, augmentation)
    val_dataset = AdvancedNeuralDataset(val_data, val_labels, train_config)
    
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, 
                            shuffle=True, num_workers=train_config.num_workers,
                            pin_memory=train_config.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size,
                          shuffle=False, num_workers=train_config.num_workers,
                          pin_memory=train_config.pin_memory)
    
    # Build model
    model = build_advanced_model(
        train_config.model_type,
        train_config.num_channels,
        train_config.num_classes,
        train_config.window_size
    )
    
    # Train
    trainer = AdvancedTrainer(model, train_config)
    trainer.train(train_loader, val_loader)
