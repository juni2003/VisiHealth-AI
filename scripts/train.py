"""
VisiHealth AI - Training Script
Complete training pipeline with multi-task learning, early stopping, and evaluation
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import get_cnn_model, get_bert_model, build_visihealth_model
from data import get_dataloader
from utils.knowledge_graph import load_knowledge_graph, RationaleGenerator


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class Trainer:
    """VisiHealth AI Trainer"""
    
    def __init__(self, config_path='../config.yaml'):
        """Initialize trainer with configuration"""
        # Handle relative paths from scripts directory
        if not os.path.isabs(config_path):
            # Try relative to script directory first
            script_relative = os.path.join(os.path.dirname(__file__), config_path)
            # Also try current working directory
            cwd_path = os.path.join(os.getcwd(), config_path.replace('../', ''))
            
            if os.path.exists(script_relative):
                config_path = script_relative
            elif os.path.exists(cwd_path):
                config_path = cwd_path
            elif os.path.exists(config_path.replace('../', '')):
                config_path = config_path.replace('../', '')
            else:
                # Last resort: try root project directory
                config_path = os.path.join(os.getcwd(), 'config.yaml')
        
        # Load config
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(
            self.config['system']['device'] 
            if torch.cuda.is_available() 
            else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        self.seed = self.config['system']['seed']
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Create directories
        self.save_dir = self.config['system']['save_dir']
        self.log_dir = self.config['system']['log_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize tensorboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(self.log_dir, f'run_{timestamp}'))
        
        # Load datasets
        print("\n" + "="*60)
        print("Loading Datasets...")
        print("="*60)
        self.train_loader, self.train_dataset = get_dataloader(
            data_dir=self.config['dataset']['root_dir'],
            split='train',
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['system']['num_workers'],
            tokenizer_name=self.config['model']['bert']['model_name']
        )
        
        # Compute class weights for imbalanced SLAKE dataset
        self.class_weights = self._compute_class_weights()
        
        # Use training vocabulary for validation to ensure consistency
        self.val_loader, self.val_dataset = get_dataloader(
            data_dir=self.config['dataset']['root_dir'],
            split='validate',
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['system']['num_workers'],
            tokenizer_name=self.config['model']['bert']['model_name'],
            train_vocab=self.train_dataset.answer_vocab  # ← Share vocabulary
        )
        
        # Update num_classes in config
        self.num_classes = self.train_dataset.num_classes
        self.config['model']['cnn']['num_classes'] = self.num_classes
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Answer vocabulary size: {self.num_classes}")
        
        # Verify vocabulary consistency
        assert self.train_dataset.num_classes == self.val_dataset.num_classes, \
            f"Vocabulary mismatch! Train: {self.train_dataset.num_classes}, Val: {self.val_dataset.num_classes}"
        print("✅ Train and validation vocabularies match!")
        
        # Build model
        print("\n" + "="*60)
        print("Building Model...")
        print("="*60)
        self.cnn = get_cnn_model(self.config).to(self.device)
        self.bert = get_bert_model(self.config).to(self.device)
        self.model = build_visihealth_model(self.config, self.cnn, self.bert).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Loss functions with class weighting for imbalanced SLAKE
        self.vqa_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.seg_criterion = nn.BCELoss()
        
        # Optimizer with differential learning rates
        # BERT (pre-trained) gets smaller LR, CNN (from scratch) gets larger LR
        bert_params = list(self.bert.parameters())
        cnn_params = list(self.cnn.parameters())
        fusion_params = [p for p in self.model.parameters() 
                        if id(p) not in [id(bp) for bp in bert_params] 
                        and id(p) not in [id(cp) for cp in cnn_params]]
        
        self.optimizer = optim.Adam([
            {'params': bert_params, 'lr': 2e-5},  # Gentle for pre-trained BERT
            {'params': cnn_params, 'lr': 1e-3},    # Higher for CNN from scratch
            {'params': fusion_params, 'lr': 5e-4}  # Medium for fusion layers
        ], weight_decay=self.config['training']['weight_decay'])
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config['training']['patience']
        )
        
        # Early stopping
        if self.config['training']['early_stopping']['enabled']:
            self.early_stopping = EarlyStopping(
                patience=self.config['training']['early_stopping']['patience'],
                min_delta=self.config['training']['early_stopping']['min_delta']
            )
        else:
            self.early_stopping = None
        
        # Training state
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
        # Load knowledge graph (for evaluation)
        kg_file = self.config['dataset'].get('kg_file', './data/SLAKE/knowledge_graph.txt')
        if os.path.exists(kg_file):
            self.kg = load_knowledge_graph(kg_file)
            self.rationale_gen = RationaleGenerator(self.kg)
        else:
            print(f"Warning: KG file not found at {kg_file}")
            self.kg = None
            self.rationale_gen = None
    
    def _compute_class_weights(self):
        """Compute class weights for handling imbalanced SLAKE dataset"""
        from collections import Counter
        
        # Count answer distribution
        all_answers = []
        for item in self.train_dataset.data:
            answer_text = str(item['answer']).lower().strip()
            answer_idx = self.train_dataset.answer_vocab.get(answer_text, 0)
            all_answers.append(answer_idx)
        
        answer_counts = Counter(all_answers)
        total_samples = len(all_answers)
        
        # Compute weights: inverse frequency
        weights = torch.zeros(self.num_classes)
        for class_idx in range(self.num_classes):
            count = answer_counts.get(class_idx, 1)  # Avoid division by zero
            weights[class_idx] = total_samples / (self.num_classes * count)
        
        print(f"✅ Computed class weights for {self.num_classes} classes")
        return weights.to(self.device)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_vqa_loss = 0.0
        total_seg_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            answers = batch['answer'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, input_ids, attention_mask)
            
            # VQA loss
            vqa_loss = self.vqa_criterion(outputs['answer_logits'], answers)
            
            # Segmentation loss (multi-task learning)
            if self.config['training']['multitask']['enabled']:
                seg_preds = outputs['segmentation_mask'].squeeze(1)
                
                # Upsample predictions to match mask size
                seg_preds = torch.nn.functional.interpolate(
                    seg_preds.unsqueeze(1), 
                    size=masks.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
                
                seg_loss = self.seg_criterion(seg_preds, masks)
                
                # Weighted combination
                vqa_weight = self.config['training']['multitask']['vqa_weight']
                seg_weight = self.config['training']['multitask']['segmentation_weight']
                loss = vqa_weight * vqa_loss + seg_weight * seg_loss
                
                total_seg_loss += seg_loss.item()
            else:
                loss = vqa_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_vqa_loss += vqa_loss.item()
            
            _, predicted = outputs['answer_logits'].max(1)
            total += answers.size(0)
            correct += predicted.eq(answers).sum().item()
            
            # Update progress bar with current batch stats
            current_avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(loss=f'{current_avg_loss:.4f}', acc=f'{100.*correct/total:.2f}%')
            
            # Log to tensorboard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        # Epoch statistics
        avg_loss = total_loss / len(self.train_loader)
        avg_vqa_loss = total_vqa_loss / len(self.train_loader)
        avg_seg_loss = total_seg_loss / len(self.train_loader) if self.config['training']['multitask']['enabled'] else 0
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'vqa_loss': avg_vqa_loss,
            'seg_loss': avg_seg_loss,
            'accuracy': accuracy
        }
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            answers = batch['answer'].to(self.device)
            
            # Forward pass
            outputs = self.model(images, input_ids, attention_mask)
            
            # Loss
            loss = self.vqa_criterion(outputs['answer_logits'], answers)
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = outputs['answer_logits'].max(1)
            total += answers.size(0)
            correct += predicted.eq(answers).sum().item()
            
            # Update progress bar with running averages
            current_avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(loss=f'{current_avg_loss:.4f}', acc=f'{100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'early_stopping_counter': self.early_stopping.counter if self.early_stopping else 0,
            'early_stopping_best_loss': self.early_stopping.best_loss if self.early_stopping else None,
            'config': self.config
        }
        
        # Save last checkpoint
        path = os.path.join(self.save_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, path)
        
        # Save periodic checkpoints every 5 epochs for recovery
        if (self.current_epoch + 1) % 5 == 0:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{self.current_epoch + 1}.pth')
            torch.save(checkpoint, path)
            print(f"Saved checkpoint at epoch {self.current_epoch + 1}")
        
        # Save best checkpoint
        if is_best:
            path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, path)
            print(f"Saved best model (acc: {self.best_val_acc:.2f}%)")
    
    def load_checkpoint(self, resume_path=None):
        """Load checkpoint to resume training"""
        if resume_path and os.path.exists(resume_path):
            checkpoint_path = resume_path
        else:
            # Find latest checkpoint
            checkpoints = sorted([f for f in os.listdir(self.save_dir) if f.startswith('checkpoint_epoch_')])
            if not checkpoints:
                print("No checkpoint found. Starting from scratch.")
                return False
            checkpoint_path = os.path.join(self.save_dir, checkpoints[-1])
        
        print(f"\n{'='*60}")
        print(f"Loading checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Restore early stopping state if available
        if self.early_stopping and 'early_stopping_counter' in checkpoint:
            self.early_stopping.counter = checkpoint['early_stopping_counter']
            self.early_stopping.best_loss = checkpoint['early_stopping_best_loss']
        
        print(f"✅ Resumed from epoch {self.start_epoch}")
        print(f"   Best val accuracy so far: {self.best_val_acc:.2f}%")
        print(f"   Best val loss so far: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return True
    
    def train(self, resume=False, resume_path=None):
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']
        
        # Load checkpoint if resuming
        if resume:
            self.load_checkpoint(resume_path)
        
        print("\n" + "="*60)
        print("Starting Training...")
        print("="*60)
        
        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
            
            # Save best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping(val_metrics['loss'])
                if self.early_stopping.early_stop:
                    print("\nEarly stopping triggered!")
                    break
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print("="*60)
        
        self.writer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VisiHealth AI')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint path to resume from')
    args = parser.parse_args()
    
    # Create trainer and train
    trainer = Trainer(config_path=args.config)
    trainer.train(resume=args.resume, resume_path=args.checkpoint)
