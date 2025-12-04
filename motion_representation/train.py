"""
Training script for Motion VAE with GRU layers.
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add motion_representation to path
motion_rep_path = Path(__file__).parent
sys.path.insert(0, str(motion_rep_path.parent))

from motion_representation.models import MotionModel, vae_loss
# Backward compatibility
MotionVAE = MotionModel
from motion_representation.data.motion_dataset import create_dataloader
from motion_representation.config import get_config


def train_epoch(model, dataloader, optimizer, device, epoch, config, writer=None, scaler=None):
    """Train for one epoch with optional gradient accumulation and mixed precision."""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    
    # Gradient accumulation
    accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    use_amp = getattr(config, 'use_amp', False)
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, motion in enumerate(pbar):
        motion = motion.to(device)  # (batch, seq_len, dim)
        
        # Forward pass with mixed precision if enabled
        if use_amp and scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                recon_motion, mean, logvar, z = model(motion)
                
                # Check for NaN in model outputs - STOP TRAINING if detected
                if torch.isnan(recon_motion).any() or torch.isnan(mean).any() or torch.isnan(logvar).any():
                    print(f"\n❌ ERROR: NaN detected in model outputs at epoch {epoch}, batch {batch_idx}")
                    print(f"  recon_motion NaN: {torch.isnan(recon_motion).sum().item()}")
                    print(f"  mean NaN: {torch.isnan(mean).sum().item()}")
                    print(f"  logvar NaN: {torch.isnan(logvar).sum().item()}")
                    print(f"  logvar range: [{logvar.min().item():.4f}, {logvar.max().item():.4f}]")
                    print("🛑 Stopping training due to NaN in model outputs!")
                    raise RuntimeError("NaN detected in model outputs - training stopped")
                
                # Compute loss within autocast context
                loss, recon_loss, kl_loss = vae_loss(
                    recon_motion, motion, mean, logvar,
                    recon_weight=config.recon_weight,
                    kl_weight=config.kl_weight,
                    use_vae=config.use_vae
                )
        else:
            recon_motion, mean, logvar, z = model(motion)
            
            # Check for NaN in model outputs - STOP TRAINING if detected
            if torch.isnan(recon_motion).any() or torch.isnan(mean).any() or torch.isnan(logvar).any():
                print(f"\n❌ ERROR: NaN detected in model outputs at epoch {epoch}, batch {batch_idx}")
                print(f"  recon_motion NaN: {torch.isnan(recon_motion).sum().item()}")
                print(f"  mean NaN: {torch.isnan(mean).sum().item()}")
                print(f"  logvar NaN: {torch.isnan(logvar).sum().item()}")
                print(f"  logvar range: [{logvar.min().item():.4f}, {logvar.max().item():.4f}]")
                print("🛑 Stopping training due to NaN in model outputs!")
                raise RuntimeError("NaN detected in model outputs - training stopped")
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(
                recon_motion, motion, mean, logvar,
                recon_weight=config.recon_weight,
                kl_weight=config.kl_weight,
                use_vae=config.use_vae
            )
        
        # Check for NaN in loss - STOP TRAINING if detected
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ ERROR: NaN/Inf loss at epoch {epoch}, batch {batch_idx}")
            print(f"  recon_loss: {recon_loss.item():.4f}, kl_loss: {kl_loss.item():.4f}")
            print("🛑 Stopping training due to NaN/Inf in loss!")
            raise RuntimeError("NaN/Inf detected in loss - training stopped")
        
        # Backward pass with gradient accumulation
        if use_amp and scaler is not None:
            scaler.scale(loss / accumulation_steps).backward()
        else:
            (loss / accumulation_steps).backward()
        
        # Update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            # Check for NaN gradients - STOP TRAINING if detected
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"\n❌ ERROR: NaN gradient detected in {name} at epoch {epoch}, batch {batch_idx}")
                    print("🛑 Stopping training due to NaN gradients!")
                    raise RuntimeError(f"NaN gradient detected in {name} - training stopped")
            
            # Gradient clipping (more aggressive for transformers)
            # Use very small clipping for transformers to prevent NaN
            max_norm = 0.1 if config.encoder_type == 'transformer' or config.decoder_type == 'transformer' else 1.0
            
            # Clip gradients with detailed error reporting
            try:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                
                # Check for NaN in gradient norm
                if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)):
                    print(f"Warning: NaN/Inf gradient norm: {grad_norm}")
                    # Check which parameters have NaN gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"  NaN gradient in: {name}")
                            # Zero out NaN gradients
                            param.grad[torch.isnan(param.grad)] = 0.0
            except RuntimeError as e:
                print(f"Error during gradient clipping: {e}")
                # Zero out all gradients if clipping fails
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                raise RuntimeError("Gradient clipping failed - stopping training")
            
            # Optimizer step
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.6f}'
        })
        
        # Log to tensorboard
        if writer is not None and batch_idx % config.log_every == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/ReconLoss', recon_loss.item(), global_step)
            writer.add_scalar('Train/KLLoss', kl_loss.item(), global_step)
    
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def validate(model, dataloader, device, config):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for motion in tqdm(dataloader, desc="Validation"):
            motion = motion.to(device)
            
            # Forward pass
            recon_motion, mean, logvar, z = model(motion)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(
                recon_motion, motion, mean, logvar,
                recon_weight=config.recon_weight,
                kl_weight=config.kl_weight,
                use_vae=config.use_vae
            )
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def save_config_txt(config, checkpoint_dir):
    """Save model configuration to a text file for reference."""
    config_path = os.path.join(checkpoint_dir, 'model_config.txt')
    with open(config_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Motion Model Configuration\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Model Architecture:\n")
        f.write(f"  Encoder Type: {config.encoder_type}\n")
        f.write(f"  Decoder Type: {config.decoder_type}\n")
        f.write(f"  Use VAE: {config.use_vae}\n")
        f.write("\n")
        
        f.write("Model Parameters:\n")
        f.write(f"  Input Dimension: {config.input_dim}\n")
        f.write(f"  Hidden Dimension: {config.hidden_dim}\n")
        f.write(f"  Number of Layers: {config.num_layers}\n")
        f.write(f"  Latent Dimension: {config.latent_dim}\n")
        f.write(f"  Sequence Length: {config.seq_len}\n")
        f.write(f"  Dropout: {config.dropout}\n")
        f.write("\n")
        
        if config.encoder_type == 'transformer' or config.decoder_type == 'transformer':
            f.write("Transformer Parameters:\n")
            f.write(f"  Number of Heads: {config.num_heads}\n")
            f.write(f"  Feed-Forward Size: {config.ff_size}\n")
            f.write(f"  Activation: {config.activation}\n")
            f.write("\n")
        
        f.write("Training Parameters:\n")
        f.write(f"  Batch Size: {config.batch_size}\n")
        f.write(f"  Gradient Accumulation Steps: {getattr(config, 'gradient_accumulation_steps', 1)}\n")
        f.write(f"  Learning Rate: {config.learning_rate}\n")
        f.write(f"  Weight Decay: {config.weight_decay}\n")
        f.write(f"  Number of Epochs: {config.num_epochs}\n")
        f.write(f"  Use AMP: {getattr(config, 'use_amp', False)}\n")
        f.write(f"  LR Scheduler: {config.lr_scheduler}\n")
        f.write(f"  Reconstruction Weight: {config.recon_weight}\n")
        f.write(f"  KL Weight: {config.kl_weight}\n")
        f.write("\n")
        
        f.write("Data Parameters:\n")
        f.write(f"  LMDB Directory: {config.lmdb_dir}\n")
        f.write(f"  Window Size: {config.window_size}\n")
        f.write(f"  Stride: {config.stride}\n")
        f.write(f"  Use Both Roles: {config.use_both_roles}\n")
        f.write(f"  Is MDM Format: {getattr(config, 'is_MDM', True)}\n")
        f.write("\n")
        
        f.write("Checkpoint Parameters:\n")
        f.write(f"  Checkpoint Directory: {config.checkpoint_dir}\n")
        f.write(f"  Model Name: {getattr(config, 'model_name', 'None')}\n")
        f.write(f"  Save Every: {config.save_every} epochs\n")
        f.write(f"  Log Every: {config.log_every} iterations\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Saved model configuration to: {config_path}")


def save_checkpoint(model, optimizer, epoch, loss, config, is_best=False, scheduler=None, scaler=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'input_dim': config.input_dim,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'latent_dim': config.latent_dim,
            'seq_len': config.seq_len,
            'dropout': config.dropout,
            'encoder_type': config.encoder_type,
            'decoder_type': config.decoder_type,
            'num_heads': config.num_heads,
            'ff_size': config.ff_size,
            'activation': config.activation,
            'use_vae': config.use_vae,
        }
    }
    
    # Save scheduler state if available
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save scaler state if available
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Always save best checkpoint
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint (loss: {loss:.4f})")
    
    # Save checkpoint every save_every epochs (default: every epoch)
    if (epoch + 1) % config.save_every == 0:
        checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Keep every 10th epoch (10, 20, 30, etc.) plus last 10 epochs
    checkpoints = sorted([f for f in os.listdir(config.checkpoint_dir) 
                          if f.startswith('checkpoint_epoch_') and f.endswith('.pth')],
                        key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(checkpoints) > 10:
        # Get epoch numbers
        epoch_nums = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
        max_epoch = max(epoch_nums)
        
        # Keep: every 10th epoch (10, 20, 30, ...) + last 10 epochs
        keep_epochs = set()
        # Every 10th epoch
        for e in range(10, max_epoch + 1, 10):
            keep_epochs.add(e)
        # Last 10 epochs
        keep_epochs.update(range(max(1, max_epoch - 9), max_epoch + 1))
        
        # Remove checkpoints not in keep_epochs
        for checkpoint_file in checkpoints:
            epoch_num = int(checkpoint_file.split('_')[-1].split('.')[0])
            if epoch_num not in keep_epochs:
                old_path = os.path.join(config.checkpoint_dir, checkpoint_file)
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"Removed old checkpoint: {checkpoint_file}")


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoint files
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('checkpoint_epoch_') and f.endswith('.pth'):
            try:
                epoch_num = int(f.split('_')[-1].split('.')[0])
                checkpoint_files.append((epoch_num, os.path.join(checkpoint_dir, f)))
            except ValueError:
                continue
    
    if not checkpoint_files:
        # Also check for best_checkpoint.pth
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
        if os.path.exists(best_path):
            return best_path
        return None
    
    # Return the checkpoint with highest epoch number
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    return checkpoint_files[0][1]


def load_checkpoint(model, optimizer, checkpoint_path, device, scheduler=None, scaler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    # Restore scheduler state if available
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print("Warning: Could not load scheduler state")
    
    # Restore scaler state if available
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except:
            print("Warning: Could not load scaler state")
    
    return epoch, loss


def main():
    """Main training function."""
    config = get_config()
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading dataset...")
    train_loader = create_dataloader(
        args=config,
        lmdb_dir=config.lmdb_dir,
        window_size=config.window_size,
        stride=config.stride,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        use_both_roles=config.use_both_roles,
    )
    
    print(f"Dataset size: {len(train_loader.dataset)} samples")
    
    # Save configuration to text file
    save_config_txt(config, config.checkpoint_dir)
    
    # Create model
    print("Creating model...")
    print(f"Architecture: Encoder={config.encoder_type}, Decoder={config.decoder_type}")
    print(f"Mode: {'VAE' if config.use_vae else 'Vanilla Autoencoder'}")
    model = MotionModel(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        latent_dim=config.latent_dim,
        seq_len=config.seq_len,
        dropout=config.dropout,
        encoder_type=config.encoder_type,
        decoder_type=config.decoder_type,
        num_heads=config.num_heads,
        ff_size=config.ff_size,
        activation=config.activation,
        use_vae=config.use_vae,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    # Use lower learning rate for transformer to prevent instability
    base_lr = config.learning_rate
    if config.encoder_type == 'transformer' or config.decoder_type == 'transformer':
        base_lr = base_lr * 0.5  # Reduce LR for transformer
        print(f"Using reduced learning rate {base_lr} for transformer architecture")
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=base_lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),  # Standard Adam betas
        eps=1e-8
    )
    
    # Learning rate scheduler (optional)
    scheduler = None
    if config.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs, eta_min=base_lr * 0.01
        )
    elif config.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.num_epochs // 3, gamma=0.5
        )
    
    # Mixed precision training scaler (if enabled)
    scaler = None
    if getattr(config, 'use_amp', False):
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("Using Automatic Mixed Precision (AMP) for training")
    
    # Print memory optimization info
    if hasattr(config, 'gradient_accumulation_steps') and config.gradient_accumulation_steps > 1:
        effective_batch_size = config.batch_size * config.gradient_accumulation_steps
        print(f"Using gradient accumulation: {config.gradient_accumulation_steps} steps")
        print(f"Effective batch size: {effective_batch_size} (actual batch size: {config.batch_size})")
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float('inf')
    resume_path = None
    
    if config.resume:
        # Use explicitly specified checkpoint
        resume_path = config.resume
        print(f"Resuming from specified checkpoint: {resume_path}")
    else:
        # Auto-detect latest checkpoint
        resume_path = find_latest_checkpoint(config.checkpoint_dir)
        if resume_path:
            print(f"Auto-detected latest checkpoint: {resume_path}")
    
    if resume_path:
        if not os.path.exists(resume_path):
            print(f"Warning: Checkpoint not found: {resume_path}. Starting from scratch.")
        else:
            start_epoch, best_loss = load_checkpoint(model, optimizer, resume_path, device, scheduler, scaler)
            print(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.4f}")
            start_epoch += 1  # Start from next epoch
            # When resuming, num_epochs means "train N more epochs"
            end_epoch = start_epoch + config.num_epochs - 1
            print(f"Continuing training from epoch {start_epoch} for {config.num_epochs} more epochs (until epoch {end_epoch})")
    else:
        # Fresh training: num_epochs means total epochs
        end_epoch = config.num_epochs - 1
    
    # Create tensorboard writer
    log_dir = os.path.join(config.checkpoint_dir, 'logs')
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print("Starting training...")
    try:
        for epoch in range(start_epoch, end_epoch + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{config.num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_recon, train_kl = train_epoch(
                model, train_loader, optimizer, device, epoch, config, writer, scaler
            )
            
            print(f"Train Loss: {train_loss:.4f} | Recon: {train_recon:.4f} | KL: {train_kl:.6f}")
            
            # Check for NaN in epoch losses - STOP TRAINING if detected
            import math
            if math.isnan(train_loss) or math.isnan(train_recon) or math.isnan(train_kl):
                print(f"\n❌ ERROR: NaN detected in epoch {epoch+1} losses!")
                print(f"   Loss: {train_loss}, Recon: {train_recon}, KL: {train_kl}")
                print("🛑 Stopping training due to NaN in epoch losses!")
                raise RuntimeError("NaN detected in epoch losses - training stopped")
            
            # Log epoch metrics
            writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            writer.add_scalar('Epoch/TrainReconLoss', train_recon, epoch)
            writer.add_scalar('Epoch/TrainKLLoss', train_kl, epoch)
            
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar('Epoch/LearningRate', current_lr, epoch)
            
            # Update best loss and save checkpoint
            is_best = train_loss < best_loss
            if is_best:
                best_loss = train_loss
            
            # Save checkpoint every epoch (save_every=1) and best checkpoint
            # Epoch numbering continues from where we resumed
            save_checkpoint(model, optimizer, epoch, train_loss, config, is_best, scheduler, scaler)
            
    except RuntimeError as e:
        print(f"\n{'='*60}")
        print("Training stopped due to error:")
        print(str(e))
        print(f"{'='*60}")
        raise
    
    writer.close()
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
