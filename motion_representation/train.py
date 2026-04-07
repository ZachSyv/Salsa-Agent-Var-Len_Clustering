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


import warnings
warnings.filterwarnings("ignore")

DEBUG_PRINTS = False

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    """
    Linear warm-up learning rate scheduler.
    Matches T2M-GPT warm-up implementation exactly.
    
    Args:
        optimizer: Optimizer to update
        nb_iter: Current iteration number (1-indexed, starts from 1)
        warm_up_iter: Total warm-up iterations
        lr: Target learning rate after warm-up
        
    Returns:
        optimizer: Updated optimizer
        current_lr: Current learning rate
    """
    # T2M-GPT formula: lr * (nb_iter + 1) / (warm_up_iter + 1)
    # Note: nb_iter is 1-indexed in T2M-GPT (range(1, warm_up_iter+1))
    # Our nb_iter is 0-indexed, so we use (nb_iter + 1)
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr


def train_epoch(model, dataloader, optimizer, device, epoch, config, writer=None, scaler=None):
    """Train for one epoch with optional gradient accumulation and mixed precision."""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_vel_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0
    
    # Gradient accumulation
    accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    use_amp = getattr(config, 'use_amp', False)
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            motion, mask = batch
            mask = mask.to(device)
        else:
            motion = batch
            mask = None
        motion = motion.to(device)  # (batch, seq_len, dim)
        
        # DATA VALIDATION: Check for corrupted input data
        has_nan_input = torch.isnan(motion).any().item()
        has_inf_input = torch.isinf(motion).any().item()
        has_extreme_values = (motion.abs() > 1e6).any().item()
        
        if has_nan_input or has_inf_input or has_extreme_values:
            print(f"\n⚠️  WARNING: Corrupted input data detected at epoch {epoch}, batch {batch_idx}!")
            print(f"  motion shape: {motion.shape}")
            print(f"  motion stats: min={motion.min().item():.4f}, max={motion.max().item():.4f}, mean={motion.mean().item():.4f}, std={motion.std().item():.4f}")
            print(f"  has NaN: {has_nan_input}, has Inf: {has_inf_input}, has extreme values (>1e6): {has_extreme_values}")
            if has_nan_input:
                nan_count = torch.isnan(motion).sum().item()
                nan_indices = torch.where(torch.isnan(motion))
                print(f"  NaN count: {nan_count}, NaN positions: {list(zip(nan_indices[0].cpu().numpy()[:10], nan_indices[1].cpu().numpy()[:10], nan_indices[2].cpu().numpy()[:10]))}")
            if has_inf_input:
                inf_count = torch.isinf(motion).sum().item()
                print(f"  Inf count: {inf_count}")
            # Save problematic batch for inspection
            try:
                import os
                debug_dir = os.path.join(config.output_dir, 'debug_batches')
                os.makedirs(debug_dir, exist_ok=True)
                torch.save(motion.cpu(), os.path.join(debug_dir, f'corrupted_batch_epoch{epoch}_batch{batch_idx}.pt'))
                print(f"  Saved corrupted batch to: {os.path.join(debug_dir, f'corrupted_batch_epoch{epoch}_batch{batch_idx}.pt')}")
            except Exception as e:
                print(f"  Could not save batch: {e}")
            # Skip this batch or raise error
            print("  ⚠️  Skipping this batch to prevent NaN propagation")
            continue
        
        # DEBUG: Check input data before forward pass
        if DEBUG_PRINTS and (batch_idx % 100 == 0 or batch_idx == 583):  # Check every 100 batches or at problematic batch
            print(f"\n[DEBUG] Batch {batch_idx} - Input Data Check:")
            print(f"  motion shape: {motion.shape}")
            print(f"  motion stats: min={motion.min().item():.4f}, max={motion.max().item():.4f}, mean={motion.mean().item():.4f}, std={motion.std().item():.4f}")
            print(f"  motion has NaN: {torch.isnan(motion).any().item()}, has Inf: {torch.isinf(motion).any().item()}")
            # Enable debug mode in encoder
            if hasattr(model, 'encoder'):
                model.encoder._debug_mode = True
        
        # Forward pass with mixed precision if enabled
        if use_amp and scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                # Handle VQ-VAE vs VAE/vanilla AE return values
                use_vqvae = getattr(config, 'use_vqvae', False)
                if use_vqvae:
                    recon_motion, z, commit_loss, perplexity, code_idx = model(motion)
                    mean = None
                    logvar = None
                else:
                    recon_motion, mean, logvar, z = model(motion)
                    commit_loss = None
                
                # Disable debug mode after forward
                if hasattr(model, 'encoder'):
                    model.encoder._debug_mode = False
                
                # DEBUG: Check model outputs (autocast path)
                if DEBUG_PRINTS and (batch_idx % 100 == 0 or batch_idx == 583):
                    print(f"\n[DEBUG] Batch {batch_idx} - Model Output Check (AMP):")
                    print(f"  recon_motion shape: {recon_motion.shape}")
                    print(f"  recon_motion stats: min={recon_motion.min().item():.4f}, max={recon_motion.max().item():.4f}, mean={recon_motion.mean().item():.4f}, std={recon_motion.std().item():.4f}")
                    print(f"  recon_motion has NaN: {torch.isnan(recon_motion).any().item()}, has Inf: {torch.isinf(recon_motion).any().item()}")
                    print(f"  z shape: {z.shape}, z stats: min={z.min().item():.4f}, max={z.max().item():.4f}, mean={z.mean().item():.4f}, std={z.std().item():.4f}")
                    print(f"  z has NaN: {torch.isnan(z).any().item()}, has Inf: {torch.isinf(z).any().item()}")
                
                # Check for NaN in model outputs - STOP TRAINING if detected
                if torch.isnan(recon_motion).any():
                    print(f"\n❌ ERROR: NaN detected in model outputs at epoch {epoch}, batch {batch_idx}")
                    print(f"  recon_motion NaN: {torch.isnan(recon_motion).sum().item()}")
                    print("🛑 Stopping training due to NaN in model outputs!")
                    raise RuntimeError("NaN detected in model outputs - training stopped")
                
                if not use_vqvae:
                    if torch.isnan(mean).any() or torch.isnan(logvar).any():
                        print(f"\n❌ ERROR: NaN detected in model outputs at epoch {epoch}, batch {batch_idx}")
                    print(f"  mean NaN: {torch.isnan(mean).sum().item()}")
                    print(f"  logvar NaN: {torch.isnan(logvar).sum().item()}")
                    print(f"  logvar range: [{logvar.min().item():.4f}, {logvar.max().item():.4f}]")
                    print("🛑 Stopping training due to NaN in model outputs!")
                    raise RuntimeError("NaN detected in model outputs - training stopped")
                
                # Compute loss within autocast context
                if mask is not None:
                    mask_ext = mask.unsqueeze(-1).float()
                    recon_motion = recon_motion * mask_ext
                    motion_target = motion * mask_ext
                else:
                    motion_target = motion
                loss, recon_loss, kl_loss, vel_loss = vae_loss(
                    recon_motion, motion_target,
                    recon_weight=config.recon_weight,
                    kl_weight=config.kl_weight,
                    use_vae=config.use_vae,
                    use_vqvae=use_vqvae,
                    mean=mean,
                    logvar=logvar,
                    commit_loss=commit_loss,
                    commit_weight=getattr(config, 'commit_weight', 0.02),
                    loss_vel_weight=getattr(config, 'loss_vel_weight', 0.0)
                )
                
                # DEBUG: Check loss values (autocast path)
                if DEBUG_PRINTS and (batch_idx % 100 == 0 or batch_idx == 583):
                    print(f"\n[DEBUG] Batch {batch_idx} - Loss Check (AMP):")
                    print(f"  total_loss: {loss.item():.6f}, has NaN: {torch.isnan(loss).item()}, has Inf: {torch.isinf(loss).item()}")
                    print(f"  recon_loss: {recon_loss.item():.6f}, has NaN: {torch.isnan(recon_loss).item()}, has Inf: {torch.isinf(recon_loss).item()}")
                    print(f"  kl_loss: {kl_loss.item():.6f}, has NaN: {torch.isnan(kl_loss).item()}, has Inf: {torch.isinf(kl_loss).item()}")
                    print(f"  vel_loss: {vel_loss.item():.6f}, has NaN: {torch.isnan(vel_loss).item()}, has Inf: {torch.isinf(vel_loss).item()}")
        else:
            # DEBUG: Check input data before forward pass
            if DEBUG_PRINTS and (batch_idx % 100 == 0 or batch_idx == 583):  # Check every 100 batches or at problematic batch
                print(f"\n[DEBUG] Batch {batch_idx} - Input Data Check:")
                print(f"  motion shape: {motion.shape}")
                print(f"  motion stats: min={motion.min().item():.4f}, max={motion.max().item():.4f}, mean={motion.mean().item():.4f}, std={motion.std().item():.4f}")
                print(f"  motion has NaN: {torch.isnan(motion).any().item()}, has Inf: {torch.isinf(motion).any().item()}")
                # Enable debug mode in encoder
                if hasattr(model, 'encoder'):
                    model.encoder._debug_mode = True
            
            # Handle VQ-VAE vs VAE/vanilla AE return values
            use_vqvae = getattr(config, 'use_vqvae', False)
            if use_vqvae:
                recon_motion, z, commit_loss, perplexity, code_idx = model(motion)
                mean = None
                logvar = None
            else:
                recon_motion, mean, logvar, z = model(motion)
                commit_loss = None
            
            # Disable debug mode after forward
            if hasattr(model, 'encoder'):
                model.encoder._debug_mode = False
            
            # DEBUG: Check model outputs and intermediate values
            if batch_idx % 100 == 0 or batch_idx == 583:
                print(f"\n[DEBUG] Batch {batch_idx} - Model Output Check:")
                print(f"  recon_motion shape: {recon_motion.shape}")
                print(f"  recon_motion stats: min={recon_motion.min().item():.4f}, max={recon_motion.max().item():.4f}, mean={recon_motion.mean().item():.4f}, std={recon_motion.std().item():.4f}")
                print(f"  recon_motion has NaN: {torch.isnan(recon_motion).any().item()}, has Inf: {torch.isinf(recon_motion).any().item()}")
                print(f"  z shape: {z.shape}, z stats: min={z.min().item():.4f}, max={z.max().item():.4f}, mean={z.mean().item():.4f}, std={z.std().item():.4f}")
                print(f"  z has NaN: {torch.isnan(z).any().item()}, has Inf: {torch.isinf(z).any().item()}")
            
            # Check for NaN in model outputs - STOP TRAINING if detected
            if torch.isnan(recon_motion).any():
                print(f"\n❌ ERROR: NaN detected in model outputs at epoch {epoch}, batch {batch_idx}")
                print(f"  recon_motion NaN: {torch.isnan(recon_motion).sum().item()}")
                print("🛑 Stopping training due to NaN in model outputs!")
                raise RuntimeError("NaN detected in model outputs - training stopped")
            
            if not use_vqvae:
                if torch.isnan(mean).any() or torch.isnan(logvar).any():
                    print(f"\n❌ ERROR: NaN detected in model outputs at epoch {epoch}, batch {batch_idx}")
                print(f"  mean NaN: {torch.isnan(mean).sum().item()}")
                print(f"  logvar NaN: {torch.isnan(logvar).sum().item()}")
                print(f"  logvar range: [{logvar.min().item():.4f}, {logvar.max().item():.4f}]")
                print("🛑 Stopping training due to NaN in model outputs!")
                raise RuntimeError("NaN detected in model outputs - training stopped")
            
            # Compute loss
            if mask is not None:
                mask_ext = mask.unsqueeze(-1).float()
                recon_motion = recon_motion * mask_ext
                motion_target = motion * mask_ext
            else:
                motion_target = motion
            loss, recon_loss, kl_loss, vel_loss = vae_loss(
                recon_motion, motion_target,
                recon_weight=config.recon_weight,
                kl_weight=config.kl_weight,
                use_vae=config.use_vae,
                use_vqvae=use_vqvae,
                mean=mean,
                logvar=logvar,
                commit_loss=commit_loss,
                commit_weight=getattr(config, 'commit_weight', 0.02),
                loss_vel_weight=getattr(config, 'loss_vel_weight', 0.0)
            )
        
        # Check for NaN in loss - STOP TRAINING if detected
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ ERROR: NaN/Inf loss at epoch {epoch}, batch {batch_idx}")
            print(f"  recon_loss: {recon_loss.item():.4f}, kl_loss: {kl_loss.item():.4f}, vel_loss: {vel_loss.item():.4f}")
            print(f"  motion stats: min={motion.min().item():.4f}, max={motion.max().item():.4f}, mean={motion.mean().item():.4f}")
            print(f"  recon_motion stats: min={recon_motion.min().item():.4f}, max={recon_motion.max().item():.4f}, mean={recon_motion.mean().item():.4f}")
            print(f"  z stats: min={z.min().item():.4f}, max={z.max().item():.4f}, mean={z.mean().item():.4f}")
            print("🛑 Stopping training due to NaN/Inf in loss!")
            raise RuntimeError("NaN/Inf detected in loss - training stopped")
        
        # Backward pass with gradient accumulation
        if use_amp and scaler is not None:
            scaler.scale(loss / accumulation_steps).backward()
        else:
            (loss / accumulation_steps).backward()
        
        # Update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            # DEBUG: Check gradients before clipping
            if DEBUG_PRINTS and (batch_idx % 100 == 0 or batch_idx == 583):
                print(f"\n[DEBUG] Batch {batch_idx} - Gradient Check (before clipping):")
                grad_stats = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_max = param.grad.abs().max().item()
                        has_nan = torch.isnan(param.grad).any().item()
                        has_inf = torch.isinf(param.grad).any().item()
                        # Always report encoder.fc_in gradients
                        if 'encoder.fc_in' in name or has_nan or has_inf or grad_norm > 10.0 or grad_max > 10.0:
                            grad_stats[name] = {
                                'norm': grad_norm,
                                'max': grad_max,
                                'has_nan': has_nan,
                                'has_inf': has_inf,
                                'param_min': param.min().item(),
                                'param_max': param.max().item(),
                                'param_mean': param.mean().item()
                            }
                if grad_stats:
                    print("  Gradient statistics:")
                    for name, stats in grad_stats.items():
                        print(f"    {name}:")
                        print(f"      grad_norm={stats['norm']:.4f}, grad_max={stats['max']:.4f}, NaN={stats['has_nan']}, Inf={stats['has_inf']}")
                        print(f"      param: min={stats['param_min']:.4f}, max={stats['param_max']:.4f}, mean={stats['param_mean']:.4f}")
                else:
                    print("  All gradients look normal")
            
            # Check for NaN gradients - STOP TRAINING if detected
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"\n" + "="*80)
                    print(f"❌ ERROR: NaN gradient detected in {name} at epoch {epoch}, batch {batch_idx}")
                    print(f"="*80)
                    
                    # Comprehensive debugging information
                    print(f"\n[1] INPUT DATA ANALYSIS:")
                    print(f"  motion shape: {motion.shape}")
                    print(f"  motion stats: min={motion.min().item():.4f}, max={motion.max().item():.4f}, mean={motion.mean().item():.4f}, std={motion.std().item():.4f}")
                    print(f"  motion has NaN: {torch.isnan(motion).any().item()}, has Inf: {torch.isinf(motion).any().item()}")
                    print(f"  motion value range check: abs_max={motion.abs().max().item():.4f}")
                    
                    # Check if any sample in batch is problematic
                    for i in range(motion.shape[0]):
                        sample = motion[i]
                        if torch.isnan(sample).any() or torch.isinf(sample).any() or (sample.abs() > 1e6).any():
                            print(f"  ⚠️  Sample {i} is problematic: NaN={torch.isnan(sample).any()}, Inf={torch.isinf(sample).any()}, extreme={(sample.abs() > 1e6).any()}")
                    
                    print(f"\n[2] MODEL OUTPUT ANALYSIS:")
                    print(f"  recon_motion shape: {recon_motion.shape}")
                    print(f"  recon_motion stats: min={recon_motion.min().item():.4f}, max={recon_motion.max().item():.4f}, mean={recon_motion.mean().item():.4f}, std={recon_motion.std().item():.4f}")
                    print(f"  recon_motion has NaN: {torch.isnan(recon_motion).any().item()}, has Inf: {torch.isinf(recon_motion).any().item()}")
                    print(f"  z shape: {z.shape}, z stats: min={z.min().item():.4f}, max={z.max().item():.4f}, mean={z.mean().item():.4f}, std={z.std().item():.4f}")
                    print(f"  z has NaN: {torch.isnan(z).any().item()}, has Inf: {torch.isinf(z).any().item()}")
                    
                    print(f"\n[3] LOSS ANALYSIS:")
                    print(f"  total_loss: {loss.item():.6f}, has NaN: {torch.isnan(loss).item()}, has Inf: {torch.isinf(loss).item()}")
                    print(f"  recon_loss: {recon_loss.item():.6f}, has NaN: {torch.isnan(recon_loss).item()}, has Inf: {torch.isinf(recon_loss).item()}")
                    print(f"  kl_loss: {kl_loss.item():.6f}, has NaN: {torch.isnan(kl_loss).item()}, has Inf: {torch.isinf(kl_loss).item()}")
                    print(f"  vel_loss: {vel_loss.item():.6f}, has NaN: {torch.isnan(vel_loss).item()}, has Inf: {torch.isinf(vel_loss).item()}")
                    if use_vqvae:
                        print(f"  commit_loss: {commit_loss.item():.6f if commit_loss is not None else 'N/A'}, perplexity: {perplexity:.4f}")
                    
                    print(f"\n[4] PARAMETER ANALYSIS ({name}):")
                    print(f"  Parameter shape: {param.shape}")
                    print(f"  Parameter stats: min={param.min().item():.4f}, max={param.max().item():.4f}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")
                    print(f"  Parameter has NaN: {torch.isnan(param).any().item()}, has Inf: {torch.isinf(param).any().item()}")
                    
                    print(f"\n[5] GRADIENT ANALYSIS ({name}):")
                    print(f"  Gradient stats: min={param.grad.min().item():.4f}, max={param.grad.max().item():.4f}, mean={param.grad.mean().item():.4f}, norm={param.grad.norm().item():.4f}")
                    print(f"  Gradient has NaN: {torch.isnan(param.grad).any().item()}, has Inf: {torch.isinf(param.grad).any().item()}")
                    nan_grad_count = torch.isnan(param.grad).sum().item()
                    if nan_grad_count > 0:
                        print(f"  NaN gradient count: {nan_grad_count} out of {param.grad.numel()}")
                        nan_indices = torch.where(torch.isnan(param.grad))
                        if len(nan_indices[0]) <= 20:
                            print(f"  NaN gradient positions: {list(zip(*[idx.cpu().numpy() for idx in nan_indices]))}")
                    
                    # Check encoder intermediate values if this is encoder.fc_in
                    if 'encoder.fc_in' in name and hasattr(model, 'encoder'):
                        print(f"\n[6] ENCODER INTERMEDIATE VALUES:")
                        # Try to get the last forward pass values
                        if hasattr(model.encoder, '_last_input'):
                            print(f"  Last encoder input stats: min={model.encoder._last_input.min().item():.4f}, max={model.encoder._last_input.max().item():.4f}")
                        if hasattr(model.encoder, '_last_fc_in_output'):
                            print(f"  Last fc_in output stats: min={model.encoder._last_fc_in_output.min().item():.4f}, max={model.encoder._last_fc_in_output.max().item():.4f}")
                    
                    # Save problematic batch and model state
                    try:
                        import os
                        debug_dir = os.path.join(config.output_dir, 'debug_batches')
                        os.makedirs(debug_dir, exist_ok=True)
                        torch.save({
                            'motion': motion.cpu(),
                            'recon_motion': recon_motion.cpu(),
                            'z': z.cpu(),
                            'loss': loss.item(),
                            'recon_loss': recon_loss.item(),
                            'kl_loss': kl_loss.item(),
                            'vel_loss': vel_loss.item(),
                            'epoch': epoch,
                            'batch_idx': batch_idx,
                            'param_name': name,
                            'param_value': param.cpu(),
                            'grad_value': param.grad.cpu() if param.grad is not None else None
                        }, os.path.join(debug_dir, f'nan_gradient_epoch{epoch}_batch{batch_idx}.pt'))
                        print(f"\n[7] SAVED DEBUG INFO:")
                        print(f"  Saved to: {os.path.join(debug_dir, f'nan_gradient_epoch{epoch}_batch{batch_idx}.pt')}")
                    except Exception as e:
                        print(f"\n[7] Could not save debug info: {e}")
                    
                    print(f"\n" + "="*80)
                    print("🛑 Stopping training due to NaN gradients!")
                    print(f"="*80)
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
        if getattr(config, 'loss_vel_weight', 0.0) > 0.0:
            total_vel_loss += vel_loss.item()
        if use_vqvae:
            # Perplexity is a tensor, convert to float
            if isinstance(perplexity, torch.Tensor):
                total_perplexity += perplexity.item()
            else:
                total_perplexity += perplexity
        num_batches += 1
        
        # Update progress bar
        loss_name = 'commit' if getattr(config, 'use_vqvae', False) else 'kl'
        postfix_dict = {
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            loss_name: f'{kl_loss.item():.6f}'
        }
        if getattr(config, 'loss_vel_weight', 0.0) > 0.0:
            vel_loss_val = vel_loss.item() if isinstance(vel_loss, torch.Tensor) else vel_loss
            postfix_dict['vel'] = f'{vel_loss_val:.6f}'
        pbar.set_postfix(postfix_dict)
        
        # Log to tensorboard
        if writer is not None and batch_idx % config.log_every == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/ReconLoss', recon_loss.item(), global_step)
            if getattr(config, 'loss_vel_weight', 0.0) > 0.0:
                writer.add_scalar('Train/VelLoss', vel_loss.item(), global_step)
            if use_vqvae:
                writer.add_scalar('Train/CommitLoss', kl_loss.item(), global_step)
                # Log perplexity for VQ-VAE
                if isinstance(perplexity, torch.Tensor):
                    writer.add_scalar('Train/Perplexity', perplexity.item(), global_step)
                else:
                    writer.add_scalar('Train/Perplexity', perplexity, global_step)
            else:
                writer.add_scalar('Train/KLLoss', kl_loss.item(), global_step)
    
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_vel_loss = total_vel_loss / num_batches if getattr(config, 'loss_vel_weight', 0.0) > 0.0 else 0.0
    use_vqvae = getattr(config, 'use_vqvae', False)
    avg_perplexity = total_perplexity / num_batches if use_vqvae else 0.0
    
    return avg_loss, avg_recon_loss, avg_kl_loss, avg_perplexity, avg_vel_loss


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
            use_vqvae = getattr(config, 'use_vqvae', False)
            if use_vqvae:
                recon_motion, z, commit_loss, perplexity, code_idx = model(motion)
                mean = None
                logvar = None
            else:
                recon_motion, mean, logvar, z = model(motion)
                commit_loss = None
            
            # Compute loss
            loss, recon_loss, kl_loss, vel_loss = vae_loss(
                recon_motion, motion,
                recon_weight=config.recon_weight,
                kl_weight=config.kl_weight,
                use_vae=config.use_vae,
                use_vqvae=use_vqvae,
                mean=mean,
                logvar=logvar,
                commit_loss=commit_loss,
                commit_weight=getattr(config, 'commit_weight', 0.02),
                loss_vel_weight=getattr(config, 'loss_vel_weight', 0.0)
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
        f.write(f"  Use VQ-VAE: {getattr(config, 'use_vqvae', False)}\n")
        if getattr(config, 'use_vqvae', False):
            f.write(f"  Number of Codes: {getattr(config, 'nb_code', 512)}\n")
            f.write(f"  Quantizer Type: {getattr(config, 'quantizer', 'ema_reset')}\n")
            f.write(f"  VQ Mu (EMA decay): {getattr(config, 'vq_mu', 0.99)}\n")
            f.write(f"  VQ Beta: {getattr(config, 'vq_beta', 1.0)}\n")
            f.write(f"  Commitment Weight: {getattr(config, 'commit_weight', 0.02)}\n")
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
        if getattr(config, 'use_vqvae', False):
            f.write(f"  Commitment Weight: {getattr(config, 'commit_weight', 0.02)}\n")
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
            'use_vqvae': getattr(config, 'use_vqvae', False),
            'nb_code': getattr(config, 'nb_code', 512),
            'quantizer': getattr(config, 'quantizer', 'ema_reset'),
            'vq_mu': getattr(config, 'vq_mu', 0.99),
            'vq_beta': getattr(config, 'vq_beta', 1.0),
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
    
    # Set input_dim and seq_len based on representation_type
    representation_type = getattr(config, 'representation_type', 'humanml3d')
    if representation_type == 'humanml3d':
        config.input_dim = 263
        config.seq_len = 20  # window_size
        print(f"Using HumanML3D representation: input_dim={config.input_dim}, seq_len={config.seq_len}")
    elif representation_type == 'interhuman':
        config.input_dim = 262
        config.seq_len = 19  # window_size - 1 (process_motion_interhuman reduces by 1)
        print(f"Using InterHuman representation: input_dim={config.input_dim}, seq_len={config.seq_len}")
    elif representation_type == 'relationship':
        config.input_dim = 4  # [w, z, x, z] - quaternion components [w, z] + position [x, z]
        config.seq_len = 19  # window_size - 1 (extracted from InterHuman)
        print(f"Using relationship features: input_dim={config.input_dim}, seq_len={config.seq_len}")
    else:
        raise ValueError(f"Unknown representation_type: {representation_type}")
    
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
        normalize=True,  # Use normalized data (default, recommended for training)
        representation_type=representation_type,
    )
    
    print(f"Dataset size: {len(train_loader.dataset)} samples")
    
    # Save configuration to text file
    save_config_txt(config, config.checkpoint_dir)
    
    # Create model
    print("Creating model...")
    print(f"Architecture: Encoder={config.encoder_type}, Decoder={config.decoder_type}")
    use_vqvae = getattr(config, 'use_vqvae', False)
    if use_vqvae:
        quantizer_type = getattr(config, 'quantizer', 'ema_reset')
        print(f"Mode: VQ-VAE (nb_code={getattr(config, 'nb_code', 512)}, quantizer={quantizer_type})")
    elif config.use_vae:
        print(f"Mode: VAE")
    else:
        print(f"Mode: Vanilla Autoencoder")
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
        use_vqvae=use_vqvae,
        nb_code=getattr(config, 'nb_code', 512),
        quantizer=getattr(config, 'quantizer', 'ema_reset'),
        vq_mu=getattr(config, 'vq_mu', 0.99),
        vq_beta=getattr(config, 'vq_beta', 1.0),
        downsampling_factor=getattr(config, 'downsampling_factor', 4)
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    # Use lower learning rate for transformer to prevent instability
    base_lr = config.learning_rate
    if config.encoder_type == 'transformer' or config.decoder_type == 'transformer':
        base_lr = base_lr * 0.5  # Reduce LR for transformer
        print(f"Using reduced learning rate {base_lr} for transformer architecture")
    
    # Use AdamW with T2M-GPT betas for VQ-VAE, Adam for others
    use_vqvae = getattr(config, 'use_vqvae', False)
    if use_vqvae:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.99),  # T2M-GPT betas
            eps=1e-8
        )
    else:
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
    elif config.lr_scheduler == 'multistep':
        # MultiStepLR scheduler (matches T2M-GPT style)
        milestones = getattr(config, 'lr_scheduler_milestones', None)
        if milestones is None:
            # Default: use epoch-based milestones (approximate conversion from T2M-GPT's iteration-based)
            # T2M-GPT uses [50000, 200000] iterations, approximate to epochs based on typical dataset
            milestones = [50, 200]  # Adjust based on your dataset size
        gamma = getattr(config, 'lr_scheduler_gamma', 0.05)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
        print(f"Using MultiStepLR scheduler with milestones: {milestones}, gamma: {gamma}")
    
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
    
    # Warm-up phase (for VQ-VAE, matches T2M-GPT)
    # Calculate warm-up iterations: use warm_up_epochs if set, otherwise use warm_up_iter
    warm_up_epochs = getattr(config, 'warm_up_epochs', 0)
    warm_up_iter = getattr(config, 'warm_up_iter', None)
    
    if warm_up_epochs > 0:
        # Convert epochs to iterations
        batches_per_epoch = len(train_loader)
        warm_up_iter = warm_up_epochs * batches_per_epoch
        print(f"Warm-up: {warm_up_epochs} epochs = {warm_up_iter} iterations (batches_per_epoch={batches_per_epoch})")
    elif warm_up_iter is None:
        warm_up_iter = 0
    
    if warm_up_iter > 0 and start_epoch == 0:  # Only warm-up if starting fresh
        print(f"\n{'='*60}")
        if warm_up_epochs > 0:
            print(f"Warm-up Phase: {warm_up_epochs} epochs ({warm_up_iter} iterations)")
        else:
            print(f"Warm-up Phase: {warm_up_iter} iterations")
        print(f"{'='*60}")
        model.train()
        avg_recons, avg_perplexity, avg_commit, avg_vel = 0., 0., 0., 0.
        
        # Create cycling iterator for warm-up
        train_loader_iter = iter(train_loader)
        
        # T2M-GPT uses range(1, warm_up_iter) - so warm_up_iter-1 iterations total
        for nb_iter in range(1, warm_up_iter):
            # Update learning rate with warm-up (matches T2M-GPT formula exactly)
            optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, warm_up_iter, base_lr)
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)

            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                motion, mask = batch
                mask = mask.to(device)
            else:
                motion = batch
                mask = None
            
            motion = motion.to(device)
            
            # Forward pass
            use_vqvae = getattr(config, 'use_vqvae', False)
            if use_vqvae:
                recon_motion, z, commit_loss, perplexity, code_idx = model(motion)
                mean = None
                logvar = None
            else:
                recon_motion, mean, logvar, z = model(motion)
                commit_loss = None
                perplexity = 0.0
            
            # Compute loss
            if mask is not None:
                mask_ext = mask.unsqueeze(-1).float()
                recon_motion = recon_motion * mask_ext
                motion_target = motion * mask_ext
            else:
                motion_target = motion
            loss, recon_loss, kl_loss, vel_loss = vae_loss(
                recon_motion, motion_target,
                recon_weight=config.recon_weight,
                kl_weight=config.kl_weight,
                use_vae=config.use_vae,
                use_vqvae=use_vqvae,
                mean=mean,
                logvar=logvar,
                commit_loss=commit_loss,
                commit_weight=getattr(config, 'commit_weight', 0.02),
                loss_vel_weight=getattr(config, 'loss_vel_weight', 0.0)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            avg_recons += recon_loss.item()
            avg_commit += kl_loss.item() if use_vqvae else 0.0
            avg_vel += vel_loss.item() if getattr(config, 'loss_vel_weight', 0.0) > 0.0 else 0.0
            if use_vqvae:
                if isinstance(perplexity, torch.Tensor):
                    avg_perplexity += perplexity.item()
                else:
                    avg_perplexity += perplexity
            
            # Print progress
            if nb_iter % getattr(config, 'log_every', 100) == 0:
                avg_recons /= config.log_every
                avg_perplexity /= config.log_every if use_vqvae else 1.0
                avg_commit /= config.log_every if use_vqvae else 1.0
                avg_vel /= config.log_every if getattr(config, 'loss_vel_weight', 0.0) > 0.0 else 1.0
                
                if use_vqvae:
                    print(f"Warmup. Iter {nb_iter}/{warm_up_iter}: lr {current_lr:.5f} | Commit: {avg_commit:.5f} | PPL: {avg_perplexity:.2f} | Recons: {avg_recons:.5f}", end="")
                    if getattr(config, 'loss_vel_weight', 0.0) > 0.0:
                        print(f" | Vel: {avg_vel:.5f}")
                    else:
                        print()
                else:
                    print(f"Warmup. Iter {nb_iter}/{warm_up_iter}: lr {current_lr:.5f} | Recons: {avg_recons:.5f}")
                
                # Log to tensorboard
                if writer is not None:
                    writer.add_scalar('Warmup/LearningRate', current_lr, nb_iter)
                    writer.add_scalar('Warmup/ReconLoss', avg_recons, nb_iter)
                    if use_vqvae:
                        writer.add_scalar('Warmup/CommitLoss', avg_commit, nb_iter)
                        writer.add_scalar('Warmup/Perplexity', avg_perplexity, nb_iter)
                    if getattr(config, 'loss_vel_weight', 0.0) > 0.0:
                        writer.add_scalar('Warmup/VelLoss', avg_vel, nb_iter)
                
                avg_recons, avg_perplexity, avg_commit, avg_vel = 0., 0., 0., 0.
        
        print(f"Warm-up completed! Starting main training with LR={base_lr:.6f}\n")
    
    # Training loop
    print("Starting training...")
    try:
        for epoch in range(start_epoch, end_epoch + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{config.num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_recon, train_kl, train_perplexity, train_vel = train_epoch(
                model, train_loader, optimizer, device, epoch, config, writer, scaler
            )
            
            # Print training metrics
            use_vqvae = getattr(config, 'use_vqvae', False)
            if use_vqvae:
                vel_str = f" | Vel: {train_vel:.6f}" if getattr(config, 'loss_vel_weight', 0.0) > 0.0 else ""
                print(f"Train Loss: {train_loss:.4f} | Recon: {train_recon:.4f} | Commit: {train_kl:.6f} | Perplexity: {train_perplexity:.2f}{vel_str}")
            else:
                vel_str = f" | Vel: {train_vel:.6f}" if getattr(config, 'loss_vel_weight', 0.0) > 0.0 else ""
                print(f"Train Loss: {train_loss:.4f} | Recon: {train_recon:.4f} | KL: {train_kl:.6f}{vel_str}")
            
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
            if getattr(config, 'loss_vel_weight', 0.0) > 0.0:
                writer.add_scalar('Epoch/TrainVelLoss', train_vel, epoch)
            if use_vqvae:
                writer.add_scalar('Epoch/TrainCommitLoss', train_kl, epoch)
                writer.add_scalar('Epoch/TrainPerplexity', train_perplexity, epoch)
            else:
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
