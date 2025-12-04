"""
Evaluation and visualization utilities for Motion VAE.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from motion_representation.models import MotionModel, vae_loss
# Backward compatibility
MotionVAE = MotionModel


def evaluate_model(model, dataloader, device, config):
    """
    Evaluate the model on a dataset.
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    
    all_reconstructions = []
    all_originals = []
    all_latents = []
    
    with torch.no_grad():
        for motion in tqdm(dataloader, desc="Evaluating"):
            motion = motion.to(device)
            
            # Forward pass
            recon_motion, mean, logvar, z = model(motion)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(
                recon_motion, motion, mean, logvar,
                recon_weight=config.recon_weight,
                kl_weight=config.kl_weight
            )
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
            
            # Store samples for visualization
            if len(all_reconstructions) < 10:  # Store first 10 batches
                all_reconstructions.append(recon_motion.cpu().numpy())
                all_originals.append(motion.cpu().numpy())
                all_latents.append(z.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    
    metrics = {
        'loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'kl_loss': avg_kl_loss,
        'reconstructions': np.concatenate(all_reconstructions, axis=0) if all_reconstructions else None,
        'originals': np.concatenate(all_originals, axis=0) if all_originals else None,
        'latents': np.concatenate(all_latents, axis=0) if all_latents else None,
    }
    
    return metrics


def visualize_reconstruction(original, reconstructed, save_path=None, num_samples=5):
    """
    Visualize original vs reconstructed motions.
    
    Args:
        original: Original motion array (batch, seq_len, dim)
        reconstructed: Reconstructed motion array (batch, seq_len, dim)
        save_path: Path to save the visualization
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, original.shape[0])
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Select a subset of dimensions to visualize (e.g., first 10)
    dims_to_plot = min(10, original.shape[2])
    
    for i in range(num_samples):
        # Original
        ax = axes[i, 0]
        for dim in range(dims_to_plot):
            ax.plot(original[i, :, dim], label=f'Dim {dim}', alpha=0.7)
        ax.set_title(f'Sample {i+1} - Original')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Reconstructed
        ax = axes[i, 1]
        for dim in range(dims_to_plot):
            ax.plot(reconstructed[i, :, dim], label=f'Dim {dim}', alpha=0.7)
        ax.set_title(f'Sample {i+1} - Reconstructed')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_latent_space(latents, save_path=None):
    """
    Visualize latent space using PCA or t-SNE.
    
    Args:
        latents: Latent representations (batch, latent_dim)
        save_path: Path to save the visualization
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.5, s=10)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Latent Space Visualization (PCA)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Latent space visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("sklearn not available. Skipping latent space visualization.")


def compute_reconstruction_metrics(original, reconstructed):
    """
    Compute detailed reconstruction metrics.
    
    Args:
        original: Original motion (batch, seq_len, dim)
        reconstructed: Reconstructed motion (batch, seq_len, dim)
    
    Returns:
        dict: Dictionary of metrics
    """
    original = torch.from_numpy(original) if isinstance(original, np.ndarray) else original
    reconstructed = torch.from_numpy(reconstructed) if isinstance(reconstructed, np.ndarray) else reconstructed
    
    # MSE per frame
    mse_per_frame = torch.mean((original - reconstructed) ** 2, dim=(0, 2))  # (seq_len,)
    
    # MSE per dimension
    mse_per_dim = torch.mean((original - reconstructed) ** 2, dim=(0, 1))  # (dim,)
    
    # Overall MSE
    mse = torch.mean((original - reconstructed) ** 2).item()
    
    # MAE
    mae = torch.mean(torch.abs(original - reconstructed)).item()
    
    # Per-sample MSE
    mse_per_sample = torch.mean((original - reconstructed) ** 2, dim=(1, 2))  # (batch,)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'mse_per_frame': mse_per_frame.numpy() if isinstance(mse_per_frame, torch.Tensor) else mse_per_frame,
        'mse_per_dim': mse_per_dim.numpy() if isinstance(mse_per_dim, torch.Tensor) else mse_per_dim,
        'mse_per_sample': mse_per_sample.numpy() if isinstance(mse_per_sample, torch.Tensor) else mse_per_sample,
    }
    
    return metrics


def sample_and_visualize(model, device, config, num_samples=5, save_dir=None):
    """
    Sample from the model and visualize generated motions.
    
    Args:
        model: Trained VAE model
        device: Device to run on
        config: Configuration object
        num_samples: Number of samples to generate
        save_dir: Directory to save visualizations
    """
    model.eval()
    
    with torch.no_grad():
        # Sample from prior (no first frame provided, will use zeros)
        z = torch.randn(num_samples, config.latent_dim, device=device)
        generated = model.decode(z, first_frame=None)  # (batch, seq_len, dim)
        generated = generated.cpu().numpy()
    
    # Visualize generated samples
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    dims_to_plot = min(10, generated.shape[2])
    
    for i in range(num_samples):
        ax = axes[i]
        for dim in range(dims_to_plot):
            ax.plot(generated[i, :, dim], label=f'Dim {dim}', alpha=0.7)
        ax.set_title(f'Generated Sample {i+1}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'generated_samples.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Generated samples visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return generated

