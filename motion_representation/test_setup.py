"""
Quick test script to verify the setup works correctly.
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from motion_representation.models import MotionModel, vae_loss
# Backward compatibility
MotionVAE = MotionModel
from motion_representation.config import get_config


def test_model():
    """Test that the model can be created and run forward pass."""
    print("Testing Motion VAE model...")
    
    # Create a simple config
    class TestConfig:
        input_dim = 263
        hidden_dim = 512
        num_layers = 2
        latent_dim = 2048
        seq_len = 20
        dropout = 0.1
        recon_weight = 1.0
        kl_weight = 0.0001
    
    config = TestConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = MotionModel(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        latent_dim=config.latent_dim,
        seq_len=config.seq_len,
        dropout=config.dropout,
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, config.seq_len, config.input_dim).to(device)
    
    print(f"Input shape: {x.shape}")
    
    recon_x, mean, logvar, z = model(x)
    
    print(f"Reconstructed shape: {recon_x.shape}")
    print(f"Latent mean shape: {mean.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"Latent z shape: {z.shape}")
    
    # Test loss
    loss, recon_loss, kl_loss = vae_loss(
        recon_x, x, mean, logvar,
        recon_weight=config.recon_weight,
        kl_weight=config.kl_weight
    )
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {recon_loss.item():.4f}")
    print(f"KL Loss: {kl_loss.item():.6f}")
    
    # Test sampling
    sampled = model.sample(batch_size=2, device=device)
    print(f"\nSampled motion shape: {sampled.shape}")
    
    print("\n✓ Model test passed!")


if __name__ == '__main__':
    test_model()

