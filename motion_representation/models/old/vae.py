"""
Unified VAE model supporting both GRU and Transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae_gru import GRUEncoder, GRUDecoder
from .vae_transformer import TransformerEncoder, TransformerDecoder


class MotionVAE(nn.Module):
    """
    Unified VAE for motion representation learning.
    Supports both GRU and Transformer architectures.
    """
    
    def __init__(self, input_dim=263, hidden_dim=512, num_layers=2, latent_dim=2048, 
                 seq_len=20, dropout=0.1, encoder_type='gru', decoder_type='gru',
                 # Transformer-specific parameters
                 num_heads=8, ff_size=2048, activation='gelu', use_vae=False):
        """
        Args:
            input_dim: Input feature dimension (263 for HumanML3D)
            hidden_dim: Hidden dimension (512 for GRU/Transformer)
            num_layers: Number of layers (GRU layers or Transformer layers)
            latent_dim: Dimension of latent representation (2048)
            seq_len: Sequence length (20 frames)
            dropout: Dropout rate
            encoder_type: 'gru' or 'transformer'
            decoder_type: 'gru' or 'transformer'
            num_heads: Number of attention heads (for transformer, default: 8)
            ff_size: Feed-forward size (for transformer, default: 2048)
            activation: Activation function for transformer ('gelu', 'relu', default: 'gelu')
            use_vae: If True, use VAE with reparameterization; if False, use vanilla autoencoder
        """
        super(MotionVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.use_vae = use_vae
        
        # Create encoder
        if encoder_type == 'gru':
            self.encoder = GRUEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                latent_dim=latent_dim,
                dropout=dropout
            )
        elif encoder_type == 'transformer':
            self.encoder = TransformerEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_size=ff_size,
                latent_dim=latent_dim,
                seq_len=seq_len,
                dropout=dropout,
                activation=activation
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'gru' or 'transformer'")
        
        # Create decoder
        if decoder_type == 'gru':
            self.decoder = GRUDecoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=input_dim,
                seq_len=seq_len,
                dropout=dropout
            )
        elif decoder_type == 'transformer':
            self.decoder = TransformerDecoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_size=ff_size,
                output_dim=input_dim,
                seq_len=seq_len,
                dropout=dropout,
                activation=activation
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}. Must be 'gru' or 'transformer'")
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick for VAE with numerical stability."""
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode(self, x):
        """Encode input to latent representation."""
        mean, logvar = self.encoder(x)
        if self.use_vae:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean  # Vanilla autoencoder: use mean directly
        return z, mean, logvar
    
    def decode(self, z, first_frame=None):
        """Decode latent representation to motion."""
        return self.decoder(z, first_frame)
    
    def forward(self, x):
        """
        Forward pass through VAE or vanilla autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len=20, input_dim=263)
        
        Returns:
            recon_x: Reconstructed motion (batch_size, seq_len, input_dim)
            mean: Mean of latent distribution (or encoder output for vanilla AE)
            logvar: Log variance of latent distribution (or zeros for vanilla AE)
            z: Latent representation (sampled for VAE, mean for vanilla AE)
        """
        mean, logvar = self.encoder(x)
        if self.use_vae:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean  # Vanilla autoencoder: use mean directly
        
        # Use first frame of input for decoder
        first_frame = x[:, 0, :]  # (batch, input_dim)
        recon_x = self.decoder(z, first_frame)
        
        return recon_x, mean, logvar, z
    
    def sample(self, batch_size, device, first_frame=None):
        """Sample from prior distribution."""
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decoder(z, first_frame)


def vae_loss(recon_x, x, mean, logvar, recon_weight=1.0, kl_weight=0.0001, use_vae=True):
    """
    VAE loss: Reconstruction loss + KL divergence (if use_vae=True).
    Vanilla autoencoder loss: Only reconstruction loss (if use_vae=False).
    
    Args:
        recon_x: Reconstructed motion (batch, seq_len, dim)
        x: Original motion (batch, seq_len, dim)
        mean: Mean of latent distribution
        logvar: Log variance of latent distribution
        recon_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence loss
        use_vae: If True, include KL loss; if False, only reconstruction loss
    
    Returns:
        total_loss: Total loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence loss (0.0 if use_vae=False)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # Check for NaN in reconstruction
    if torch.isnan(recon_loss):
        print("Warning: NaN in reconstruction loss!")
        recon_loss = torch.tensor(0.0, device=recon_loss.device)
    
    # KL divergence loss (only for VAE)
    if use_vae:
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # KL divergence loss with numerical stability
        # Use more stable formula: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        # Clamp exp(logvar) to prevent overflow
        var = torch.clamp(torch.exp(logvar), min=1e-8, max=1e8)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - var, dim=1)
        kl_loss = torch.mean(kl_loss)
        
        # Check for NaN in KL loss
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            print("Warning: NaN/Inf in KL loss! Clamping...")
            kl_loss = torch.clamp(kl_loss, min=-1e6, max=1e6)
            if torch.isnan(kl_loss):
                kl_loss = torch.tensor(0.0, device=kl_loss.device)
    else:
        # Vanilla autoencoder: no KL loss
        kl_loss = torch.tensor(0.0, device=recon_loss.device)
    
    # Total loss
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss
    
    # Final NaN check
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"Warning: NaN/Inf in total loss! recon={recon_loss:.4f}, kl={kl_loss:.4f}")
        total_loss = torch.clamp(total_loss, min=-1e6, max=1e6)
        if torch.isnan(total_loss):
            total_loss = recon_weight * recon_loss  # Fallback to reconstruction only
    
    return total_loss, recon_loss, kl_loss

