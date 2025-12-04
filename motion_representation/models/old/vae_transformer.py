"""
Transformer-based VAE encoder and decoder for motion sequences.
Uses query token approach for encoder and autoregressive decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    """Transformer-based encoder for motion sequences using query token approach.
    Operates directly in latent space for simplicity and fewer parameters."""
    
    def __init__(self, input_dim=263, hidden_dim=512, num_layers=8, num_heads=8, 
                 ff_size=2048, latent_dim=2048, seq_len=20, dropout=0.1, activation='gelu'):
        """
        Args:
            input_dim: Input feature dimension (263 for HumanML3D)
            hidden_dim: Not used (kept for compatibility), transformer operates in latent_dim
            num_layers: Number of transformer encoder layers (8)
            num_heads: Number of attention heads (8)
            ff_size: Feed-forward network size (2048)
            latent_dim: Dimension of latent representation (also transformer dimension)
            seq_len: Sequence length (20 frames)
            dropout: Dropout rate
            activation: Activation function ('gelu', 'relu', etc.)
        """
        super(TransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        
        # Input embedding: directly to latent dimension (simpler, fewer parameters)
        self.embed_motion = nn.Linear(input_dim, latent_dim)
        
        # Query token (learnable parameter, like CLS token in BERT)
        # Operates directly in latent space
        # Initialize with very small values to prevent gradient explosion
        self.query_token = nn.Parameter(torch.zeros(1, latent_dim))
        # Small random initialization
        nn.init.normal_(self.query_token, mean=0.0, std=0.01)
        
        # Positional encoding in latent space
        self.pos_encoder = PositionalEncoding(latent_dim, dropout, max_len=seq_len + 1)
        
        # Transformer encoder layers operating directly in latent space
        # Use norm_first=True (pre-norm) for better training stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,  # Operate directly in latent space
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm for better stability (prevents gradient explosion)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm before output
        self.out_ln = nn.LayerNorm(latent_dim)
        
        # Projection to latent space (mean and log-variance)
        # Since we're already in latent_dim, these are just identity or small refinements
        # Mean: can use query output directly or small projection
        self.fc_mean = nn.Linear(latent_dim, latent_dim)
        # Logvar: small projection to ensure stable initialization
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        
        # Initialize with identity-like behavior (small changes)
        nn.init.xavier_uniform_(self.fc_mean.weight, gain=0.1)
        nn.init.eye_(self.fc_mean.weight)  # Start close to identity
        nn.init.constant_(self.fc_mean.bias, 0.0)
        
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.01)
        nn.init.constant_(self.fc_logvar.bias, -2.0)  # Start with small variance
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len=20, input_dim=263)
        
        Returns:
            mean: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        batch_size = x.size(0)
        
        # Embed motion frames: (batch, 20, 263) -> (batch, 20, hidden_dim)
        x_emb = self.embed_motion(x)
        
        # Get query token for this batch: (1, hidden_dim) -> (batch, 1, hidden_dim)
        # Clamp query token to prevent extreme values
        query_token_clamped = torch.clamp(self.query_token, min=-10, max=10)
        query = query_token_clamped.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate query token with motion frames: (batch, 21, hidden_dim)
        # [query_token, frame_1, frame_2, ..., frame_20]
        emb = torch.cat([query, x_emb], dim=1)
        
        # Add positional encoding
        h = self.pos_encoder(emb)
        
        # Transformer encoder (no masking needed for fixed-length sequences)
        h = self.transformer(h)
        
        # Layer norm
        h = self.out_ln(h)
        
        # Extract query token output (first position): (batch, latent_dim)
        query_output = h[:, 0]
        
        # Project to mean and logvar (small refinements since we're already in latent_dim)
        mean = self.fc_mean(query_output)  # (batch, latent_dim)
        logvar = self.fc_logvar(query_output)  # (batch, latent_dim)
        
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mean, logvar


class TransformerDecoder(nn.Module):
    """Autoregressive transformer-based decoder for motion sequences.
    Operates directly in latent space for consistency with encoder."""
    
    def __init__(self, latent_dim=2048, hidden_dim=512, num_layers=8, num_heads=8,
                 ff_size=2048, output_dim=263, seq_len=20, dropout=0.1, activation='gelu'):
        """
        Args:
            latent_dim: Dimension of latent representation (also transformer dimension)
            hidden_dim: Not used (kept for compatibility), decoder operates in latent_dim
            num_layers: Number of transformer decoder layers (8)
            num_heads: Number of attention heads (8)
            ff_size: Feed-forward network size (2048)
            output_dim: Output feature dimension (263 for HumanML3D)
            seq_len: Sequence length (20 frames)
            dropout: Dropout rate
            activation: Activation function ('gelu', 'relu', etc.)
        """
        super(TransformerDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Embed first frame directly to latent dimension
        self.embed_first_frame = nn.Sequential(
            nn.Linear(output_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Embed generated frames (for autoregressive generation) to latent dimension
        self.embed_frame = nn.Linear(output_dim, latent_dim)
        
        # Positional encoding in latent space
        self.pos_encoder = PositionalEncoding(latent_dim, dropout, max_len=seq_len)
        
        # Transformer decoder layers operating directly in latent space
        # Use norm_first=True (pre-norm) for better training stability
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,  # Operate directly in latent space
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm for better stability (prevents gradient explosion)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection from latent space to output dimension
        self.fc_out = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),  # Intermediate projection
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, output_dim)  # Final output
        )
        
    def forward(self, z, first_frame=None):
        """
        Autoregressive decoding from latent representation.
        Uses causal masking to generate all frames in parallel during training.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim=2048)
            first_frame: First frame of original motion (batch_size, output_dim=263)
                        If None, uses zeros
        
        Returns:
            output: Reconstructed motion (batch_size, seq_len=20, output_dim=263)
        """
        batch_size = z.size(0)
        device = z.device
        
        # Use first frame as initial input, or zeros if not provided
        if first_frame is None:
            first_frame = torch.zeros(batch_size, self.output_dim, device=device)
        
        # Embed first frame directly to latent space: (batch, output_dim) -> (batch, latent_dim)
        first_frame_emb = self.embed_first_frame(first_frame)  # (batch, latent_dim)
        
        # Initialize sequence with first frame
        # We'll build the sequence autoregressively
        outputs = []
        current_seq = first_frame_emb.unsqueeze(1)  # (batch, 1, latent_dim)
        
        # Use latent z directly as memory (no projection needed)
        memory = z.unsqueeze(1).expand(-1, self.seq_len, -1)  # (batch, seq_len, latent_dim)
        
        for t in range(self.seq_len):
            # Add positional encoding
            seq_with_pos = self.pos_encoder(current_seq)  # (batch, t+1, latent_dim)
            
            # Create causal mask: prevent attending to future positions
            tgt_len = seq_with_pos.size(1)
            causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
                diagonal=1
            )
            
            # Transformer decoder with cross-attention to latent
            decoder_output = self.transformer_decoder(
                tgt=seq_with_pos,
                memory=memory[:, :tgt_len],  # Latent as memory
                tgt_mask=causal_mask
            )  # (batch, t+1, latent_dim)
            
            # Get output for the last (current) position
            current_output = decoder_output[:, -1:]  # (batch, 1, latent_dim)
            
            # Project from latent space to output dimension
            frame_output = self.fc_out(current_output)  # (batch, 1, output_dim)
            outputs.append(frame_output)
            
            # Embed generated frame for next iteration
            if t < self.seq_len - 1:  # Don't need to embed for last frame
                next_frame_emb = self.embed_frame(frame_output)  # (batch, 1, latent_dim)
                current_seq = torch.cat([current_seq, next_frame_emb], dim=1)
        
        # Concatenate all outputs
        output = torch.cat(outputs, dim=1)  # (batch, seq_len, output_dim)
        
        return output

