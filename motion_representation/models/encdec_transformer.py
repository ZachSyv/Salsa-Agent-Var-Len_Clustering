"""
Transformer-based encoder and decoder for motion sequences.
Clean implementation without VAE-specific components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    """Transformer-based encoder for motion sequences using query token approach."""
    
    def __init__(self, input_dim=263, hidden_dim=512, num_layers=8, num_heads=8, 
                 ff_size=2048, output_dim=512, seq_len=20, dropout=0.1, activation='gelu'):
        """
        Args:
            input_dim: Input feature dimension (263 for HumanML3D)
            hidden_dim: Not used (kept for compatibility), transformer operates in output_dim
            num_layers: Number of transformer encoder layers (8)
            num_heads: Number of attention heads (8)
            ff_size: Feed-forward network size (2048)
            output_dim: Output dimension (will be used as latent_dim in MotionModel)
            seq_len: Sequence length (20 frames)
            dropout: Dropout rate
            activation: Activation function ('gelu', 'relu', etc.)
        """
        super(TransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Input embedding: directly to output dimension
        self.embed_motion = nn.Linear(input_dim, output_dim)
        
        # Query token (learnable parameter, like CLS token in BERT)
        # Initialize with very small values to prevent gradient explosion
        self.query_token = nn.Parameter(torch.zeros(1, output_dim))
        # Small random initialization
        nn.init.normal_(self.query_token, mean=0.0, std=0.01)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(output_dim, dropout, max_len=seq_len + 1)
        
        # Transformer encoder layers
        # Use norm_first=True (pre-norm) for better training stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm before output
        self.out_ln = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len=20, input_dim=263)
        
        Returns:
            output: Encoded representation (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # Embed motion frames: (batch, 20, 263) -> (batch, 20, output_dim)
        x_emb = self.embed_motion(x)
        
        # Get query token for this batch: (1, output_dim) -> (batch, 1, output_dim)
        # Clamp query token to prevent extreme values
        query_token_clamped = torch.clamp(self.query_token, min=-10, max=10)
        query = query_token_clamped.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate query token with motion frames: (batch, 21, output_dim)
        # [query_token, frame_1, frame_2, ..., frame_20]
        emb = torch.cat([query, x_emb], dim=1)
        
        # Add positional encoding
        h = self.pos_encoder(emb)
        
        # Transformer encoder (no masking needed for fixed-length sequences)
        h = self.transformer(h)
        
        # Layer norm
        h = self.out_ln(h)
        
        # Extract query token output (first position): (batch, output_dim)
        output = h[:, 0]
        
        return output


class TransformerDecoder(nn.Module):
    """Autoregressive transformer-based decoder for motion sequences."""
    
    def __init__(self, input_dim=512, hidden_dim=512, num_layers=8, num_heads=8,
                 ff_size=2048, output_dim=263, seq_len=20, dropout=0.1, activation='gelu'):
        """
        Args:
            input_dim: Input dimension (latent_dim from MotionModel)
            hidden_dim: Not used (kept for compatibility), decoder operates in input_dim
            num_layers: Number of transformer decoder layers (8)
            num_heads: Number of attention heads (8)
            ff_size: Feed-forward network size (2048)
            output_dim: Output feature dimension (263 for HumanML3D)
            seq_len: Sequence length (20 frames)
            dropout: Dropout rate
            activation: Activation function ('gelu', 'relu', etc.)
        """
        super(TransformerDecoder, self).__init__()
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Embed first frame directly to input dimension
        self.embed_first_frame = nn.Sequential(
            nn.Linear(output_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Embed generated frames (for autoregressive generation) to input dimension
        self.embed_frame = nn.Linear(output_dim, input_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(input_dim, dropout, max_len=seq_len)
        
        # Transformer decoder layers
        # Use norm_first=True (pre-norm) for better training stability
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection from input dimension to output dimension
        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),  # Intermediate projection
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim)  # Final output
        )
        
    def forward(self, z, first_frame=None):
        """
        Autoregressive decoding from latent representation.
        
        Args:
            z: Latent tensor of shape (batch_size, input_dim)
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
        
        # Embed first frame directly to input dimension: (batch, output_dim) -> (batch, input_dim)
        first_frame_emb = self.embed_first_frame(first_frame)  # (batch, input_dim)
        
        # Initialize sequence with first frame
        # We'll build the sequence autoregressively
        outputs = []
        current_seq = first_frame_emb.unsqueeze(1)  # (batch, 1, input_dim)
        
        # Use latent z directly as memory (no projection needed)
        memory = z.unsqueeze(1).expand(-1, self.seq_len, -1)  # (batch, seq_len, input_dim)
        
        for t in range(self.seq_len):
            # Add positional encoding
            seq_with_pos = self.pos_encoder(current_seq)  # (batch, t+1, input_dim)
            
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
            )  # (batch, t+1, input_dim)
            
            # Get output for the last (current) position
            current_output = decoder_output[:, -1:]  # (batch, 1, input_dim)
            
            # Project from input dimension to output dimension
            frame_output = self.fc_out(current_output)  # (batch, 1, output_dim)
            outputs.append(frame_output)
            
            # Embed generated frame for next iteration
            if t < self.seq_len - 1:  # Don't need to embed for last frame
                next_frame_emb = self.embed_frame(frame_output)  # (batch, 1, input_dim)
                current_seq = torch.cat([current_seq, next_frame_emb], dim=1)
        
        # Concatenate all outputs
        output = torch.cat(outputs, dim=1)  # (batch, seq_len, output_dim)
        
        return output

