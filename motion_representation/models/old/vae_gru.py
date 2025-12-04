"""
Vanilla VAE with GRU or Transformer layers for Salsa dance motion representation learning.
Supports both GRU and Transformer architectures via config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    """Bidirectional GRU-based encoder for motion sequences with FC layers."""
    
    def __init__(self, input_dim=263, hidden_dim=512, num_layers=2, latent_dim=2048, dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension (263 for HumanML3D)
            hidden_dim: Hidden dimension of GRU layers
            num_layers: Number of GRU layers
            latent_dim: Dimension of latent representation (2048)
            dropout: Dropout rate
        """
        super(GRUEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.bidirectional = True
        
        # Input FC layer with activation
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional GRU layers
        self.gru = nn.GRU(
            hidden_dim,  # Input after FC
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional!
        )
        
        # GRU output dimension (bidirectional: 2 * hidden_dim)
        gru_output_dim = hidden_dim * 2
        
        # Intermediate FC layer
        self.fc_intermediate = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Projection to latent space (mean and log-variance)
        # Initialize with smaller weights to prevent large initial logvar
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize logvar projection to output small values initially
        nn.init.xavier_uniform_(self.fc_mean.weight, gain=0.1)
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
        # x: (batch, 20, 263)
        # Apply input FC
        x = self.fc_in(x)  # (batch, 20, hidden_dim)
        
        # Bidirectional GRU
        gru_out, hidden = self.gru(x)
        # gru_out: (batch, 20, hidden_dim * 2) [bidirectional]
        # hidden: (num_layers * 2, batch, hidden_dim) [bidirectional]
        
        # Concatenate forward and backward hidden states from last layer
        forward_hidden = hidden[self.num_layers - 1]  # (batch, hidden_dim)
        backward_hidden = hidden[2 * self.num_layers - 1]  # (batch, hidden_dim)
        last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_dim * 2)
        
        # Apply intermediate FC
        last_hidden = self.fc_intermediate(last_hidden)  # (batch, hidden_dim)
        
        # Project to latent space
        mean = self.fc_mean(last_hidden)  # (batch, latent_dim)
        logvar = self.fc_logvar(last_hidden)  # (batch, latent_dim)
        
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mean, logvar


class GRUDecoder(nn.Module):
    """GRU-based decoder for motion sequences with first frame input and autoregressive reconstruction."""
    
    def __init__(self, latent_dim=2048, hidden_dim=512, num_layers=2, output_dim=263, seq_len=20, dropout=0.1):
        """
        Args:
            latent_dim: Dimension of latent representation (2048)
            hidden_dim: Hidden dimension of GRU layers
            num_layers: Number of GRU layers
            output_dim: Output feature dimension (263 for HumanML3D)
            seq_len: Sequence length (20 frames)
            dropout: Dropout rate
        """
        super(GRUDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Project latent to initial hidden state
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * num_layers)
        )
        
        # Input FC for first frame
        self.fc_first_frame = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GRU layers
        self.gru = nn.GRU(
            hidden_dim,  # Input after FC
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection with activation
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z, first_frame=None):
        """
        Args:
            z: Latent tensor of shape (batch_size, latent_dim=2048)
            first_frame: First frame of original motion (batch_size, output_dim=263)
                        If None, uses zeros
        
        Returns:
            output: Reconstructed motion (batch_size, seq_len=20, output_dim=263)
        """
        batch_size = z.size(0)
        
        # Project latent to initial hidden state
        hidden = self.latent_to_hidden(z)  # (batch, hidden_dim * num_layers)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden_dim)
        
        # Use first frame as initial input, or zeros if not provided
        if first_frame is None:
            decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=z.device)
        else:
            decoder_input = first_frame.unsqueeze(1)  # (batch, 1, output_dim)
        
        # Apply FC to first frame input
        decoder_input = self.fc_first_frame(decoder_input)  # (batch, 1, hidden_dim)
        
        # Decode step by step (autoregressive)
        outputs = []
        for t in range(self.seq_len):
            gru_out, hidden = self.gru(decoder_input, hidden)
            # gru_out: (batch, 1, hidden_dim)
            output = self.fc_out(gru_out)  # (batch, 1, output_dim)
            outputs.append(output)
            
            # Use output as next input (autoregressive)
            # Apply FC to prepare for next GRU step
            decoder_input = self.fc_first_frame(output)  # (batch, 1, hidden_dim)
        
        # Concatenate all outputs
        output = torch.cat(outputs, dim=1)  # (batch, seq_len, output_dim)
        
        return output


# MotionVAE and vae_loss moved to vae.py for unified architecture support
# This file now only contains GRU-specific encoder/decoder implementations

