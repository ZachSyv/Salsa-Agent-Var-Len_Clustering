"""
Positional encoding for transformer models.
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    """
    
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        """
        Args:
            d_model: Model dimension (hidden dimension)
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:x.shape[1], :].unsqueeze(0)
        return self.dropout(x)

