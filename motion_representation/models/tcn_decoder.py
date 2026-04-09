import torch
import torch.nn as nn
from .tcn_encoder import ResBlock1D

class TCNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, downsampling_factor=4):
        super().__init__()
        
        self.from_latent = nn.Conv1d(
            in_channels=latent_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1
        )
        
        self.res_blocks = nn.Sequential(
            ResBlock1D(hidden_dim),
            ResBlock1D(hidden_dim),
            ResBlock1D(hidden_dim)
        )
        
        # reverses the stride compression from the encoder
        self.upsample = nn.ConvTranspose1d(
            in_channels=hidden_dim, 
            out_channels=output_dim, 
            kernel_size=downsampling_factor, 
            stride=downsampling_factor
        )

    def forward(self, x, first_frame=None): # accept first frame so motion_model doesnt crash
        # x input shape from VQ: (Batch, Downsampled_Time, Latent_Dim)
        # convert to (Batch, Channels, Time)
        x = x.permute(0, 2, 1)
        
        x = self.from_latent(x)
        x = self.res_blocks(x)
        x = self.upsample(x)
        
        # convert back to (Batch, Restored_Time, Feature) to compute reconstruction loss
        x = x.permute(0, 2, 1)
        return x