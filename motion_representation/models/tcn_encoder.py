import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        return self.relu2(out + residual)

class TCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, downsampling_factor=4):
        super().__init__()
        
        # Setting kernel_size and stride to the downsampling_factor with 0 padding 
        # guarantees clean sequence compression without fractional temporal dimension dropoff.
        self.downsample = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=hidden_dim, 
            kernel_size=downsampling_factor, 
            stride=downsampling_factor
        )
        
        self.res_blocks = nn.Sequential(
            ResBlock1D(hidden_dim),
            ResBlock1D(hidden_dim),
            ResBlock1D(hidden_dim)
        )
        
        self.to_latent = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=latent_dim, 
            kernel_size=3, 
            padding=1
        )

    def forward(self, x):
        # x input shape: (Batch, Time, Feature)
        # Convert to (Batch, Channels, Time)
        x = x.permute(0, 2, 1)
        
        x = self.downsample(x)
        x = self.res_blocks(x)
        x = self.to_latent(x)
        
        # Convert back to (Batch, Downsampled_Time, Latent_Dim) for the VQ layer
        x = x.permute(0, 2, 1)
        return x