# VAE Visualization App

Interactive web application for visualizing VAE model reconstructions and exploring the latent space.

## Features

1. **Model & Dataset Loading**
   - Load VAE checkpoints
   - Load motion dataset from LMDB
   - Pre-filled default paths

2. **Reconstruction Visualization**
   - Browse samples by index
   - View original vs reconstructed motions side by side
   - Display reconstruction metrics (MSE, MAE, latent statistics)

3. **Latent Space Visualization**
   - Select number of samples to encode (10-10000)
   - Randomly sample from dataset
   - Encode using VAE encoder
   - Visualize latent space using t-SNE

## Usage

### Launch the App

```bash
cd /localhome/pjomeyaz/Payam_Files/Projects/Salsa_Dance/scripts/New_2025/Salsa-Agent
python -m motion_representation.visualization.vae_visualization_app
```

The app will be available at: `http://localhost:7862`

### Quick Start

1. **Load Model**: Click "Load Model" (default path: `motion_representation/checkpoints/best_checkpoint.pth`)
2. **Load Dataset**: Click "Load Dataset" (default path: `dataset_processed_New/lmdb_Salsa_pair/lmdb_train`)
3. **Visualize Reconstruction**: 
   - Navigate to a sample index
   - Click "Visualize Reconstruction" to see original vs reconstructed
4. **Explore Latent Space**:
   - Set number of samples (e.g., 100, 1000)
   - Click "Generate Latent Space Visualization"
   - View t-SNE plot of encoded latents

## Requirements

- gradio
- scikit-learn (for t-SNE)
- matplotlib
- torch

