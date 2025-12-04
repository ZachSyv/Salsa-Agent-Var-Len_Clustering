# Quick Start Guide

## Setup Verification

First, test that the model works correctly:

```bash
cd /localhome/pjomeyaz/Payam_Files/Projects/Salsa_Dance/scripts/New_2025/Salsa-Agent
python -m motion_representation.test_setup
```

## Training

### Basic Training

```bash
python -m motion_representation.train \
    --lmdb_dir dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
    --is_MDM \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --checkpoint_dir motion_representation/checkpoints
```

### Training with Custom Parameters

```bash
python -m motion_representation.train \
    --lmdb_dir dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
    --is_MDM \
    --batch_size 64 \
    --num_epochs 200 \
    --learning_rate 5e-5 \
    --hidden_dim 1024 \
    --num_layers 3 \
    --latent_dim 2048 \
    --recon_weight 1.0 \
    --kl_weight 0.0001 \
    --checkpoint_dir motion_representation/checkpoints
```

### Resume Training

```bash
python -m motion_representation.train \
    --lmdb_dir dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
    --is_MDM \
    --resume motion_representation/checkpoints/checkpoint_epoch_50.pth \
    --checkpoint_dir motion_representation/checkpoints
```

## Evaluation

```bash
python -m motion_representation.eval \
    --checkpoint motion_representation/checkpoints/best_checkpoint.pth \
    --lmdb_dir dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
    --is_MDM \
    --output_dir motion_representation/eval_outputs \
    --num_samples 20
```

## Visualization (Original vs Reconstructed)

Create side-by-side comparison videos of original and reconstructed motions:

```bash
python -m motion_representation.visualize_comparison \
    --checkpoint motion_representation/checkpoints/best_checkpoint.pth \
    --lmdb_dir dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
    --is_MDM \
    --num_samples 5 \
    --output_dir motion_representation/visualizations \
    --fps 20
```

This will create separate video files for each sample:
- `comparison_sample_000_mse_0.XXXX_original.mp4` - Original motion
- `comparison_sample_000_mse_0.XXXX_reconstructed.mp4` - Reconstructed motion

You can view them side by side or use a video editor to create a split-screen comparison.

## Data Cache

The data loader automatically creates a cache of 20-frame windows on first use. The cache is stored at:
- `{lmdb_dir}_VAE_20frames_cache/`

This naming clearly identifies it as the VAE model cache with 20-frame windows, avoiding confusion with other caches.

If you need to regenerate the cache, delete this directory and run training again.

## Key Parameters

- `window_size`: 20 (fixed frame window size)
- `stride`: 10 (overlap between windows)
- `latent_dim`: 2048 (latent representation dimension)
- `hidden_dim`: 512 (GRU hidden dimension)
- `num_layers`: 2 (number of GRU layers)
- `recon_weight`: 1.0 (reconstruction loss weight)
- `kl_weight`: 0.0001 (KL divergence weight)

## Monitoring Training

Training logs are saved to TensorBoard:
```bash
tensorboard --logdir motion_representation/checkpoints/logs
```

## Output Files

- Checkpoints: `motion_representation/checkpoints/checkpoint_epoch_{N}.pth`
- Best model: `motion_representation/checkpoints/best_checkpoint.pth`
- Evaluation outputs: `motion_representation/eval_outputs/`

