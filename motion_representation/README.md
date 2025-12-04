# Motion Representation Learning for Salsa Dance Pairs

This module implements a vanilla VAE with GRU layers for learning motion representations from Salsa dance pairs data.

## Structure

```
motion_representation/
├── models/
│   └── vae_gru.py          # VAE model with GRU encoder/decoder
├── data/
│   └── motion_dataset.py   # Data loader for 20-frame windows
├── utils/
│   └── eval.py             # Evaluation and visualization utilities
├── checkpoints/            # Saved model checkpoints
├── config.py               # Configuration file
├── train.py                # Training script
└── eval.py                 # Evaluation script
```

## Model Architecture

- **Encoder**: GRU-based encoder that takes (batch, 20, 263) motion sequences and outputs 2048-dim latent representation
- **Decoder**: GRU-based decoder that reconstructs (batch, 20, 263) motion from 2048-dim latent
- **Loss**: Reconstruction loss (MSE) + KL divergence regularization

## Data Format

- **Input**: Fixed 20-frame windows of HumanML3D motion vectors (263-dim)
- **Format**: (batch_size, 20, 263)
- **Source**: Salsa dance pairs (leader + follower) from LMDB database

## Usage

### Training

The training script supports multiple architectures (GRU, Transformer) and can work with or without VAE. It also supports automatic resume from the latest checkpoint.

#### Vanilla GRU Autoencoder (No VAE)

```bash
cd /localhome/pjomeyaz/Payam_Files/Projects/Salsa_Dance/scripts/New_2025/Salsa-Agent
conda activate motionagent
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 2 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --model_name Vanilla_GRU
```

#### VAE with GRU

```bash
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 2 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --use_vae \
    --model_name VAE_GRU
```

#### Vanilla Transformer Autoencoder (No VAE)

```bash
python -m motion_representation.train \
    --encoder_type transformer \
    --decoder_type transformer \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 8 \
    --num_heads 8 \
    --ff_size 2048 \
    --activation gelu \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --model_name Vanilla_Transformer
```

#### VAE with Transformer

```bash
python -m motion_representation.train \
    --encoder_type transformer \
    --decoder_type transformer \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 8 \
    --num_heads 8 \
    --ff_size 2048 \
    --activation gelu \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --use_vae \
    --model_name VAE_Transformer
```

#### Resuming Training

The training script automatically detects and resumes from the latest checkpoint in the checkpoint directory. Epoch numbering continues from where it left off.

**Automatic Resume (Recommended):**
```bash
# Simply run the same command again - it will auto-detect the latest checkpoint
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 2 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 200 \
    --model_name Vanilla_GRU
# If checkpoints exist, training continues from the latest epoch
# Epoch numbering: if last checkpoint was epoch 50, new checkpoints start at 51, 52, etc.
```

**Manual Resume (Specify Checkpoint):**
```bash
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 2 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 200 \
    --model_name Vanilla_GRU \
    --resume motion_representation/checkpoints_Vanilla_GRU/checkpoint_epoch_50.pth
```

**Note:** 
- If `--resume` is not specified, the script automatically finds the latest checkpoint in the checkpoint directory
- Epoch numbering continues from the last checkpoint (e.g., if last checkpoint was epoch 50, new checkpoints are saved as epoch 51, 52, etc.)
- The `--num_epochs` parameter specifies the total number of epochs. If resuming from epoch 50 with `--num_epochs 200`, training will continue for 150 more epochs (epochs 51-200)

### Evaluation

```bash
python -m motion_representation.eval \
    --checkpoint motion_representation/checkpoints/best_checkpoint.pth \
    --lmdb_dir dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
    --output_dir motion_representation/eval_outputs
```

### Visualization (Original vs Reconstructed)

Visualize original and reconstructed motions side by side:

```bash
python -m motion_representation.visualize_comparison \
    --checkpoint motion_representation/checkpoints/best_checkpoint.pth \
    --lmdb_dir dataset_processed_New/lmdb_Salsa_pair/lmdb_train \
    --is_MDM \
    --num_samples 5 \
    --output_dir motion_representation/visualizations \
    --fps 20
```

This creates separate video files for original and reconstructed motions that you can view side by side. The filenames include the MSE (Mean Squared Error) for easy comparison.

## Data Cache

The data loader automatically creates a cache of 20-frame windows on first use. The cache is stored at:
- `{lmdb_dir}_VAE_20frames_cache/`

This naming clearly identifies it as the VAE model cache with 20-frame windows, avoiding confusion with other caches.

If you need to regenerate the cache, delete this directory and run training again.

## Configuration

Key parameters:
- `window_size`: 20 (fixed frame window)
- `latent_dim`: 2048 (latent representation dimension)
- `hidden_dim`: 512 (GRU hidden dimension)
- `num_layers`: 2 (number of GRU layers)
- `recon_weight`: 1.0 (reconstruction loss weight)
- `kl_weight`: 0.0001 (KL divergence weight)

## Future Extensions

- VQVAE with codebook quantization
- Relative position modeling (leader-follower relationships)
- Duet generation models (conditional generation)
- Role-specific training (leader-only, follower-only)

