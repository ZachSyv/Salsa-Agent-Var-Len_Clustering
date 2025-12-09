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

#### Vanilla GRU Autoencoder (No VAE, Continuous Latent Space)

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
    --learning_rate 2e-4 \
    --num_epochs 100 \
    --loss_vel_weight 0.1 \
    --model_name Vanilla_GRU_Continuous
```

**Loss Formulation (matches T2M-GPT):**
- **L1 reconstruction loss** (default, matches T2M-GPT's `ReConsLoss` with `recons_loss='l1'`)
- **L1 velocity loss** (temporal smoothness, weight=0.1, matches T2M-GPT's `forward_vel`)
- **Total loss = L1_reconstruction + 0.1 × L1_velocity**

**Note:** This trains a continuous vanilla autoencoder (no VAE, no VQ-VAE). The latent space is continuous (not quantized) and deterministic (no reparameterization).

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

#### VQ-VAE with GRU (Default - Matches T2M-GPT Training Strategy)

```bash
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 2 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --num_epochs 100 \
    --use_vqvae \
    --nb_code 512 \
    --quantizer ema_reset \
    --vq_mu 0.99 \
    --commit_weight 0.02 \
    --loss_vel_weight 0.1 \
    --warm_up_iter 1000 \
    --lr_scheduler multistep \
    --lr_scheduler_milestones 50 200 \
    --lr_scheduler_gamma 0.05 \
    --model_name VQVAE_GRU
```

#### Best Working VQ-VAE Configuration (Recommended)

This configuration has been tested and shows stable training with good perplexity and reconstruction quality:

```bash
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_layers 2 \
    --batch_size 2048 \
    --learning_rate 1e-4 \
    --num_epochs 2000 \
    --use_vqvae \
    --nb_code 512 \
    --quantizer ema_reset \
    --vq_mu 0.95 \
    --commit_weight 0.02 \
    --loss_vel_weight 0.1 \
    --warm_up_epochs 5 \
    --lr_scheduler_gamma 0.05 \
    --model_name VQVAE_GRU
```

**Key differences from default:**
- Larger batch size (2048) for more stable gradients
- Lower learning rate (1e-4) for better convergence
- More epochs (2000) for extended training
- Lower EMA decay (vq_mu 0.95) for more adaptive codebook updates
- Warm-up specified in epochs (5) instead of iterations
- No explicit milestones (uses default [50, 200] if `--lr_scheduler multistep` is set, or constant LR if scheduler is omitted)

#### VQ-VAE with Different Quantizers

**EMA with Reset (Default):**
```bash
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --use_vqvae \
    --quantizer ema_reset \
    --vq_mu 0.99 \
    --model_name VQVAE_GRU_ema_reset
```

**Original Quantizer:**
```bash
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --use_vqvae \
    --quantizer orig \
    --vq_beta 1.0 \
    --model_name VQVAE_GRU_orig
```

**EMA Only:**
```bash
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --use_vqvae \
    --quantizer ema \
    --vq_mu 0.99 \
    --model_name VQVAE_GRU_ema
```

**Reset Only:**
```bash
python -m motion_representation.train \
    --encoder_type gru \
    --decoder_type gru \
    --use_vqvae \
    --quantizer reset \
    --model_name VQVAE_GRU_reset
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
- `latent_dim`: 512 (latent representation dimension)
- `hidden_dim`: 512 (GRU hidden dimension)
- `num_layers`: 2 (number of GRU layers)
- `recon_weight`: 1.0 (reconstruction loss weight)
- `kl_weight`: 0.0001 (KL divergence weight for VAE)

### VQ-VAE Specific Parameters

- `--use_vqvae`: Enable VQ-VAE mode (overrides `--use_vae`)
- `--nb_code`: Number of codebook entries (default: 512)
- `--quantizer`: Quantizer type - `ema_reset` (default), `orig`, `ema`, or `reset`
- `--vq_mu`: EMA decay rate for codebook updates (default: 0.99, used for `ema_reset` and `ema`)
- `--vq_beta`: Beta parameter for original quantizer (default: 1.0, used for `quantizer=orig`)
- `--commit_weight`: Weight for commitment loss (default: 0.02, matches T2M-GPT)
- `--loss_vel_weight`: Weight for velocity loss (default: 0.0, recommended: 0.1 for VQ-VAE to match T2M-GPT)
- `--warm_up_iter`: Number of warm-up iterations with linear LR increase (default: 1000, matches T2M-GPT)
- `--learning_rate`: Learning rate (default: 2e-4, matches T2M-GPT)
- `--lr_scheduler`: Learning rate scheduler - `multistep` recommended for VQ-VAE (matches T2M-GPT)
- `--lr_scheduler_milestones`: Milestones for MultiStepLR in epochs (default: [50, 200] if multistep selected)
- `--lr_scheduler_gamma`: Decay factor for MultiStepLR (default: 0.05, matches T2M-GPT)

## Training Monitoring with TensorBoard

### Launching TensorBoard

TensorBoard logs are automatically saved to `{checkpoint_dir}/logs/`. To monitor training:

```bash
# From the project root directory
tensorboard --logdir motion_representation/checkpoints_VQVAE_GRU/logs --port 6006
```

Then open `http://localhost:6006` in your browser.

### VQ-VAE Metrics Tracked

When training with VQ-VAE (`--use_vqvae`), TensorBoard tracks:

1. **Train/Loss**: Total loss (reconstruction + commitment)
2. **Train/ReconLoss**: Reconstruction loss (MSE between input and reconstructed motion)
3. **Train/CommitLoss**: Commitment loss (ensures encoder commits to codebook)
4. **Train/Perplexity**: Codebook usage perplexity (higher = more codes used, indicates codebook diversity)

### Interpreting VQ-VAE Metrics

- **Reconstruction Loss**: Should decrease over time. Lower is better.
- **Commitment Loss**: Should stabilize. Very high values may indicate codebook collapse.
- **Perplexity**: 
  - Higher perplexity (closer to `nb_code`) = more diverse codebook usage (good)
  - Lower perplexity = fewer codes being used (potential codebook collapse)
  - Ideal range: 50-80% of `nb_code` (e.g., 256-410 for `nb_code=512`)

### Example TensorBoard Dashboard

For VQ-VAE training, you should see:
- **Loss curves** showing convergence
- **Reconstruction loss** decreasing steadily
- **Commitment loss** stabilizing (not growing unbounded)
- **Perplexity** increasing and stabilizing (indicating healthy codebook usage)

### Monitoring Tips

1. **Codebook Collapse**: If perplexity drops significantly (< 10% of `nb_code`), the codebook may be collapsing. Try:
   - Lower learning rate
   - Different quantizer type (`ema_reset` is more robust)
   - Adjust `commit_weight` (try 0.01-0.05)

2. **High Commitment Loss**: If commitment loss is very high, try:
   - Lower `commit_weight` (default 0.02)
   - Check if reconstruction loss is reasonable

3. **Poor Reconstruction**: If reconstruction loss is high, try:
   - Increase `recon_weight` (default 1.0)
   - Check model capacity (increase `hidden_dim` or `num_layers`)

## Future Extensions

- Relative position modeling (leader-follower relationships)
- Duet generation models (conditional generation)
- Role-specific training (leader-only, follower-only)

