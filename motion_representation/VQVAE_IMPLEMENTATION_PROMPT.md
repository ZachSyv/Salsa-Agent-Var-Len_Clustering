# Prompt for VQ-VAE Implementation

## Task
I want to convert my existing motion representation model from a standard VAE to a VQ-VAE (Vector Quantized Variational Autoencoder) with minimal changes to the codebase. I will provide you with a VQ-VAE example implementation that you should use as reference.

## Current Motion Model Architecture

### Overview
- **Model Class**: `MotionModel` in `motion_representation/models/motion_model.py`
- **Current Implementation**: Standard VAE with optional vanilla autoencoder mode
- **Architecture Support**: GRU or Transformer encoders/decoders
- **Input**: Motion sequences of shape `(batch_size, seq_len=20, input_dim=263)` (HumanML3D format)
- **Output**: Reconstructed motion sequences of same shape

### Key Components

#### 1. MotionModel (`motion_representation/models/motion_model.py`)

**Current Structure:**
- `__init__()`: Creates encoder and decoder based on `encoder_type` and `decoder_type` ('gru' or 'transformer')
- `encode(x)`: Encodes input to latent representation
  - If `use_vae=True`: Projects to mean/logvar, then reparameterizes
  - If `use_vae=False`: Returns encoder output directly
  - Returns: `(z, mean, logvar)` where `z` is the latent code
- `decode(z, first_frame)`: Decodes latent representation to motion
- `forward(x)`: Full forward pass, returns `(recon_x, mean, logvar, z)`

**Current VAE Implementation:**
```python
# In encode() method when use_vae=True:
mean = self.fc_mean(encoded)  # Linear projection to mean
logvar = self.fc_logvar(encoded)  # Linear projection to logvar
z = self.reparameterize(mean, logvar)  # Reparameterization trick
```

**Loss Function:**
- `vae_loss()` function in same file
- Reconstruction loss: MSE between `recon_x` and `x`
- KL loss: KL divergence between latent distribution and standard normal (only if `use_vae=True`)
- Total: `recon_weight * recon_loss + kl_weight * kl_loss`

#### 2. Encoders/Decoders

**GRU Encoder** (`motion_representation/models/encdec_gru.py`):
- Bidirectional GRU with FC layers
- Input: `(batch, seq_len=20, input_dim=263)`
- Output: `(batch, latent_dim=512)`

**Transformer Encoder** (`motion_representation/models/encdec_transformer.py`):
- Multi-head self-attention with positional encoding
- Input: `(batch, seq_len=20, input_dim=263)`
- Output: `(batch, latent_dim=512)`

**Decoders**: Similar structure, take `(batch, latent_dim)` and output `(batch, seq_len, input_dim)`

#### 3. Training Script (`motion_representation/train.py`)

**Key Training Flow:**
```python
# Forward pass
recon_motion, mean, logvar, z = model(motion)

# Loss computation
loss, recon_loss, kl_loss = vae_loss(
    recon_motion, motion, mean, logvar,
    recon_weight=config.recon_weight,
    kl_weight=config.kl_weight,
    use_vae=config.use_vae
)
```

**Configuration** (`motion_representation/config.py`):
- `--use_vae`: Flag to enable VAE mode
- `--recon_weight`: Weight for reconstruction loss (default: 1.0)
- `--kl_weight`: Weight for KL loss (default: 0.0001)
- `--latent_dim`: Latent dimension (default: 512)

### Current File Structure
```
motion_representation/
├── models/
│   ├── motion_model.py          # Main MotionModel class + vae_loss()
│   ├── encdec_gru.py            # GRU encoder/decoder
│   └── encdec_transformer.py    # Transformer encoder/decoder
├── train.py                      # Training script
└── config.py                     # Configuration parser
```

## Requirements for VQ-VAE Implementation

### What Needs to Change

1. **Replace reparameterization with vector quantization**:
   - Remove `fc_mean` and `fc_logvar` projections
   - Add codebook (embedding table) and quantization layer
   - Replace `reparameterize()` with vector quantization

2. **Update loss function**:
   - Remove KL divergence loss
   - Add codebook loss (commitment loss)
   - Add quantization loss (stop-gradient trick)
   - Keep reconstruction loss

3. **Maintain compatibility**:
   - Keep same interface: `encode()`, `decode()`, `forward()` should work similarly
   - Support both GRU and Transformer architectures
   - Keep `use_vae` flag or add new `use_vqvae` flag
   - Maintain checkpoint compatibility where possible

4. **Minimal changes**:
   - Don't modify encoder/decoder implementations
   - Don't change training loop structure significantly
   - Keep configuration system similar
   - Preserve checkpoint saving/loading logic

### Expected VQ-VAE Components

Based on standard VQ-VAE architecture:
- **Codebook**: Embedding table of size `(num_embeddings, embedding_dim)`
- **Quantization**: Find nearest embedding, return quantized vector
- **Losses**:
  - Reconstruction loss (same as before)
  - Codebook loss: `||sg(z_e) - z_q||^2` (stop gradient on encoder output)
  - Commitment loss: `||z_e - sg(z_q)||^2` (stop gradient on quantized)

### Configuration Additions Needed

- `--use_vqvae`: Flag to enable VQ-VAE mode
- `--num_embeddings`: Size of codebook (e.g., 512, 1024)
- `--embedding_dim`: Dimension of codebook embeddings (typically same as `latent_dim`)
- `--vq_commitment_cost`: Weight for commitment loss (e.g., 0.25)

## Implementation Guidelines

1. **Use the provided VQ-VAE example** as reference for the quantization layer implementation
2. **Integrate into existing MotionModel** class with minimal refactoring
3. **Preserve existing functionality**: Vanilla autoencoder and standard VAE modes should still work
4. **Update loss function** to handle VQ-VAE losses appropriately
5. **Test compatibility**: Ensure training loop works without major modifications

## Next Steps

After you provide the VQ-VAE example code, I will share it and you can:
1. Analyze the example implementation
2. Integrate it into the MotionModel architecture
3. Update the loss function
4. Ensure minimal changes to the codebase
5. Provide updated training commands

Please confirm you understand the architecture and are ready to proceed with the implementation once I provide the example code.

