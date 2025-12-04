# VAE Architecture Options

The MotionVAE model now supports both GRU and Transformer architectures.

## Architecture Selection

### GRU (Default)
- **Encoder**: Bidirectional GRU with FC layers
- **Decoder**: Autoregressive GRU decoder
- **Use Case**: Faster training, fewer parameters, good for short sequences

### Transformer
- **Encoder**: Transformer encoder with query token approach
- **Decoder**: Autoregressive transformer decoder with causal masking
- **Use Case**: Better long-range dependencies, more expressive, better for complex patterns

## Configuration

### Command Line Arguments

```bash
# Use GRU (default)
python -m motion_representation.train --encoder_type gru --decoder_type gru

# Use Transformer
python -m motion_representation.train \
    --encoder_type transformer \
    --decoder_type transformer \
    --num_layers 8 \
    --num_heads 8 \
    --ff_size 2048 \
    --activation gelu

# Mix architectures (not recommended, but possible)
python -m motion_representation.train \
    --encoder_type transformer \
    --decoder_type gru
```

### Transformer Hyperparameters

- `num_layers`: 8 (number of transformer layers)
- `num_heads`: 8 (attention heads)
- `ff_size`: 2048 (feed-forward network size)
- `activation`: 'gelu' (activation function)
- `hidden_dim`: 512 (transformer dimension, same as GRU hidden_dim)

## Architecture Details

### Transformer Encoder
- Input embedding: `263 → 512`
- Query token: Learnable parameter (like CLS token)
- Positional encoding: Sinusoidal
- Transformer encoder: 8 layers, 8 heads
- Output: Mean and logvar `(batch, 2048)`

### Transformer Decoder
- Latent projection: `2048 → 512`
- First frame embedding: `263 → 512`
- Autoregressive generation with causal masking
- Transformer decoder: 8 layers, 8 heads
- Output: Reconstructed motion `(batch, 20, 263)`

## Performance Comparison

| Aspect | GRU | Transformer |
|--------|-----|-------------|
| Parameters | ~10M | ~15-20M |
| Training Speed | Faster | Slower |
| Memory | Lower | Higher |
| Long-range Dependencies | Limited | Excellent |
| Interpretability | Lower | Higher (attention) |

## Recommendations

- **Start with GRU**: Faster to train, good baseline
- **Use Transformer**: For better quality, longer sequences, or when GRU plateaus
- **Experiment**: Try both and compare reconstruction quality

