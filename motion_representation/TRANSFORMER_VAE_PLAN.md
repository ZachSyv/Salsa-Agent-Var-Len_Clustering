# Transformer VAE Architecture Plan

## Analysis of Provided Transformer Encoder Example

### Key Components:

1. **Query Token Approach**:
   - Uses a learnable `query_token` parameter (like CLS token in BERT)
   - Prepended to the sequence: `[query_token, motion_frame_1, motion_frame_2, ...]`
   - Final embedding extracted from query token position: `h[:, 0]`
   - This aggregates information from all frames into a single representation

2. **Input Processing**:
   - `embed_motion`: Linear layer (input_feats*2 → latent_dim)
   - The reshape `x.reshape(B, T, 2, -1)[..., :-4]` seems specific to their data format
   - For our case: Direct embedding from (263-dim) → latent_dim

3. **Positional Encoding**:
   - Sinusoidal positional encoding (standard transformer approach)
   - Adds temporal information to embeddings

4. **Transformer Encoder**:
   - Standard PyTorch `nn.TransformerEncoder`
   - Multi-head self-attention
   - Feed-forward networks
   - Layer normalization

5. **Output**:
   - Single embedding vector (not sequence)
   - Used for downstream tasks

## Adaptation for VAE Architecture

### Encoder Design:

**Input**: `(batch, 20, 263)` HumanML3D motion

**Architecture**:
```
1. Input Embedding: Linear(263 → latent_dim)
2. Query Token: Learnable parameter (1, latent_dim)
3. Concatenate: [query_token, frame_1, frame_2, ..., frame_20] → (batch, 21, latent_dim)
4. Positional Encoding: Add sinusoidal positional encodings
5. Transformer Encoder: Multi-layer self-attention
6. Extract Query Token Output: h[:, 0] → (batch, latent_dim)
7. Project to Latent: 
   - FC_mean: (latent_dim → 2048)
   - FC_logvar: (latent_dim → 2048)
```

**Key Differences from Example**:
- No input reshaping (we use 263-dim directly)
- Output mean/logvar for VAE instead of single embedding
- No masking needed (fixed 20-frame sequences)

### Decoder Design:

**Input**: 
- Latent `z`: (batch, 2048)
- First frame: (batch, 263)

**Architecture**:
```
1. Latent Projection: Linear(2048 → latent_dim) → (batch, latent_dim)
2. First Frame Embedding: Linear(263 → latent_dim) → (batch, latent_dim)
3. Prepare Sequence:
   - Latent as "query" or prepend to sequence
   - First frame as first token
   - Generate remaining 19 frames autoregressively
4. Positional Encoding: Add to sequence
5. Transformer Decoder: 
   - Uses cross-attention to latent
   - Self-attention for sequence
6. Output Projection: Linear(latent_dim → 263) for each frame
```

**Two Decoder Options**:

**Option A: Autoregressive Transformer Decoder**
- Use `nn.TransformerDecoder` with causal masking
- Generate frames one by one
- More similar to GPT-style generation

**Option B: Non-autoregressive with Latent Conditioning**
- Use `nn.TransformerDecoder` with latent as memory
- Generate all frames in parallel
- Faster but potentially less coherent

## Proposed Architecture

### TransformerEncoderVAE:

```python
class TransformerEncoderVAE(nn.Module):
    def __init__(self, input_dim=263, latent_dim=2048, 
                 hidden_dim=512, num_layers=4, num_heads=8,
                 ff_size=2048, dropout=0.1, seq_len=20):
        # Encoder
        self.embed_motion = nn.Linear(input_dim, hidden_dim)
        self.query_token = nn.Parameter(torch.randn(1, hidden_dim))
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=seq_len+1)
        self.transformer_encoder = nn.TransformerEncoder(...)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.embed_first_frame = nn.Linear(input_dim, hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(...)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
```

### Key Design Decisions:

1. **Encoder Query Token**: 
   - ✅ Use query token approach (proven effective)
   - Extract mean/logvar from query token output

2. **Decoder Strategy**:
   - **Recommendation**: Option A (Autoregressive) for better quality
   - Use causal masking for autoregressive generation
   - First frame as initial input, generate rest

3. **Hidden Dimension**:
   - Use `hidden_dim` (e.g., 512) for transformer
   - Project to `latent_dim` (2048) only at output

4. **Positional Encoding**:
   - Use same PositionalEncoding class
   - Max length: 21 (query + 20 frames) for encoder
   - Max length: 20 for decoder

## Implementation Plan

1. **Create `TransformerEncoderVAE` class**:
   - TransformerEncoder for encoding
   - TransformerDecoder for decoding
   - Keep same interface as GRU VAE

2. **Add config option**:
   - `encoder_type`: 'gru' or 'transformer'
   - `decoder_type`: 'gru' or 'transformer'
   - Can mix (e.g., transformer encoder + GRU decoder)

3. **Update MotionVAE**:
   - Factory pattern to create encoder/decoder based on config
   - Keep backward compatibility with GRU

4. **Hyperparameters**:
   - `num_heads`: 8 (standard)
   - `num_layers`: 4-6 (transformer layers)
   - `ff_size`: 2048 (feed-forward dimension)
   - `hidden_dim`: 512 (transformer dimension)

## Advantages of Transformer:

1. **Better Long-range Dependencies**: Self-attention captures all frame relationships
2. **Parallel Processing**: Can process all frames simultaneously (encoder)
3. **Scalability**: Can handle longer sequences better than GRU
4. **Interpretability**: Attention weights show which frames are important

## Potential Challenges:

1. **Computational Cost**: More expensive than GRU for short sequences (20 frames)
2. **Overfitting**: More parameters, need regularization
3. **Training Stability**: May need learning rate scheduling

## Questions for Confirmation:

1. **Decoder Strategy**: Autoregressive (Option A) or Parallel (Option B)?
2. **Mixing Architectures**: Allow transformer encoder + GRU decoder?
3. **Hyperparameters**: Default values for num_layers, num_heads, ff_size?
4. **First Frame Handling**: Keep first frame input approach for transformer decoder?

