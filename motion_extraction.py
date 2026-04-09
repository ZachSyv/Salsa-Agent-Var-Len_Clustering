import torch
import numpy as np
from scipy.spatial.distance import cdist
import pickle
import os
from tqdm import tqdm

from motion_representation.models.motion_model import MotionModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# make sure these parameters match the training config
model = MotionModel(
    encoder_type='tcn',
    decoder_type='tcn',
    downsampling_factor=4,
    latent_dim=512,
    hidden_dim=512,
    num_layers=2,
    use_vqvae=True,
    nb_code=512
).to(device)

checkpoint_path = "motion_representation/checkpoints_VQVAE_TCN_ds4_1/best_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # CRITICAL for inference

norm_path = "dataset_processed_New/lmdb_Salsa_pair/lmdb_train_humanml3d_20frames_cache/normalization_stats.pkl"
with open(norm_path, "rb") as f:
    norm_stats = pickle.load(f)

print("\n--- STEP 1: CODEBOOK DISTANCES ---")
with torch.no_grad():
    state_dict = model.state_dict()
    
    embed_key = [k for k in state_dict.keys() if 'codebook' in k or 'embed' in k][0]
    embeddings = state_dict[embed_key].cpu().numpy()
    
    # Calculate lookup table
    token_distance_matrix = cdist(embeddings, embeddings, metric='euclidean')
    np.save("token_distance_matrix.npy", token_distance_matrix)
    print(f"Saved {embeddings.shape} distance matrix to token_distance_matrix.npy")

print("\n--- STEP 2: EXTRACT TOKEN SEQUENCES ---")
data_path = "dataset_processed_New/clean_uncut_salsa_train.pkl"
with open(data_path, "rb") as f:
    raw_salsanet_data = pickle.load(f)

all_clip_tokens = {}

with torch.no_grad():
    for vid_name, clips in tqdm(raw_salsanet_data.items(), desc="Processing Videos"):
        for clip_idx, clip_dict in enumerate(clips):
            # extract tokens for both the Leader (_L) and Follower (_F)
            for role in ['L', 'F']:
                key_name = f'HML3D_joints_vec_{role}'
                
                if key_name in clip_dict:
                    raw_motion = clip_dict[key_name] # Shape: (Seq_Len, 263)
                    normalized_clip = (raw_motion - norm_stats['mean']) / norm_stats['std']
                    
                    clip_tensor = torch.tensor(normalized_clip, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Instead of breaking it apart, just do a full forward pass, the model naturally returns the token IDs (code_idx) as the 5th output
                    _, _, _, _, code_idx = model(clip_tensor)
                    
                    unique_id = f"{vid_name}_clip_{clip_idx}_{'Leader' if role == 'L' else 'Follower'}"
                    all_clip_tokens[unique_id] = code_idx.cpu().numpy()

# Save the final variable-length token arrays
with open("extracted_salsa_tokens.pkl", "wb") as f:
    pickle.dump(all_clip_tokens, f)

print(f"\nExtraction complete! Saved {len(all_clip_tokens)} variable-length token sequences.")
