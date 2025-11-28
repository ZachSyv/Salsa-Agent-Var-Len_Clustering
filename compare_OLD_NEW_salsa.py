#!/usr/bin/env python3
"""
Compare motion data between old and new cache LMDB files.
This script loads one sample from each cache and compares motion-related entries
to identify differences that might cause VQVAE reconstruction issues.
"""

import os
import sys
import numpy as np
import torch
import lmdb
import pyarrow
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Paths
OLD_CACHE_PATH = "dataset_processed_old1/lmdb_Salsa_pair/lmdb_train_cache_MDM"
NEW_CACHE_PATH = "dataset_processed_New/lmdb_Salsa_pair/lmdb_train_cache_MDM"

def load_sample_from_cache(cache_path, idx=0):
    """Load a sample from cache LMDB."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache path not found: {cache_path}")
    
    lmdb_env = lmdb.open(cache_path, readonly=True, lock=False)
    
    with lmdb_env.begin() as txn:
        n_samples = txn.stat()['entries']
        print(f"  Total samples in cache: {n_samples}")
        
        if idx >= (n_samples - 1):
            idx = 0
            print(f"  Index {idx} out of range, using index 0")
        
        key = "{:010}".format(idx).encode("ascii")
        value = txn.get(key)
        
        if value is None:
            raise KeyError(f"Key {key.decode()} not found in LMDB")
        
        sample = pyarrow.deserialize(value)
    
    lmdb_env.close()
    return sample, idx

def compare_arrays(name, arr_old, arr_new, tolerance=1e-6):
    """Compare two arrays and print statistics."""
    print(f"\n{'='*80}")
    print(f"Comparing: {name}")
    print(f"{'='*80}")
    
    # Basic info
    print(f"  Shape OLD: {arr_old.shape if hasattr(arr_old, 'shape') else 'N/A'}")
    print(f"  Shape NEW: {arr_new.shape if hasattr(arr_new, 'shape') else 'N/A'}")
    print(f"  Dtype OLD: {arr_old.dtype if hasattr(arr_old, 'dtype') else type(arr_old)}")
    print(f"  Dtype NEW: {arr_new.dtype if hasattr(arr_new, 'dtype') else type(arr_new)}")
    
    if not isinstance(arr_old, np.ndarray) or not isinstance(arr_new, np.ndarray):
        print(f"  WARNING: One or both are not numpy arrays!")
        return
    
    if arr_old.shape != arr_new.shape:
        print(f"  ⚠️  SHAPE MISMATCH!")
        return
    
    # Convert to numpy if torch tensors
    if isinstance(arr_old, torch.Tensor):
        arr_old = arr_old.detach().cpu().numpy()
    if isinstance(arr_new, torch.Tensor):
        arr_new = arr_new.detach().cpu().numpy()
    
    # Statistics
    print(f"\n  Statistics OLD:")
    print(f"    Min: {arr_old.min():.6f}, Max: {arr_old.max():.6f}")
    print(f"    Mean: {arr_old.mean():.6f}, Std: {arr_old.std():.6f}")
    print(f"    Median: {np.median(arr_old):.6f}")
    
    print(f"\n  Statistics NEW:")
    print(f"    Min: {arr_new.min():.6f}, Max: {arr_new.max():.6f}")
    print(f"    Mean: {arr_new.mean():.6f}, Std: {arr_new.std():.6f}")
    print(f"    Median: {np.median(arr_new):.6f}")
    
    # Differences
    diff = arr_old - arr_new
    abs_diff = np.abs(diff)
    
    print(f"\n  Differences:")
    print(f"    Max absolute diff: {abs_diff.max():.6f}")
    print(f"    Mean absolute diff: {abs_diff.mean():.6f}")
    print(f"    RMS diff: {np.sqrt(np.mean(diff**2)):.6f}")
    
    # Check if identical
    if np.allclose(arr_old, arr_new, atol=tolerance):
        print(f"  ✓ Arrays are identical (within tolerance {tolerance})")
    else:
        print(f"  ⚠️  Arrays differ!")
        num_different = np.sum(abs_diff > tolerance)
        total_elements = arr_old.size
        print(f"    {num_different}/{total_elements} elements differ (>{tolerance})")
        print(f"    Percentage different: {100*num_different/total_elements:.2f}%")
        
        # Show locations of largest differences
        if num_different > 0:
            flat_idx = np.argmax(abs_diff)
            idx = np.unravel_index(flat_idx, arr_old.shape)
            print(f"    Largest diff at index {idx}: OLD={arr_old[idx]:.6f}, NEW={arr_new[idx]:.6f}, DIFF={abs_diff[idx]:.6f}")
    
    # For 3D keypoints, analyze each dimension
    if len(arr_old.shape) == 3 and arr_old.shape[2] == 3:
        print(f"\n  Per-dimension analysis (X, Y, Z):")
        for dim, dim_name in enumerate(['X', 'Y', 'Z']):
            old_dim = arr_old[:, :, dim]
            new_dim = arr_new[:, :, dim]
            diff_dim = np.abs(old_dim - new_dim)
            print(f"    {dim_name}: Max diff={diff_dim.max():.6f}, Mean diff={diff_dim.mean():.6f}, RMS={np.sqrt(np.mean((old_dim - new_dim)**2)):.6f}")

def compare_sample_structures(sample_old, sample_new):
    """Compare the structure and content of two samples."""
    print(f"\n{'#'*80}")
    print(f"# SAMPLE STRUCTURE COMPARISON")
    print(f"{'#'*80}")
    
    print(f"\nOLD sample type: {type(sample_old)}")
    print(f"NEW sample type: {type(sample_new)}")
    
    if isinstance(sample_old, (list, tuple)) and isinstance(sample_new, (list, tuple)):
        print(f"\nOLD sample length: {len(sample_old)}")
        print(f"NEW sample length: {len(sample_new)}")
        
        if len(sample_old) != len(sample_new):
            print(f"  ⚠️  LENGTH MISMATCH!")
            return
        
        # Expected structure for MDM format (13 elements):
        # poses_keypoints3d_L, poses_rotmat_L, HML3D_L, ms_desc_L, vq_tokens_L,
        # poses_keypoints3d_F, poses_rotmat_F, HML3D_F, ms_des_F, vq_tokens_F,
        # audio_tokens, audio_raw, aux_info
        
        field_names = [
            'poses_keypoints3d_L', 'poses_rotmat_L', 'HML3D_L', 'ms_desc_L', 'vq_tokens_L',
            'poses_keypoints3d_F', 'poses_rotmat_F', 'HML3D_F', 'ms_des_F', 'vq_tokens_F',
            'audio_tokens', 'audio_raw', 'aux_info'
        ]
        
        if len(sample_old) == 13:
            print("\n  Detected MDM format (13 elements)")
        elif len(sample_old) == 11:
            print("\n  Detected non-MDM format (11 elements)")
            field_names = [
                'poses_keypoints3d_L', 'poses_rotmat_L', 'ms_desc_L', 'vq_tokens_L',
                'poses_keypoints3d_F', 'poses_rotmat_F', 'ms_des_F', 'vq_tokens_F',
                'audio_tokens', 'audio_raw', 'aux_info'
            ]
        else:
            print(f"\n  ⚠️  Unexpected format with {len(sample_old)} elements")
            field_names = [f'field_{i}' for i in range(len(sample_old))]
        
        # Compare motion-related fields
        motion_fields = [
            'poses_keypoints3d_L', 'poses_keypoints3d_F',
            'poses_rotmat_L', 'poses_rotmat_F',
            'HML3D_L', 'HML3D_F',
            'vq_tokens_L', 'vq_tokens_F'
        ]
        
        for i, field_name in enumerate(field_names):
            if i < len(sample_old) and i < len(sample_new):
                val_old = sample_old[i]
                val_new = sample_new[i]
                
                if field_name in motion_fields:
                    print(f"\n  Field {i}: {field_name}")
                    if isinstance(val_old, (np.ndarray, torch.Tensor)) and isinstance(val_new, (np.ndarray, torch.Tensor)):
                        compare_arrays(field_name, val_old, val_new)
                    else:
                        print(f"    OLD type: {type(val_old)}")
                        print(f"    NEW type: {type(val_new)}")
                        if isinstance(val_old, (list, tuple)) and isinstance(val_new, (list, tuple)):
                            print(f"    OLD length: {len(val_old)}")
                            print(f"    NEW length: {len(val_new)}")
                elif field_name == 'aux_info':
                    print(f"\n  Field {i}: {field_name} (metadata, skipping detailed comparison)")
                    if isinstance(val_old, dict) and isinstance(val_new, dict):
                        print(f"    OLD keys: {list(val_old.keys())}")
                        print(f"    NEW keys: {list(val_new.keys())}")
                        if 'vid' in val_old and 'vid' in val_new:
                            print(f"    OLD vid: {val_old['vid']}")
                            print(f"    NEW vid: {val_new['vid']}")

def main():
    print("="*80)
    print("COMPARING OLD vs NEW SALSA CACHE DATA")
    print("="*80)
    
    # Load samples
    print(f"\nLoading OLD sample from: {OLD_CACHE_PATH}")
    try:
        sample_old, idx_old = load_sample_from_cache(OLD_CACHE_PATH, idx=0)
        print(f"  ✓ Loaded sample at index {idx_old}")
    except Exception as e:
        print(f"  ✗ Error loading OLD sample: {e}")
        return
    
    print(f"\nLoading NEW sample from: {NEW_CACHE_PATH}")
    try:
        sample_new, idx_new = load_sample_from_cache(NEW_CACHE_PATH, idx=0)
        print(f"  ✓ Loaded sample at index {idx_new}")
    except Exception as e:
        print(f"  ✗ Error loading NEW sample: {e}")
        return
    
    # Compare structures
    compare_sample_structures(sample_old, sample_new)
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

