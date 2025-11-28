#!/usr/bin/env python3
"""
Diagnose NaN issue in HML3D processing.
Check keypoints before and during process_file to identify where NaN is introduced.
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

import utils.salsa_utils.HumanMl3D_functions as HM3D_F

NEW_CACHE_PATH = "dataset_processed_New/lmdb_Salsa_pair/lmdb_train_cache_MDM"

def check_keypoints_before_processing(keypoints, name="keypoints"):
    """Check keypoints for issues before processing."""
    print(f"\n{'='*80}")
    print(f"Checking {name} BEFORE process_file")
    print(f"{'='*80}")
    
    print(f"  Shape: {keypoints.shape}")
    print(f"  Dtype: {keypoints.dtype}")
    print(f"  Has NaN: {np.isnan(keypoints).any()}")
    print(f"  Has Inf: {np.isinf(keypoints).any()}")
    print(f"  Min: {keypoints.min():.6f}, Max: {keypoints.max():.6f}")
    print(f"  Mean: {keypoints.mean():.6f}, Std: {keypoints.std():.6f}")
    
    # Check specific joints used in process_file
    face_joint_indx = [2, 1, 17, 16]  # r_hip, l_hip, sdr_r, sdr_l
    joint_names = ['r_hip', 'l_hip', 'sdr_r', 'sdr_l']
    
    print(f"\n  Joints used for orientation (first frame):")
    root_pos_init = keypoints[0]
    for i, (idx, name) in enumerate(zip(face_joint_indx, joint_names)):
        if idx < keypoints.shape[1]:
            joint_pos = root_pos_init[idx]
            print(f"    {name} (joint {idx}): {joint_pos}")
        else:
            print(f"    {name} (joint {idx}): OUT OF RANGE!")
    
    # Check across vector calculation
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    if all(idx < keypoints.shape[1] for idx in face_joint_indx):
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across_norm = np.sqrt((across ** 2).sum())
        
        print(f"\n  Across vector calculation:")
        print(f"    across1 (hip diff): {across1}, norm: {np.linalg.norm(across1):.6f}")
        print(f"    across2 (shoulder diff): {across2}, norm: {np.linalg.norm(across2):.6f}")
        print(f"    across (sum): {across}, norm: {across_norm:.6f}")
        
        if across_norm < 1e-6:
            print(f"    ⚠️  WARNING: across vector is too small! This will cause division by zero!")
        else:
            print(f"    ✓ across vector norm is OK")
        
        # Check forward_init
        forward_init = np.cross(np.array([0, 1, 0]), across)
        forward_norm = np.linalg.norm(forward_init)
        print(f"    forward_init: {forward_init}, norm: {forward_norm:.6f}")
        
        if forward_norm < 1e-6:
            print(f"    ⚠️  WARNING: forward_init vector is too small! This will cause division by zero!")
        else:
            print(f"    ✓ forward_init norm is OK")

def test_process_file_step_by_step(keypoints, name="keypoints"):
    """Test process_file step by step to find where NaN is introduced."""
    print(f"\n{'='*80}")
    print(f"Testing process_file step by step for {name}")
    print(f"{'='*80}")
    
    positions = keypoints.copy()
    
    # Step 1: Uniform skeleton
    try:
        print("\n  Step 1: Uniform skeleton...")
        positions = HM3D_F.uniform_skeleton(positions, HM3D_F.tgt_offsets)
        print(f"    ✓ Shape: {positions.shape}, Has NaN: {np.isnan(positions).any()}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return
    
    # Step 2: Put on floor
    try:
        print("\n  Step 2: Put on floor...")
        floor_height = positions.min(axis=0).min(axis=0)[1]
        positions[:, :, 1] -= floor_height
        print(f"    ✓ Floor height: {floor_height:.6f}, Has NaN: {np.isnan(positions).any()}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return
    
    # Step 3: XZ at origin
    try:
        print("\n  Step 3: XZ at origin...")
        root_pos_init = positions[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        positions = positions - root_pose_init_xz
        print(f"    ✓ Has NaN: {np.isnan(positions).any()}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return
    
    # Step 4: All initially face Z+ (THIS IS WHERE THE ISSUE LIKELY IS)
    try:
        print("\n  Step 4: All initially face Z+ (CRITICAL STEP)...")
        r_hip, l_hip, sdr_r, sdr_l = HM3D_F.face_joint_indx
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across_norm = np.sqrt((across ** 2).sum(axis=-1))
        print(f"    across norm: {across_norm:.10f}")
        
        if across_norm < 1e-10:
            print(f"    ⚠️  CRITICAL: across norm is too small! Division by zero will occur!")
            print(f"    across1: {across1}, norm: {np.linalg.norm(across1):.10f}")
            print(f"    across2: {across2}, norm: {np.linalg.norm(across2):.10f}")
            return
        
        across = across / across_norm[..., np.newaxis]
        print(f"    ✓ Normalized across: {across}")
        
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward_norm = np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
        print(f"    forward_init norm: {forward_norm[0, 0]:.10f}")
        
        if forward_norm[0, 0] < 1e-10:
            print(f"    ⚠️  CRITICAL: forward_init norm is too small! Division by zero will occur!")
            return
        
        forward_init = forward_init / forward_norm
        print(f"    ✓ Normalized forward_init: {forward_init[0]}")
        
        target = np.array([[0, 0, 1]])
        root_quat_init = HM3D_F.qbetween_np(forward_init, target)
        print(f"    ✓ root_quat_init: {root_quat_init[0]}, Has NaN: {np.isnan(root_quat_init).any()}")
        
        root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
        positions = HM3D_F.qrot_np(root_quat_init, positions)
        print(f"    ✓ After rotation, Has NaN: {np.isnan(positions).any()}")
        
    except Exception as e:
        print(f"    ✗ Error in Step 4: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Continue with rest of process_file
    try:
        print("\n  Step 5: Continue with process_file...")
        data, _, _, _ = HM3D_F.process_file(keypoints, 0.002)
        print(f"    ✓ Final data shape: {data.shape}, Has NaN: {np.isnan(data).any()}")
        if np.isnan(data).any():
            nan_count = np.isnan(data).sum()
            total = data.size
            print(f"    ⚠️  NaN count: {nan_count}/{total} ({100*nan_count/total:.2f}%)")
    except Exception as e:
        print(f"    ✗ Error in process_file: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("="*80)
    print("DIAGNOSING NaN ISSUE IN HML3D PROCESSING")
    print("="*80)
    
    # Load sample from new cache
    if not os.path.exists(NEW_CACHE_PATH):
        print(f"Error: Cache path not found: {NEW_CACHE_PATH}")
        return
    
    lmdb_env = lmdb.open(NEW_CACHE_PATH, readonly=True, lock=False)
    
    with lmdb_env.begin() as txn:
        n_samples = txn.stat()['entries']
        print(f"\nTotal samples in cache: {n_samples}")
        
        key = "{:010}".format(0).encode("ascii")
        value = txn.get(key)
        
        if value is None:
            print(f"Error: Key {key.decode()} not found")
            return
        
        sample = pyarrow.deserialize(value)
    
    lmdb_env.close()
    
    # Extract keypoints (assuming MDM format)
    if len(sample) == 13:
        poses_keypoints3d_L = sample[0]  # Leader keypoints
        poses_keypoints3d_F = sample[5]  # Follower keypoints
    else:
        print(f"Unexpected sample format with {len(sample)} elements")
        return
    
    # Convert to numpy if torch tensor
    if isinstance(poses_keypoints3d_L, torch.Tensor):
        poses_keypoints3d_L = poses_keypoints3d_L.detach().cpu().numpy()
    if isinstance(poses_keypoints3d_F, torch.Tensor):
        poses_keypoints3d_F = poses_keypoints3d_F.detach().cpu().numpy()
    
    # Check leader
    check_keypoints_before_processing(poses_keypoints3d_L, "Leader keypoints")
    test_process_file_step_by_step(poses_keypoints3d_L, "Leader")
    
    # Check follower
    check_keypoints_before_processing(poses_keypoints3d_F, "Follower keypoints")
    test_process_file_step_by_step(poses_keypoints3d_F, "Follower")
    
    print(f"\n{'='*80}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

