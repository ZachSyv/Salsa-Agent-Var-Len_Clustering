"""
Script to check if HumanML3D data in LMDB is normalized (root position/orientation reset).
This will verify whether leader and follower maintain relative positioning or are normalized independently.
"""

import os
import sys
import numpy as np
import torch
import lmdb
import pyarrow
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import recovery functions
try:
    from utils.salsa_utils.HumanMl3D_functions import recover_root_rot_pos, recover_from_ric
except ImportError as e:
    print(f"Import error: {e}")
    # Try absolute path
    import importlib.util
    hm3d_path = os.path.join(project_root, "utils", "salsa_utils", "HumanMl3D_functions.py")
    if os.path.exists(hm3d_path):
        spec = importlib.util.spec_from_file_location("HumanMl3D_functions", hm3d_path)
        hm3d_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hm3d_module)
        recover_root_rot_pos = hm3d_module.recover_root_rot_pos
        recover_from_ric = hm3d_module.recover_from_ric
    else:
        raise ImportError(f"Could not find HumanMl3D_functions.py at {hm3d_path}")


def check_lmdb_normalization(lmdb_dir, num_samples=5):
    """
    Check normalization status of HumanML3D data in LMDB.
    
    Args:
        lmdb_dir: Path to LMDB database directory
        num_samples: Number of samples to check
    """
    print("="*80)
    print("CHECKING LMDB NORMALIZATION STATUS")
    print("="*80)
    print(f"LMDB Directory: {lmdb_dir}")
    print(f"Checking {num_samples} samples...")
    print()
    
    # Open LMDB
    if not os.path.exists(lmdb_dir):
        print(f"ERROR: LMDB directory not found: {lmdb_dir}")
        return
    
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        total_entries = sum(1 for _ in cursor)
        print(f"Total entries in LMDB: {total_entries}")
        print()
        
        # Reset cursor
        cursor.first()
        
        for sample_idx in range(min(num_samples, total_entries)):
            key = cursor.key()
            value = cursor.value()
            
            # Deserialize
            video_data = pyarrow.deserialize(value)
            
            print(f"{'='*80}")
            print(f"SAMPLE {sample_idx + 1}")
            print(f"{'='*80}")
            print(f"Video ID: {video_data.get('vid', 'N/A')}")
            
            if 'clips' not in video_data or len(video_data['clips']) == 0:
                print("  No clips found in this video")
                cursor.next()
                continue
            
            clip = video_data['clips'][0]
            
            # Get HumanML3D data
            if 'HML3D_joints_vec_L' not in clip or 'HML3D_joints_vec_F' not in clip:
                print("  Missing HML3D data")
                cursor.next()
                continue
            
            data_L = clip['HML3D_joints_vec_L']  # (seq_len, 263)
            data_F = clip['HML3D_joints_vec_F']  # (seq_len, 263)
            
            # Convert to numpy if needed
            if isinstance(data_L, torch.Tensor):
                data_L = data_L.cpu().numpy()
            if isinstance(data_F, torch.Tensor):
                data_F = data_F.cpu().numpy()
            
            seq_len = data_L.shape[0]
            print(f"  Sequence length: {seq_len} frames")
            
            # Recover root positions and rotations
            data_L_tensor = torch.from_numpy(data_L).float().unsqueeze(0)  # (1, seq_len, 263)
            data_F_tensor = torch.from_numpy(data_F).float().unsqueeze(0)  # (1, seq_len, 263)
            
            leader_rot_ang, leader_pos = recover_root_rot_pos(data_L_tensor)
            follower_rot_ang, follower_pos = recover_root_rot_pos(data_F_tensor)
            
            # Remove batch dimension
            leader_rot_ang = leader_rot_ang[0].cpu().numpy()  # (seq_len,)
            leader_pos = leader_pos[0].cpu().numpy()  # (seq_len, 3)
            follower_rot_ang = follower_rot_ang[0].cpu().numpy()  # (seq_len,)
            follower_pos = follower_pos[0].cpu().numpy()  # (seq_len, 3)
            
            # Check frame 0 values
            print(f"\n  [FRAME 0 - ROOT POSITIONS]")
            print(f"  Leader root position:  [{leader_pos[0, 0]:.6f}, {leader_pos[0, 1]:.6f}, {leader_pos[0, 2]:.6f}]")
            print(f"  Follower root position: [{follower_pos[0, 0]:.6f}, {follower_pos[0, 1]:.6f}, {follower_pos[0, 2]:.6f}]")
            print(f"  Leader root rotation:   {leader_rot_ang[0]:.6f} radians ({np.degrees(leader_rot_ang[0]):.2f} degrees)")
            print(f"  Follower root rotation: {follower_rot_ang[0]:.6f} radians ({np.degrees(follower_rot_ang[0]):.2f} degrees)")
            
            # Check if normalized (should be ~0 for X, Z, rotation)
            print(f"\n  [NORMALIZATION CHECK]")
            leader_xz_norm = np.sqrt(leader_pos[0, 0]**2 + leader_pos[0, 2]**2)
            follower_xz_norm = np.sqrt(follower_pos[0, 0]**2 + follower_pos[0, 2]**2)
            print(f"  Leader XZ distance from origin:   {leader_xz_norm:.6f} (should be ~0 if normalized)")
            print(f"  Follower XZ distance from origin: {follower_xz_norm:.6f} (should be ~0 if normalized)")
            print(f"  Leader rotation magnitude:        {abs(leader_rot_ang[0]):.6f} (should be ~0 if normalized)")
            print(f"  Follower rotation magnitude:      {abs(follower_rot_ang[0]):.6f} (should be ~0 if normalized)")
            
            # Compute relative pose at frame 0
            print(f"\n  [RELATIVE POSE AT FRAME 0]")
            sx = follower_pos[0, 0] - leader_pos[0, 0]
            sz = follower_pos[0, 2] - leader_pos[0, 2]
            sθ = follower_rot_ang[0] - leader_rot_ang[0]
            print(f"  sx (X difference): {sx:.6f}")
            print(f"  sz (Z difference): {sz:.6f}")
            print(f"  sθ (rotation diff): {sθ:.6f} radians ({np.degrees(sθ):.2f} degrees)")
            
            # Check raw HumanML3D values at frame 0
            print(f"\n  [RAW HUMANML3D VALUES AT FRAME 0]")
            print(f"  Leader:")
            print(f"    root_rot_velocity[0]:    {data_L[0, 0]:.6f}")
            print(f"    root_linear_velocity[0]: [{data_L[0, 1]:.6f}, {data_L[0, 2]:.6f}]")
            print(f"    root_y[0]:               {data_L[0, 3]:.6f}")
            print(f"  Follower:")
            print(f"    root_rot_velocity[0]:    {data_F[0, 0]:.6f}")
            print(f"    root_linear_velocity[0]: [{data_F[0, 1]:.6f}, {data_F[0, 2]:.6f}]")
            print(f"    root_y[0]:               {data_F[0, 3]:.6f}")
            
            # Check if original keypoints are available
            if 'keypoints3d_L' in clip and 'keypoints3d_F' in clip:
                kp_L = clip['keypoints3d_L']
                kp_F = clip['keypoints3d_F']
                
                if isinstance(kp_L, torch.Tensor):
                    kp_L = kp_L.cpu().numpy()
                if isinstance(kp_F, torch.Tensor):
                    kp_F = kp_F.cpu().numpy()
                
                print(f"\n  [ORIGINAL KEYPOINTS AT FRAME 0]")
                print(f"  Leader root (pelvis):   [{kp_L[0, 0, 0]:.6f}, {kp_L[0, 0, 1]:.6f}, {kp_L[0, 0, 2]:.6f}]")
                print(f"  Follower root (pelvis):  [{kp_F[0, 0, 0]:.6f}, {kp_F[0, 0, 1]:.6f}, {kp_F[0, 0, 2]:.6f}]")
                
                # Compute relative pose from original keypoints
                orig_sx = kp_F[0, 0, 0] - kp_L[0, 0, 0]
                orig_sz = kp_F[0, 0, 2] - kp_L[0, 0, 2]
                print(f"  Original relative X: {orig_sx:.6f}")
                print(f"  Original relative Z: {orig_sz:.6f}")
                print(f"  Recovered relative X: {sx:.6f}")
                print(f"  Recovered relative Z: {sz:.6f}")
                print(f"  Difference: X={abs(orig_sx - sx):.6f}, Z={abs(orig_sz - sz):.6f}")
            
            # Check a few more frames to see if positions change
            if seq_len > 5:
                print(f"\n  [FRAME 5 - ROOT POSITIONS]")
                print(f"  Leader root position:  [{leader_pos[5, 0]:.6f}, {leader_pos[5, 1]:.6f}, {leader_pos[5, 2]:.6f}]")
                print(f"  Follower root position: [{follower_pos[5, 0]:.6f}, {follower_pos[5, 1]:.6f}, {follower_pos[5, 2]:.6f}]")
                print(f"  Relative X at frame 5: {follower_pos[5, 0] - leader_pos[5, 0]:.6f}")
                print(f"  Relative Z at frame 5: {follower_pos[5, 2] - leader_pos[5, 2]:.6f}")
            
            print()
            
            # Move to next sample
            if not cursor.next():
                break
    
    env.close()
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print("If data is normalized:")
    print("  - Frame 0 root X and Z should be ~0 for both leader and follower")
    print("  - Frame 0 root rotation should be ~0 for both")
    print("  - Relative pose (sx, sz, sθ) at frame 0 should be ~(0, 0, 0)")
    print()
    print("If data maintains relative positioning:")
    print("  - Frame 0 root positions may have non-zero X, Z values")
    print("  - Relative pose (sx, sz, sθ) should show actual spatial relationship")
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    print("Starting script...", flush=True)
    
    parser = argparse.ArgumentParser(description='Check LMDB normalization status')
    parser.add_argument('--lmdb_dir', type=str, 
                       default='dataset_processed_New/lmdb_Salsa_pair/lmdb_train',
                       help='Path to LMDB database directory')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to check')
    
    args = parser.parse_args()
    
    print(f"Arguments: lmdb_dir={args.lmdb_dir}, num_samples={args.num_samples}", flush=True)
    
    try:
        check_lmdb_normalization(args.lmdb_dir, args.num_samples)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()

