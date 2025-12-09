"""
Comprehensive analysis of HumanML3D data representation.
Analyzes the 263-dimensional motion vectors and provides detailed statistics.
"""
import torch
import numpy as np
import argparse
import sys
import os
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_representation.data.motion_dataset import create_dataloader

# HumanML3D joint names (22 joints total)
HML_JOINT_NAMES = [
    'pelvis',        # 0
    'left_hip',      # 1
    'right_hip',     # 2
    'spine1',        # 3
    'left_knee',     # 4
    'right_knee',    # 5
    'spine2',        # 6
    'left_ankle',    # 7
    'right_ankle',    # 8
    'spine3',        # 9
    'left_foot',     # 10
    'right_foot',    # 11
    'neck',          # 12
    'left_collar',   # 13
    'right_collar',  # 14
    'head',          # 15
    'left_shoulder', # 16
    'right_shoulder',# 17
    'left_elbow',    # 18
    'right_elbow',   # 19
    'left_wrist',    # 20
    'right_wrist',   # 21
]

NUM_HML_JOINTS = 22

# Dimension breakdown for 263-dim HumanML3D representation
DIM_BREAKDOWN = {
    'root_rot_velocity': {'start': 0, 'end': 1, 'dims': 1, 'description': 'Root rotation velocity along y-axis (rad/s)'},
    'root_linear_velocity': {'start': 1, 'end': 3, 'dims': 2, 'description': 'Root linear velocity on xz plane (m/s)'},
    'root_y': {'start': 3, 'end': 4, 'dims': 1, 'description': 'Root height (y-coordinate in meters)'},
    'ric_data': {'start': 4, 'end': 67, 'dims': 63, 'description': 'Rotation-invariant positions for 21 joints (excluding root), each with 3D (x,y,z)'},
    'rot_data': {'start': 67, 'end': 193, 'dims': 126, 'description': '6D rotation representation for 21 joints (excluding root), each with 6D continuous rotation'},
    'local_velocity': {'start': 193, 'end': 259, 'dims': 66, 'description': 'Local velocity for all 22 joints, each with 3D velocity (x,y,z)'},
    'foot_contact': {'start': 259, 'end': 263, 'dims': 4, 'description': 'Foot contact binary flags: [left_toe, left_heel, right_toe, right_heel]'},
}

# Joint indices for ric_data and rot_data (excluding root/pelvis)
RIC_ROT_JOINT_INDICES = list(range(1, NUM_HML_JOINTS))  # Joints 1-21 (excluding pelvis)


def analyze_data(lmdb_dir, batch_size=32, num_batches=None, window_size=20, stride=10):
    """
    Analyze HumanML3D data statistics.
    
    Args:
        lmdb_dir: Path to LMDB dataset directory
        batch_size: Batch size for loading
        num_batches: Number of batches to analyze (None = all)
        window_size: Motion window size
        stride: Stride for window sampling
    """
    print("="*80)
    print("HUMANML3D DATA REPRESENTATION ANALYSIS")
    print("="*80)
    
    # Create minimal args object
    class Args:
        pass
    args = Args()
    
    # Load dataset
    print(f"\nLoading dataset from: {lmdb_dir}")
    dataloader = create_dataloader(
        args=args,
        lmdb_dir=lmdb_dir,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        use_both_roles=True,
    )
    
    dataset = dataloader.dataset
    print(f"Dataset size: {len(dataset)} samples")
    
    # Accumulate statistics
    print(f"\nAnalyzing data...")
    all_data = []
    total_samples = 0
    
    for batch_idx, motion in enumerate(tqdm(dataloader, desc="Processing batches")):
        if num_batches is not None and batch_idx >= num_batches:
            break
        all_data.append(motion.numpy())
        total_samples += motion.shape[0]
    
    # Concatenate all data: (total_samples, seq_len, 263)
    all_data = np.concatenate(all_data, axis=0)
    print(f"Total samples analyzed: {total_samples}")
    print(f"Data shape: {all_data.shape} (samples, seq_len={window_size}, dims=263)")
    
    # Flatten to (total_samples * seq_len, 263) for per-dimension statistics
    data_flat = all_data.reshape(-1, 263)  # (total_samples * seq_len, 263)
    
    # Calculate statistics for each dimension
    print(f"\n" + "="*80)
    print("DIMENSION BREAKDOWN")
    print("="*80)
    
    print(f"\nThe 263-dimensional HumanML3D representation consists of:")
    print(f"  • 22 body joints (SMPL-H skeleton)")
    print(f"  • Multiple motion features per frame\n")
    
    for group_name, info in DIM_BREAKDOWN.items():
        start, end = info['start'], info['end']
        dims = info['dims']
        desc = info['description']
        
        print(f"\n[{group_name.upper()}]")
        print(f"  Dimensions: {start} to {end-1} (total: {dims} dims)")
        print(f"  Description: {desc}")
        
        # Calculate statistics for this group
        group_data = data_flat[:, start:end]
        mean = np.mean(group_data, axis=0)
        std = np.std(group_data, axis=0)
        min_vals = np.min(group_data, axis=0)
        max_vals = np.max(group_data, axis=0)
        
        print(f"  Statistics:")
        print(f"    Mean: {mean}")
        print(f"    Std:  {std}")
        print(f"    Min:  {min_vals}")
        print(f"    Max:  {max_vals}")
        
        # Special interpretation for specific groups
        if group_name == 'root_rot_velocity':
            print(f"  Interpretation:")
            print(f"    • Rotation velocity around vertical (y) axis")
            print(f"    • Values typically in range [-π, π] radians/second")
            print(f"    • Mean near 0 indicates balanced rotation")
        
        elif group_name == 'root_linear_velocity':
            print(f"  Interpretation:")
            print(f"    • Linear velocity on horizontal (xz) plane")
            print(f"    • [x_velocity, z_velocity] in meters/second")
            print(f"    • Mean near 0 indicates balanced movement")
        
        elif group_name == 'root_y':
            print(f"  Interpretation:")
            print(f"    • Root/pelvis height above ground")
            print(f"    • Typically around 0.9-1.0 meters for standing")
            print(f"    • Lower values indicate crouching/sitting")
        
        elif group_name == 'ric_data':
            print(f"  Interpretation:")
            print(f"    • Local joint positions relative to parent (rotation-invariant)")
            print(f"    • Organized as: [joint_1_x, joint_1_y, joint_1_z, joint_2_x, ...]")
            print(f"    • Joints 1-21 (excluding pelvis/root)")
            print(f"    • Values represent bone lengths and relative positions")
            for i, joint_idx in enumerate(RIC_ROT_JOINT_INDICES[:5]):  # Show first 5
                joint_name = HML_JOINT_NAMES[joint_idx]
                ric_start = i * 3
                ric_end = (i + 1) * 3
                print(f"      Joint {joint_idx} ({joint_name}): dims {start+ric_start}-{start+ric_end-1}, "
                      f"mean={mean[ric_start:ric_end]}, std={std[ric_start:ric_end]}")
            print(f"      ... (showing first 5 of 21 joints)")
        
        elif group_name == 'rot_data':
            print(f"  Interpretation:")
            print(f"    • 6D continuous rotation representation (Zhou et al.)")
            print(f"    • Organized as: [joint_1_rot_6d, joint_2_rot_6d, ...]")
            print(f"    • Each joint has 6 dimensions (first 2 columns of rotation matrix)")
            print(f"    • Joints 1-21 (excluding pelvis/root)")
            for i, joint_idx in enumerate(RIC_ROT_JOINT_INDICES[:5]):  # Show first 5
                joint_name = HML_JOINT_NAMES[joint_idx]
                rot_start = i * 6
                rot_end = (i + 1) * 6
                print(f"      Joint {joint_idx} ({joint_name}): dims {start+rot_start}-{start+rot_end-1}, "
                      f"mean={mean[rot_start:rot_end]}, std={std[rot_start:rot_end]}")
            print(f"      ... (showing first 5 of 21 joints)")
        
        elif group_name == 'local_velocity':
            print(f"  Interpretation:")
            print(f"    • Local velocity for all 22 joints (including root)")
            print(f"    • Organized as: [joint_0_vel_xyz, joint_1_vel_xyz, ...]")
            print(f"    • Each joint has 3D velocity (x, y, z)")
            print(f"    • Values in meters/second")
            for i in range(min(5, NUM_HML_JOINTS)):  # Show first 5
                joint_name = HML_JOINT_NAMES[i]
                vel_start = i * 3
                vel_end = (i + 1) * 3
                print(f"      Joint {i} ({joint_name}): dims {start+vel_start}-{start+vel_end-1}, "
                      f"mean={mean[vel_start:vel_end]}, std={std[vel_start:vel_end]}")
            print(f"      ... (showing first 5 of 22 joints)")
        
        elif group_name == 'foot_contact':
            print(f"  Interpretation:")
            print(f"    • Binary foot contact flags: [left_toe, left_heel, right_toe, right_heel]")
            print(f"    • Values typically 0 (no contact) or 1 (contact)")
            print(f"    • Mean indicates contact frequency")
            contact_labels = ['left_toe', 'left_heel', 'right_toe', 'right_heel']
            for i, label in enumerate(contact_labels):
                print(f"      {label}: dim {start+i}, mean={mean[i]:.4f}, std={std[i]:.4f}")
    
    # Overall statistics
    print(f"\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    overall_mean = np.mean(data_flat, axis=0)
    overall_std = np.std(data_flat, axis=0)
    overall_min = np.min(data_flat, axis=0)
    overall_max = np.max(data_flat, axis=0)
    
    print(f"\nPer-dimension statistics (all 263 dimensions):")
    print(f"  Mean range: [{overall_mean.min():.6f}, {overall_mean.max():.6f}]")
    print(f"  Std range:  [{overall_std.min():.6f}, {overall_std.max():.6f}]")
    print(f"  Min range:  [{overall_min.min():.6f}, {overall_min.max():.6f}]")
    print(f"  Max range:  [{overall_max.min():.6f}, {overall_max.max():.6f}]")
    
    # Check for potential issues
    print(f"\n" + "="*80)
    print("DATA QUALITY CHECKS")
    print("="*80)
    
    has_nan = np.isnan(data_flat).any()
    has_inf = np.isinf(data_flat).any()
    extreme_values = np.abs(data_flat) > 1e6
    
    print(f"\nCorruption checks:")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  Extreme values (>1e6): {extreme_values.sum()} occurrences")
    
    # Check for constant dimensions
    constant_dims = []
    for dim in range(263):
        if overall_std[dim] < 1e-8:
            constant_dims.append(dim)
    
    if constant_dims:
        print(f"  Constant dimensions (std < 1e-8): {len(constant_dims)} dims: {constant_dims}")
    else:
        print(f"  ✓ No constant dimensions found")
    
    # Summary table
    print(f"\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"\n{'Component':<25} {'Dims':<10} {'Range':<15} {'Mean (sample)':<20} {'Std (sample)':<20}")
    print("-" * 90)
    
    for group_name, info in DIM_BREAKDOWN.items():
        start, end = info['start'], info['end']
        group_data = data_flat[:, start:end]
        mean_sample = np.mean(group_data)
        std_sample = np.std(group_data)
        min_val = np.min(group_data)
        max_val = np.max(group_data)
        print(f"{group_name:<25} {start}-{end-1:<10} [{min_val:>7.4f}, {max_val:>7.4f}] {mean_sample:>19.6f} {std_sample:>19.6f}")
    
    print(f"\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze HumanML3D data representation')
    parser.add_argument('--lmdb_dir', type=str, required=True, help='Path to LMDB dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for analysis')
    parser.add_argument('--num_batches', type=int, default=None, help='Number of batches to analyze (None = all)')
    parser.add_argument('--window_size', type=int, default=20, help='Motion window size')
    parser.add_argument('--stride', type=int, default=10, help='Stride for window sampling')
    
    args = parser.parse_args()
    
    analyze_data(
        args.lmdb_dir,
        args.batch_size,
        args.num_batches,
        args.window_size,
        args.stride
    )

