#!/usr/bin/env python
"""Test script to verify relationship features extraction from keypoints3d"""

import os
import sys
import numpy as np
import torch
import lmdb
import pyarrow
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from motion_representation.utils.relationship_features import extract_relationship_features_from_keypoints3d

# Test with sample data
print("Testing extract_relationship_features_from_keypoints3d...")

# Create synthetic test data: (seq_len=10, 22 joints, 3D)
seq_len = 10
num_joints = 22

# Leader: starts at origin (0, 0, 0), moves forward in +Z direction
leader_keypoints3d = np.zeros((seq_len, num_joints, 3))
leader_keypoints3d[:, 0, :] = np.array([0, 0, 0])  # Root at origin
leader_keypoints3d[:, 0, 2] = np.linspace(0, 1, seq_len)  # Move forward in Z
leader_keypoints3d[:, 12, :] = leader_keypoints3d[:, 0, :] + np.array([0, 0.2, 0])  # Neck above root

# Follower: starts at (1, 0, 0) - to the right of leader, also moves forward
follower_keypoints3d = np.zeros((seq_len, num_joints, 3))
follower_keypoints3d[:, 0, :] = np.array([1, 0, 0])  # Root at (1, 0, 0)
follower_keypoints3d[:, 0, 2] = np.linspace(0, 1, seq_len)  # Move forward in Z
follower_keypoints3d[:, 12, :] = follower_keypoints3d[:, 0, :] + np.array([0, 0.2, 0])  # Neck above root

print(f"Leader root positions (first 3 frames):")
print(leader_keypoints3d[:3, 0, :])
print(f"\nFollower root positions (first 3 frames):")
print(follower_keypoints3d[:3, 0, :])

# Extract relationship features
relationship_features = extract_relationship_features_from_keypoints3d(
    leader_keypoints3d, follower_keypoints3d
)

print(f"\nRelationship features shape: {relationship_features.shape}")
print(f"Expected: (seq_len={seq_len}, 4)")
print(f"\nRelationship features (first 3 frames):")
print(relationship_features[:3, :])
print(f"\nFeature interpretation:")
print(f"  [x_diff, z_diff, y_diff, rotation_diff]")
print(f"  x_diff: follower_x - leader_x (should be ~1.0)")
print(f"  z_diff: follower_z - leader_z (should be ~0.0)")
print(f"  y_diff: follower_y - leader_y (should be ~0.0)")
print(f"  rotation_diff: follower_yaw - leader_yaw (should be ~0.0)")

# Verify expected values
x_diff = relationship_features[:, 0]
z_diff = relationship_features[:, 1]
y_diff = relationship_features[:, 2]
rotation_diff = relationship_features[:, 3]

print(f"\nVerification:")
print(f"  x_diff mean: {x_diff.mean():.6f} (expected ~1.0)")
print(f"  z_diff mean: {z_diff.mean():.6f} (expected ~0.0)")
print(f"  y_diff mean: {y_diff.mean():.6f} (expected ~0.0)")
print(f"  rotation_diff mean: {rotation_diff.mean():.6f} (expected ~0.0)")

# Test with real LMDB data if available
lmdb_dir = 'dataset_processed_New/lmdb_Salsa_pair/lmdb_train'
if os.path.exists(lmdb_dir):
    print(f"\n\nTesting with real LMDB data from {lmdb_dir}...")
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        key = cursor.key()
        value = cursor.value()
        
        video_data = pyarrow.deserialize(value)
        
        if 'clips' in video_data and len(video_data['clips']) > 0:
            clip = video_data['clips'][0]
            
            if 'keypoints3d_L' in clip and 'keypoints3d_F' in clip:
                keypoints3d_L = clip['keypoints3d_L']
                keypoints3d_F = clip['keypoints3d_F']
                
                if isinstance(keypoints3d_L, torch.Tensor):
                    keypoints3d_L = keypoints3d_L.cpu().numpy()
                if isinstance(keypoints3d_F, torch.Tensor):
                    keypoints3d_F = keypoints3d_F.cpu().numpy()
                
                print(f"Loaded keypoints3d_L shape: {keypoints3d_L.shape}")
                print(f"Loaded keypoints3d_F shape: {keypoints3d_F.shape}")
                
                # Extract relationship features
                relationship_features = extract_relationship_features_from_keypoints3d(
                    keypoints3d_L, keypoints3d_F
                )
                
                print(f"Relationship features shape: {relationship_features.shape}")
                print(f"First frame features: {relationship_features[0, :]}")
                print(f"Feature ranges:")
                print(f"  x_diff: [{relationship_features[:, 0].min():.3f}, {relationship_features[:, 0].max():.3f}]")
                print(f"  z_diff: [{relationship_features[:, 1].min():.3f}, {relationship_features[:, 1].max():.3f}]")
                print(f"  y_diff: [{relationship_features[:, 2].min():.3f}, {relationship_features[:, 2].max():.3f}]")
                print(f"  rotation_diff: [{relationship_features[:, 3].min():.3f}, {relationship_features[:, 3].max():.3f}]")
            else:
                print("  Warning: keypoints3d_L or keypoints3d_F not found in clip")
        else:
            print("  Warning: No clips found in video data")
    
    env.close()
else:
    print(f"\nLMDB directory not found: {lmdb_dir}")
    print("  Skipping real data test")

print("\nTest complete!")

