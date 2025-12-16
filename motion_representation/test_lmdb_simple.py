#!/usr/bin/env python
"""Simple test to check LMDB normalization"""

import os
import sys
import numpy as np
import torch
import lmdb
import pyarrow
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("Importing recovery functions...", flush=True)
from utils.salsa_utils.HumanMl3D_functions import recover_root_rot_pos

lmdb_dir = 'dataset_processed_New/lmdb_Salsa_pair/lmdb_train'
print(f"Opening LMDB: {lmdb_dir}", flush=True)

env = lmdb.open(lmdb_dir, readonly=True, lock=False)
with env.begin() as txn:
    cursor = txn.cursor()
    cursor.first()
    key = cursor.key()
    value = cursor.value()
    
    print(f"Got sample with key: {key}", flush=True)
    video_data = pyarrow.deserialize(value)
    
    if 'clips' in video_data and len(video_data['clips']) > 0:
        clip = video_data['clips'][0]
        
        data_L = clip['HML3D_joints_vec_L']
        data_F = clip['HML3D_joints_vec_F']
        
        if isinstance(data_L, torch.Tensor):
            data_L = data_L.cpu().numpy()
        if isinstance(data_F, torch.Tensor):
            data_F = data_F.cpu().numpy()
        
        print(f"Leader shape: {data_L.shape}, Follower shape: {data_F.shape}", flush=True)
        
        # Recover positions
        data_L_tensor = torch.from_numpy(data_L).float().unsqueeze(0)
        data_F_tensor = torch.from_numpy(data_F).float().unsqueeze(0)
        
        leader_rot, leader_pos = recover_root_rot_pos(data_L_tensor)
        follower_rot, follower_pos = recover_root_rot_pos(data_F_tensor)
        
        leader_pos = leader_pos[0].cpu().numpy()
        follower_pos = follower_pos[0].cpu().numpy()
        leader_rot = leader_rot[0].cpu().numpy()
        follower_rot = follower_rot[0].cpu().numpy()
        
        print(f"\nFRAME 0 RESULTS:", flush=True)
        print(f"Leader pos[0]:  [{leader_pos[0,0]:.6f}, {leader_pos[0,1]:.6f}, {leader_pos[0,2]:.6f}]", flush=True)
        print(f"Follower pos[0]: [{follower_pos[0,0]:.6f}, {follower_pos[0,1]:.6f}, {follower_pos[0,2]:.6f}]", flush=True)
        print(f"Leader rot[0]:   {leader_rot[0]:.6f} rad ({np.degrees(leader_rot[0]):.2f} deg)", flush=True)
        print(f"Follower rot[0]: {follower_rot[0]:.6f} rad ({np.degrees(follower_rot[0]):.2f} deg)", flush=True)
        
        sx = follower_pos[0, 0] - leader_pos[0, 0]
        sz = follower_pos[0, 2] - leader_pos[0, 2]
        sθ = follower_rot[0] - leader_rot[0]
        
        print(f"\nRELATIVE POSE (sx, sz, sθ):", flush=True)
        print(f"  sx = {sx:.6f}", flush=True)
        print(f"  sz = {sz:.6f}", flush=True)
        print(f"  sθ = {sθ:.6f} rad ({np.degrees(sθ):.2f} deg)", flush=True)
        
        print(f"\nNORMALIZATION CHECK:", flush=True)
        print(f"  Leader XZ distance from origin: {np.sqrt(leader_pos[0,0]**2 + leader_pos[0,2]**2):.6f}", flush=True)
        print(f"  Follower XZ distance from origin: {np.sqrt(follower_pos[0,0]**2 + follower_pos[0,2]**2):.6f}", flush=True)
        print(f"  If normalized, both should be ~0", flush=True)

env.close()
