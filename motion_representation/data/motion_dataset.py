"""
Data loader for 20-frame window sampling from Salsa dance pairs dataset.
Follows the original DataPreprocessor pipeline to extract windows directly from raw LMDB.
"""

import os
import sys
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import lmdb
import pyarrow
import pickle
from typing import Tuple, Optional
from tqdm import tqdm

# Add parent directory to path to import existing utilities
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class MotionWindowDataset(Dataset):
    """
    Dataset for 20-frame motion windows from Salsa dance pairs.
    Loads both leader and follower motions and treats them equally (joint training).
    """
    
    def __init__(
        self,
        args,
        lmdb_dir: str,
        window_size: int = 20,
        stride: int = 10,
        use_both_roles: bool = True,
        cache_dir: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            args: Arguments object (from existing codebase)
            lmdb_dir: Path to LMDB database directory
            window_size: Size of motion window (default: 20 frames)
            stride: Stride for window sampling (default: 10 frames)
            use_both_roles: If True, use both leader and follower motions (joint training)
            cache_dir: Optional cache directory for preprocessed windows
            normalize: If True, normalize data using precomputed mean/std (default: True)
        """
        self.args = args
        self.lmdb_dir = lmdb_dir
        self.window_size = window_size
        self.stride = stride
        self.use_both_roles = use_both_roles
        self.normalize = normalize
        
        # Determine cache directory - use descriptive name for VAE 20-frame windows
        if cache_dir is None:
            cache_dir = lmdb_dir + f"_VAE_20frames_cache"
        self.cache_dir = cache_dir
        
        # Check if cache exists, if not create it
        if not os.path.exists(self.cache_dir):
            print(f"Cache not found. Creating cache at {self.cache_dir}...")
            self._create_cache()
        else:
            print(f"Loading cached windows from {self.cache_dir}...")
        
        # Load cached windows
        self.lmdb_env = lmdb.open(self.cache_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]
        
        print(f"Loaded {self.n_samples} motion windows from cache.")
        
        # Load normalization statistics
        self.mean = None
        self.std = None
        self.epsilon = 1e-8  # Small value to avoid division by zero
        
        if self.normalize:
            stats_path = os.path.join(self.cache_dir, 'normalization_stats.pkl')
            if os.path.exists(stats_path):
                print(f"Loading normalization statistics from {stats_path}...")
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)
                    self.mean = torch.from_numpy(stats['mean']).float()
                    self.std = torch.from_numpy(stats['std']).float()
                    print(f"  Mean shape: {self.mean.shape}, Std shape: {self.std.shape}")
                    print(f"  Mean range: [{self.mean.min():.6f}, {self.mean.max():.6f}]")
                    print(f"  Std range: [{self.std.min():.6f}, {self.std.max():.6f}]")
                    # Check for constant dimensions
                    constant_dims = (self.std < self.epsilon).sum().item()
                    if constant_dims > 0:
                        print(f"  Warning: {constant_dims} dimensions have std < {self.epsilon} (will not be normalized)")
            else:
                print(f"Warning: Normalization requested but stats file not found at {stats_path}")
                print(f"  Set normalize=False or recreate cache to compute statistics.")
    
    def _create_cache(self):
        """
        Create cache of 20-frame windows directly from raw LMDB.
        Also computes normalization statistics from all original frames (not windows).
        Follows the same pipeline as DataPreprocessor._sample_from_clip_pair.
        """
        # Open raw LMDB (same as DataPreprocessor does)
        src_lmdb_env = lmdb.open(self.lmdb_dir, readonly=True, lock=False)
        
        # Create cache LMDB
        map_size = 1024 * 30  # 30 GB
        map_size <<= 20  # Convert to bytes
        cache_env = lmdb.open(self.cache_dir, map_size=map_size)
        
        window_idx = 0
        skeleton_resampling_fps = 20  # Same as original pipeline
        
        print("Creating 20-frame window cache from raw LMDB...")
        print("Following original DataPreprocessor pipeline...")
        
        # Collect all frames for normalization statistics (not windows, to avoid double-counting)
        all_frames = []  # Will collect all individual frames from raw data
        
        # Follow the same structure as DataPreprocessor.run()
        src_txn = src_lmdb_env.begin(write=False)
        total_count = src_txn.stat()['entries']
        
        cursor = src_txn.cursor()
        counter = 0
        
        for key, value in tqdm(cursor, total=total_count, desc="Processing videos"):
            try:
                video = pyarrow.deserialize(value)
                vid = video['vid']
                clips = video['clips']
                
                for clip_idx, clip in enumerate(clips):
                    # Extract windows from this clip (following _sample_from_clip_pair logic)
                    windows_from_clip = self._extract_windows_from_clip(clip)
                    
                    # Save all windows from this clip
                    for window in windows_from_clip:
                        with cache_env.begin(write=True) as txn:
                            key = "{:010}".format(window_idx).encode("ascii")
                            value = pyarrow.serialize(window).to_buffer()
                            txn.put(key, value)
                            window_idx += 1
                    
                    # Collect all frames from this clip for normalization stats
                    # Extract raw motion data (not windows) to avoid double-counting overlapping frames
                    clip_HM3D_joint_vec_L = clip['HML3D_joints_vec_L']
                    clip_HM3D_joint_vec_F = clip['HML3D_joints_vec_F']
                    
                    # Convert to numpy if needed
                    if isinstance(clip_HM3D_joint_vec_L, torch.Tensor):
                        clip_HM3D_joint_vec_L = clip_HM3D_joint_vec_L.cpu().numpy()
                    if isinstance(clip_HM3D_joint_vec_F, torch.Tensor):
                        clip_HM3D_joint_vec_F = clip_HM3D_joint_vec_F.cpu().numpy()
                    
                    # Add all frames from both roles (if enabled)
                    if self.use_both_roles:
                        all_frames.append(clip_HM3D_joint_vec_L)  # (seq_len, 263)
                        all_frames.append(clip_HM3D_joint_vec_F)  # (seq_len, 263)
                    else:
                        all_frames.append(clip_HM3D_joint_vec_L)  # (seq_len, 263)
                    
                    counter += 1
                    
                    if counter % 10 == 0:
                        print(f"Processed {counter} clips, created {window_idx} windows...")
            
            except Exception as e:
                print(f"Error processing video {counter}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        cache_env.sync()
        cache_env.close()
        src_lmdb_env.close()
        print(f"Cache creation complete. Total windows: {window_idx}")
        
        # Compute normalization statistics from all original frames
        print(f"\nComputing normalization statistics from all original frames...")
        print(f"  Collected {len(all_frames)} motion sequences")
        
        # Concatenate all frames: (total_frames, 263)
        all_frames_array = np.concatenate(all_frames, axis=0)
        total_frames = all_frames_array.shape[0]
        print(f"  Total frames: {total_frames}")
        print(f"  Frame shape: {all_frames_array.shape[1]} (should be 263)")
        
        # Calculate mean and std per dimension
        mean = np.mean(all_frames_array, axis=0)  # (263,)
        std = np.std(all_frames_array, axis=0)   # (263,)
        
        print(f"  Mean shape: {mean.shape}, Std shape: {std.shape}")
        print(f"  Mean range: [{mean.min():.6f}, {mean.max():.6f}]")
        print(f"  Std range: [{std.min():.6f}, {std.max():.6f}]")
        
        # Check for constant dimensions
        epsilon = 1e-8
        constant_dims = (std < epsilon).sum()
        if constant_dims > 0:
            print(f"  Warning: {constant_dims} dimensions have std < {epsilon}")
            print(f"    These dimensions will not be normalized (std will be set to 1.0)")
            # Set std to 1.0 for constant dimensions to avoid division by zero
            std[std < epsilon] = 1.0
        
        # Save normalization statistics
        stats_path = os.path.join(self.cache_dir, 'normalization_stats.pkl')
        stats = {
            'mean': mean,
            'std': std,
            'total_frames': total_frames,
            'total_sequences': len(all_frames),
            'window_size': self.window_size,
            'stride': self.stride,
            'use_both_roles': self.use_both_roles,
        }
        
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        
        print(f"  Saved normalization statistics to: {stats_path}")
        print(f"  Statistics computed from {total_frames} frames across {len(all_frames)} sequences")
    
    def _extract_windows_from_clip(self, clip: dict):
        """
        Extract 20-frame windows from a clip, following DataPreprocessor._sample_from_clip_pair.
        
        Args:
            clip: Clip dictionary from raw LMDB (same format as in DataPreprocessor)
        
        Returns:
            List of 20-frame windows (each is numpy array of shape (20, 263))
        """
        windows = []
        
        # Extract HML3D data from clip (same as _sample_from_clip_pair)
        clip_HM3D_joint_vec_L = clip['HML3D_joints_vec_L']
        clip_HM3D_joint_vec_F = clip['HML3D_joints_vec_F']
        
        # Convert to numpy if needed
        if isinstance(clip_HM3D_joint_vec_L, torch.Tensor):
            clip_HM3D_joint_vec_L = clip_HM3D_joint_vec_L.cpu().numpy()
        if isinstance(clip_HM3D_joint_vec_F, torch.Tensor):
            clip_HM3D_joint_vec_F = clip_HM3D_joint_vec_F.cpu().numpy()
        
        # Process motions (same logic as original, but extract 20-frame windows)
        motions_to_process = []
        if self.use_both_roles:
            motions_to_process = [
                ('L', clip_HM3D_joint_vec_L),
                ('F', clip_HM3D_joint_vec_F)
            ]
        else:
            motions_to_process = [('L', clip_HM3D_joint_vec_L)]
        
        for role, motion in motions_to_process:
            seq_len = motion.shape[0]
            
            # Skip if sequence is too short
            if seq_len < self.window_size:
                continue
            
            # Create overlapping windows (same logic as original subdivision)
            # num_subdivision = floor((seq_len - window_size) / stride) + 1
            num_windows = math.floor((seq_len - self.window_size) / self.stride) + 1
            
            for i in range(num_windows):
                start_idx = i * self.stride
                fin_idx = start_idx + self.window_size
                
                # Check bounds (same as original)
                if fin_idx > seq_len:
                    continue
                
                window = motion[start_idx:fin_idx]  # (window_size, 263)
                
                # Ensure exactly window_size frames
                if window.shape[0] != self.window_size:
                    continue
                
                windows.append(window)
        
        return windows
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a motion window at the specified index.
        
        Args:
            idx: Index of the sample
        
        Returns:
            motion: Motion tensor of shape (window_size=20, 263), normalized if normalize=True
        """
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:010}".format(idx).encode("ascii")
            sample = txn.get(key)
            
            if sample is None:
                raise IndexError(f"Index {idx} out of range")
            
            motion = pyarrow.deserialize(sample)
            # Copy array to make it writable and avoid warnings
            motion = torch.from_numpy(motion.copy()).float()
            
            # Ensure correct shape: (window_size, 263)
            assert motion.shape == (self.window_size, 263), \
                f"Expected shape ({self.window_size}, 263), got {motion.shape}"
            
            # Apply normalization if enabled and stats are available
            if self.normalize and self.mean is not None and self.std is not None:
                # Normalize: (motion - mean) / (std + epsilon)
                # mean and std are shape (263,), will broadcast to (window_size, 263)
                motion = (motion - self.mean) / (self.std + self.epsilon)
            
            return motion


def create_dataloader(
    args,
    lmdb_dir: str,
    window_size: int = 20,
    stride: int = 10,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    use_both_roles: bool = True,
    normalize: bool = True,
):
    """
    Create a DataLoader for motion windows.
    
    Args:
        args: Arguments object
        lmdb_dir: Path to LMDB database
        window_size: Size of motion window (default: 20)
        stride: Stride for window sampling (default: 10)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes
        use_both_roles: Whether to use both leader and follower
        normalize: Whether to normalize data using precomputed mean/std (default: True)
    
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    dataset = MotionWindowDataset(
        args=args,
        lmdb_dir=lmdb_dir,
        window_size=window_size,
        stride=stride,
        use_both_roles=use_both_roles,
        normalize=normalize,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return dataloader

