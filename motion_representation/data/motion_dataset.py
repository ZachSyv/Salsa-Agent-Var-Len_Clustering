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
        """
        self.args = args
        self.lmdb_dir = lmdb_dir
        self.window_size = window_size
        self.stride = stride
        self.use_both_roles = use_both_roles
        
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
    
    def _create_cache(self):
        """
        Create cache of 20-frame windows directly from raw LMDB.
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
            motion: Motion tensor of shape (window_size=20, 263)
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
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return dataloader

