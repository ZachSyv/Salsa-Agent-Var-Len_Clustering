"""Utility functions for visualizing skeleton data, motion tokens, and audio from LMDB files."""
import os
import sys
import numpy as np
import torch
import lmdb
import pyarrow
from pathlib import Path
from typing import Tuple, Dict, Optional
import tempfile
import subprocess
from glob import glob

# Global debug flag
DEBUG = True

def debug_print(*args, **kwargs):
    """Print only if DEBUG is True."""
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain


class LMDBLoader:
    """Simple LMDB loader for visualization purposes."""
    
    def __init__(self, lmdb_dir: str, is_MDM: bool = False):
        """
        Initialize LMDB loader.
        
        Args:
            lmdb_dir: Path to LMDB directory (will auto-detect cache directory like Salsa_Dataset)
            is_MDM: Whether the data is in MDM format
        """
        self.is_MDM = is_MDM
        
        # Auto-detect cache directory like Salsa_Dataset does
        preloaded_dir = lmdb_dir + "_cache"
        if self.is_MDM:
            preloaded_dir += '_MDM'
        
        # Check if cache exists, if not use original directory
        if os.path.exists(preloaded_dir):
            debug_print(f"Found cached/processed LMDB at: {preloaded_dir}")
            self.lmdb_dir = preloaded_dir
        else:
            debug_print(f"Cache not found at {preloaded_dir}, using original directory: {lmdb_dir}")
            debug_print("Note: Original directory contains raw video data. You may need to process it first.")
            self.lmdb_dir = lmdb_dir
        
        self.lmdb_env = lmdb.open(self.lmdb_dir, readonly=True, lock=False)
        
        # Get total entries from LMDB (same as Salsa_Dataset)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']
        
        # Don't pre-validate - just use n_samples - 1 like Salsa_Dataset does
        # We'll validate on-demand when loading samples
        # Cache of valid indices we've discovered
        self.valid_indices_cache = {}
        debug_print(f"LMDB loaded from {self.lmdb_dir} with {self.n_samples} total entries (using {self.n_samples - 1} as max index)")
    
    def __len__(self):
        # Same as Salsa_Dataset: return n_samples - 1 (last item is None)
        return self.n_samples - 1
    
    def _is_valid_sample(self, sample) -> bool:
        """Check if a deserialized sample has the expected structure."""
        try:
            if sample is None:
                debug_print("Sample is None")
                return False
            
            sample_len = len(sample)
            debug_print(f"Sample length: {sample_len}, expected: {12 if self.is_MDM else 10}")
            
            # Try to unpack based on format
            if self.is_MDM:
                # MDM format: 13 elements (added audio_raw)
                if sample_len != 13:
                    debug_print(f"MDM format: expected 13 elements, got {sample_len}")
                    return False
                poses_keypoints3d_L, poses_rotmat_L, HML3D_L, ms_desc_L, vq_tokens_L, \
                    poses_keypoints3d_F, poses_rotmat_F, HML3D_F, ms_des_F, vq_tokens_F, \
                    audio_tokens, audio_raw, aux_info = sample
            else:
                # Non-MDM format: 11 elements (added audio_raw)
                if sample_len != 11:
                    debug_print(f"Non-MDM format: expected 11 elements, got {sample_len}")
                    return False
                poses_keypoints3d_L, poses_rotmat_L, ms_desc_L, vq_tokens_L, \
                 poses_keypoints3d_F, poses_rotmat_F, ms_des_F, vq_tokens_F, \
                 audio_tokens, audio_raw, aux_info = sample
            
            # Check that aux_info is a dict (basic validation)
            if not isinstance(aux_info, dict):
                debug_print(f"aux_info is not a dict, got {type(aux_info)}")
                return False
            
            debug_print("Sample validation passed")
            return True
        except Exception as e:
            debug_print(f"Exception during sample validation: {e}")
            return False
    
    def get_sample(self, idx: int) -> Dict:
        """
        Get a sample from the LMDB.
        
        Args:
            idx: Sample index (0-based, same as Salsa_Dataset)
            
        Returns:
            Dictionary containing sample data
        """
        debug_print(f"get_sample called with idx={idx}, n_samples={self.n_samples}, is_MDM={self.is_MDM}")
        
        if idx < 0 or idx >= (self.n_samples - 1):
            raise IndexError(f"Index {idx} out of range (0-{self.n_samples - 2})")
        
        # Use sequential keys like Salsa_Dataset does
        key = "{:010}".format(idx).encode("ascii")
        debug_print(f"Looking for key: {key.decode()}")
        
        with self.lmdb_env.begin(write=False) as txn:
            sample = txn.get(key)
            if sample is None:
                debug_print(f"Key {key.decode()} not found in LMDB")
                raise KeyError(f"Key {key.decode()} not found in LMDB")
            
            debug_print(f"Found key, deserializing...")
            sample = pyarrow.deserialize(sample)
            debug_print(f"Deserialized sample type: {type(sample)}, length: {len(sample) if hasattr(sample, '__len__') else 'N/A'}")
            
            # Check if sample is already a dict (some LMDBs store as dict)
            if isinstance(sample, dict):
                debug_print(f"Sample is already a dict with keys: {list(sample.keys())}")
                # Check if it has the expected structure
                if 'poses_keypoints3d_L' in sample or 'keypoints3d_L' in sample:
                    debug_print("Dict has expected keys, returning as-is")
                    # Ensure all expected keys exist with None defaults
                    result = {
                        'poses_keypoints3d_L': sample.get('poses_keypoints3d_L') or sample.get('keypoints3d_L'),
                        'poses_rotmat_L': sample.get('poses_rotmat_L') or sample.get('rotmat_L'),
                        'poses_keypoints3d_F': sample.get('poses_keypoints3d_F') or sample.get('keypoints3d_F'),
                        'poses_rotmat_F': sample.get('poses_rotmat_F') or sample.get('rotmat_F'),
                        'ms_desc_L': sample.get('ms_desc_L') or sample.get('ms_description_L'),
                        'ms_des_F': sample.get('ms_des_F') or sample.get('ms_description_F'),
                        'vq_tokens_L': sample.get('vq_tokens_L'),
                        'vq_tokens_F': sample.get('vq_tokens_F'),
                        'audio_tokens': sample.get('audio_tokens'),
                        'audio_raw': sample.get('audio_raw'),
                        'aux_info': sample.get('aux_info') or sample.get('aux'),
                        'HML3D_L': sample.get('HML3D_L') or sample.get('HML3D_vec_L'),
                        'HML3D_F': sample.get('HML3D_F') or sample.get('HML3D_vec_F'),
                    }
                    return result
                else:
                    # Dict with unexpected structure - print keys for debugging
                    debug_print(f"Dict has unexpected keys: {list(sample.keys())}")
                    # Try to see if it's a list/tuple inside
                    if len(sample) == 2:
                        debug_print(f"Dict has 2 items, checking values...")
                        for k, v in sample.items():
                            debug_print(f"  {k}: type={type(v)}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")
                    raise ValueError(f"Sample at index {idx} is a dict but doesn't have expected keys. Keys: {list(sample.keys())}")
            
            # Validate sample structure (for tuple/list format)
            if not self._is_valid_sample(sample):
                raise ValueError(f"Sample at index {idx} has invalid structure (got {len(sample) if sample else 0} elements, expected {12 if self.is_MDM else 10})")
            
            debug_print("Sample validation passed, unpacking...")
        
        if not self.is_MDM:
            poses_keypoints3d_L, poses_rotmat_L, ms_desc_L, vq_tokens_L, \
             poses_keypoints3d_F, poses_rotmat_F, ms_des_F, vq_tokens_F, \
             audio_tokens, audio_raw, aux_info = sample
            HML3D_L = None
            HML3D_F = None
        else:
            poses_keypoints3d_L, poses_rotmat_L, HML3D_L, ms_desc_L, vq_tokens_L, \
                poses_keypoints3d_F, poses_rotmat_F, HML3D_F, ms_des_F, vq_tokens_F, \
                audio_tokens, audio_raw, aux_info = sample
        
        return {
            'poses_keypoints3d_L': poses_keypoints3d_L,
            'poses_rotmat_L': poses_rotmat_L,
            'poses_keypoints3d_F': poses_keypoints3d_F,
            'poses_rotmat_F': poses_rotmat_F,
            'ms_desc_L': ms_desc_L,
            'ms_des_F': ms_des_F,
            'vq_tokens_L': vq_tokens_L,
            'vq_tokens_F': vq_tokens_F,
            'audio_tokens': audio_tokens,
            'audio_raw': audio_raw,
            'aux_info': aux_info,
            'HML3D_L': HML3D_L,
            'HML3D_F': HML3D_F,
        }
    
    def close(self):
        """Close the LMDB environment."""
        if self.lmdb_env:
            self.lmdb_env.close()


def wrap_title(title: str, max_length: int = 30) -> str:
    """
    Wrap long titles to fit in figure.
    
    Args:
        title: Title string
        max_length: Maximum characters per line
        
    Returns:
        Wrapped title string
    """
    if len(title) <= max_length:
        return title
    
    words = title.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_len = len(word)
        if current_length + word_len + 1 <= max_length:
            current_line.append(word)
            current_length += word_len + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_len
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)


def render_combined_skeletons(
    leader_keypoints: np.ndarray,
    follower_keypoints: np.ndarray,
    output_path: str,
    title: str = "Dancing Together",
    fps: int = 20,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 100,
    leader_offset: Tuple[float, float, float] = (0, 0, 0),
    follower_offset: Tuple[float, float, float] = (0, 0, 0)
) -> str:
    """
    Render both leader and follower skeletons together in the same animation.
    
    Uses matplotlib's FuncAnimation with 3D plotting to create a combined skeleton animation.
    
    Args:
        leader_keypoints: Array of shape (n_frames, n_joints, 3) for leader
        follower_keypoints: Array of shape (n_frames, n_joints, 3) for follower
        output_path: Path to save the video
        title: Title for the animation (will be wrapped if too long)
        fps: Frames per second
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for output
        leader_offset: (x, y, z) offset for leader position
        follower_offset: (x, y, z) offset for follower position
        
    Returns:
        Path to the saved video file
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Ensure numpy arrays
        if isinstance(leader_keypoints, torch.Tensor):
            leader_keypoints = leader_keypoints.detach().cpu().numpy()
        if isinstance(follower_keypoints, torch.Tensor):
            follower_keypoints = follower_keypoints.detach().cpu().numpy()
        
        # Reshape if needed
        if len(leader_keypoints.shape) == 2:
            n_frames = leader_keypoints.shape[0]
            n_joints = leader_keypoints.shape[1] // 3
            leader_keypoints = leader_keypoints.reshape(n_frames, n_joints, 3)
        if len(follower_keypoints.shape) == 2:
            n_frames = follower_keypoints.shape[0]
            n_joints = follower_keypoints.shape[1] // 3
            follower_keypoints = follower_keypoints.reshape(n_frames, n_joints, 3)
        
        # Ensure same number of frames
        min_frames = min(leader_keypoints.shape[0], follower_keypoints.shape[0])
        leader_keypoints = leader_keypoints[:min_frames]
        follower_keypoints = follower_keypoints[:min_frames]
        
        debug_print(f"Rendering combined skeletons: {min_frames} frames")
        debug_print(f"Leader: {leader_keypoints.shape[1]} joints, Follower: {follower_keypoints.shape[1]} joints")
        
        # Create output directory
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # Wrap title
        wrapped_title = wrap_title(title, max_length=40)
        
        # Prepare data
        leader_data = leader_keypoints.copy().reshape(len(leader_keypoints), -1, 3)
        follower_data = follower_keypoints.copy().reshape(len(follower_keypoints), -1, 3)
        
        # Apply offsets
        leader_data = leader_data.copy()
        follower_data = follower_data.copy()
        leader_data[:, :, 0] += leader_offset[0]
        leader_data[:, :, 1] += leader_offset[1]
        leader_data[:, :, 2] += leader_offset[2]
        follower_data[:, :, 0] += follower_offset[0]
        follower_data[:, :, 1] += follower_offset[1]
        follower_data[:, :, 2] += follower_offset[2]
        
        # Normalize heights (put both on ground)
        leader_height_offset = leader_data[:, :, 1].min()
        follower_height_offset = follower_data[:, :, 1].min()
        leader_data[:, :, 1] -= leader_height_offset
        follower_data[:, :, 1] -= follower_height_offset
        
        # Center both around a common origin (center between their root positions)
        # First, get the center point between leader and follower roots
        leader_root = leader_data[:, 0:1, :]  # (n_frames, 1, 3)
        follower_root = follower_data[:, 0:1, :]  # (n_frames, 1, 3)
        center_root = (leader_root + follower_root) / 2  # (n_frames, 1, 3)
        
        # Center both around this common center
        leader_data[:, :, 0] -= center_root[:, :, 0]
        leader_data[:, :, 2] -= center_root[:, :, 2]
        follower_data[:, :, 0] -= center_root[:, :, 0]
        follower_data[:, :, 2] -= center_root[:, :, 2]
        
        # Calculate combined bounds from actual data
        all_data = np.concatenate([leader_data, follower_data], axis=1)
        MINS = all_data.min(axis=0).min(axis=0)
        MAXS = all_data.max(axis=0).max(axis=0)
        
        # Calculate dynamic limits with minimal padding (10% margin for tight framing)
        x_range = MAXS[0] - MINS[0]
        z_range = MAXS[2] - MINS[2]
        y_range = MAXS[1] - MINS[1]
        max_range = max(x_range, z_range, y_range)
        
        # Use smaller padding for tighter framing
        padding = max(max_range * 0.1, 0.1)  # 10% padding, minimum 0.1
        
        x_center = (MINS[0] + MAXS[0]) / 2
        z_center = (MINS[2] + MAXS[2]) / 2
        y_max = MAXS[1]
        
        # Use the actual range plus small padding for each axis
        x_limit = max(x_range / 2, max_range / 2) + padding
        z_limit = max(z_range / 2, max_range / 2) + padding
        y_limit = y_max + padding
        
        # Colors: Leader = blue/red, Follower = green/orange
        leader_colors = ['blue', 'cyan', 'darkblue', 'blue', 'cyan',
                        'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
                        'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
        follower_colors = ['green', 'lime', 'darkgreen', 'green', 'lime',
                          'darkgreen', 'darkgreen', 'darkgreen', 'darkgreen', 'darkgreen',
                          'orange', 'orange', 'orange', 'orange', 'orange']
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set dynamic limits based on actual data bounds
        ax.set_xlim(x_center - x_limit, x_center + x_limit)
        ax.set_ylim(0, y_limit)
        ax.set_zlim(z_center - z_limit, z_center + z_limit)
        fig.suptitle(wrapped_title, fontsize=14)
        ax.grid(False)
        
        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.3))
            ax.add_collection3d(xz_plane)
        
        def update(index):
            # Clear previous frame
            for line in ax.lines[:]:
                line.remove()
            for collection in ax.collections[:]:
                collection.remove()
            
            ax.view_init(elev=120, azim=-90)
            # Adjust camera distance based on data range for better framing
            # Smaller distance = closer view = larger figures
            ax.dist = max(max_range * 1.2, 4.5)  # Scale distance with data size, minimum 4.5 for closer view
            
            # Plot ground plane using actual bounds
            plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
            
            # Plot leader skeleton (blue/red)
            for i, chain in enumerate(t2m_kinematic_chain):
                if i < len(leader_colors):
                    color = leader_colors[i]
                    linewidth = 3.0 if i < 5 else 2.0
                    if i < len(leader_data[index]):
                        chain_valid = [j for j in chain if j < leader_data.shape[1]]
                        if len(chain_valid) > 1:
                            ax.plot3D(
                                leader_data[index, chain_valid, 0],
                                leader_data[index, chain_valid, 1],
                                leader_data[index, chain_valid, 2],
                                linewidth=linewidth,
                                color=color,
                                label='Leader' if i == 0 else ''
                            )
            
            # Plot follower skeleton (green/orange)
            for i, chain in enumerate(t2m_kinematic_chain):
                if i < len(follower_colors):
                    color = follower_colors[i]
                    linewidth = 3.0 if i < 5 else 2.0
                    if i < len(follower_data[index]):
                        chain_valid = [j for j in chain if j < follower_data.shape[1]]
                        if len(chain_valid) > 1:
                            ax.plot3D(
                                follower_data[index, chain_valid, 0],
                                follower_data[index, chain_valid, 1],
                                follower_data[index, chain_valid, 2],
                                linewidth=linewidth,
                                color=color,
                                label='Follower' if i == 0 else ''
                            )
            
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        
        # Create animation
        frame_number = min_frames
        ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
        
        # Set matplotlib parameters for minimal whitespace
        original_settings = {
            'savefig.dpi': plt.rcParams.get('savefig.dpi', 'figure'),
            'savefig.bbox': plt.rcParams.get('savefig.bbox', None),
            'savefig.pad_inches': plt.rcParams.get('savefig.pad_inches', 0.1),
            'figure.subplot.left': plt.rcParams.get('figure.subplot.left', 0.125),
            'figure.subplot.right': plt.rcParams.get('figure.subplot.right', 0.9),
            'figure.subplot.bottom': plt.rcParams.get('figure.subplot.bottom', 0.1),
            'figure.subplot.top': plt.rcParams.get('figure.subplot.top', 0.9),
        }
        
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.05
        plt.rcParams['figure.subplot.left'] = 0.05
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.05
        plt.rcParams['figure.subplot.top'] = 0.92
        
        # Save animation
        debug_print(f"Saving combined animation to {output_path}")
        ani.save(output_path, fps=fps)
        plt.close()
        
        # Restore settings
        for key, value in original_settings.items():
            plt.rcParams[key] = value
        
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Video file was not created at {output_path}")
        
        file_size = os.path.getsize(output_path)
        debug_print(f"Combined video saved: {output_path}, size: {file_size} bytes")
        return output_path
        
    except Exception as e:
        import traceback
        error_msg = f"Error in render_combined_skeletons: {str(e)}\n{traceback.format_exc()}"
        debug_print(error_msg)
        raise RuntimeError(error_msg) from e


def render_skeleton_from_keypoints(
    keypoints: np.ndarray,
    output_path: str,
    title: str = "Skeleton Animation",
    fps: int = 20,
    radius: int = 4,
    figsize: Tuple[int, int] = (6, 6),
    dpi: int = 100
) -> str:
    """
    Render skeleton animation from 3D keypoints.
    
    Uses matplotlib's FuncAnimation with 3D plotting to create skeleton animations.
    The animation shows 3D joint positions connected by lines representing the skeleton structure.
    
    Args:
        keypoints: Array of shape (n_frames, n_joints, 3) or (n_frames, n_joints*3)
        output_path: Path to save the video
        title: Title for the animation (will be wrapped if too long)
        fps: Frames per second (default: 20, lower values = smaller files)
        radius: Radius for visualization (not used in current implementation)
        figsize: Figure size in inches (width, height). Default (6, 6) for smaller files.
                 Larger values = higher quality but larger file size.
        dpi: Dots per inch for output. Default 100. Lower values = smaller files.
        
    Returns:
        Path to the saved video file
    """
    try:
        # Ensure it's numpy array
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.detach().cpu().numpy()
        
        # Reshape if needed
        if len(keypoints.shape) == 2:
            # Assume (n_frames, n_joints*3)
            n_frames = keypoints.shape[0]
            n_joints = keypoints.shape[1] // 3
            if n_joints * 3 != keypoints.shape[1]:
                raise ValueError(f"Cannot reshape keypoints from shape {keypoints.shape} to (n_frames, n_joints, 3)")
            keypoints = keypoints.reshape(n_frames, n_joints, 3)
        elif len(keypoints.shape) != 3:
            raise ValueError(f"Expected 2D or 3D keypoints array, got shape {keypoints.shape}")
        
        if keypoints.shape[2] != 3:
            raise ValueError(f"Expected 3D coordinates, got shape {keypoints.shape}")
        
        debug_print(f"Rendering skeleton: {keypoints.shape[0]} frames, {keypoints.shape[1]} joints")
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        os.makedirs(output_dir, exist_ok=True)
        debug_print(f"Output directory: {output_dir}, exists: {os.path.exists(output_dir)}")
        
        # Wrap title to fit in figure
        wrapped_title = wrap_title(title, max_length=35)
        debug_print(f"Original title: {title}, Wrapped: {wrapped_title}")
        
        # Use existing plot_3d_motion function
        # Note: This uses matplotlib's FuncAnimation with 3D plotting (mpl_toolkits.mplot3d)
        # It creates an animated 3D skeleton visualization frame by frame
        debug_print(f"Calling plot_3d_motion with output_path: {output_path}, figsize={figsize}, fps={fps}, dpi={dpi}")
        try:
            # Set matplotlib parameters to minimize whitespace and reduce file size
            import matplotlib.pyplot as plt
            import matplotlib
            
            # Save original settings
            original_settings = {
                'savefig.dpi': plt.rcParams.get('savefig.dpi', 'figure'),
                'savefig.bbox': plt.rcParams.get('savefig.bbox', None),
                'savefig.pad_inches': plt.rcParams.get('savefig.pad_inches', 0.1),
                'figure.subplot.left': plt.rcParams.get('figure.subplot.left', 0.125),
                'figure.subplot.right': plt.rcParams.get('figure.subplot.right', 0.9),
                'figure.subplot.bottom': plt.rcParams.get('figure.subplot.bottom', 0.1),
                'figure.subplot.top': plt.rcParams.get('figure.subplot.top', 0.9),
                'figure.titlesize': plt.rcParams.get('figure.titlesize', 'large'),
            }
            
            # Minimize whitespace - reduce padding and margins
            plt.rcParams['savefig.dpi'] = dpi
            plt.rcParams['savefig.bbox'] = 'tight'  # Remove extra whitespace
            plt.rcParams['savefig.pad_inches'] = 0.05  # Minimal padding
            plt.rcParams['figure.subplot.left'] = 0.05  # Reduce left margin
            plt.rcParams['figure.subplot.right'] = 0.95  # Reduce right margin
            plt.rcParams['figure.subplot.bottom'] = 0.05  # Reduce bottom margin
            plt.rcParams['figure.subplot.top'] = 0.92  # Reduce top margin (leave space for title)
            plt.rcParams['figure.titlesize'] = 'small'  # Smaller title font
            
            plot_3d_motion(output_path, t2m_kinematic_chain, keypoints, title=wrapped_title, fps=fps, radius=radius, figsize=figsize)
            
            # Restore original settings
            for key, value in original_settings.items():
                plt.rcParams[key] = value
            
            debug_print("plot_3d_motion completed")
        except Exception as e:
            debug_print(f"Error in plot_3d_motion: {e}")
            raise
        
        if not os.path.exists(output_path):
            debug_print(f"Video file was not created at {output_path}")
            raise FileNotFoundError(f"Video file was not created at {output_path}")
        
        file_size = os.path.getsize(output_path)
        debug_print(f"Video saved successfully to {output_path}, size: {file_size} bytes")
        return output_path
    except Exception as e:
        import traceback
        error_msg = f"Error in render_skeleton_from_keypoints: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise RuntimeError(error_msg) from e


def render_skeleton_from_ric(
    ric_data: np.ndarray,
    output_path: str,
    title: str = "Skeleton Animation",
    fps: int = 20,
    radius: int = 4,
    joints_num: int = 22
) -> str:
    """
    Render skeleton animation from RIC (Root, Imitation, Contact) format data.
    
    Args:
        ric_data: Array in RIC format
        output_path: Path to save the video
        title: Title for the animation
        fps: Frames per second
        radius: Radius for visualization
        joints_num: Number of joints
        
    Returns:
        Path to the saved video file
    """
    if isinstance(ric_data, torch.Tensor):
        ric_data = ric_data.detach().cpu().numpy()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positions = recover_from_ric(torch.from_numpy(ric_data).float().to(device), joints_num)
    positions = positions.squeeze().detach().cpu().numpy()
    
    return render_skeleton_from_keypoints(positions, output_path, title, fps, radius)


def render_vqvae_reconstruction_comparison(
    original_keypoints: np.ndarray,
    reconstructed_keypoints: np.ndarray,
    output_path: str,
    title: str = "VQVAE Reconstruction",
    figsize: Tuple[int, int] = (6, 6),
    dpi: int = 100,
    fps: int = 20
) -> str:
    """
    Render an overlapped view of original and reconstructed motion.
    
    Args:
        original_keypoints: Original motion keypoints (T, 22, 3)
        reconstructed_keypoints: Reconstructed motion keypoints (T, 22, 3)
        output_path: Path to save the video
        title: Title for the animation
        figsize: Figure size
        dpi: DPI for the figure
        fps: Frames per second
        
    Returns:
        Path to the saved video
    """
    try:
        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Ensure same number of frames
        min_frames = min(len(original_keypoints), len(reconstructed_keypoints))
        original_keypoints = original_keypoints[:min_frames]
        reconstructed_keypoints = reconstructed_keypoints[:min_frames]
        
        # Calculate bounds for both motions with minimal padding
        all_data = np.concatenate([original_keypoints, reconstructed_keypoints], axis=0)
        MINS = all_data.min(axis=(0, 1))
        MAXS = all_data.max(axis=(0, 1))
        # Use tighter padding (5% instead of default) for less whitespace
        padding = (MAXS - MINS) * 0.05
        max_range = max(MAXS - MINS) / 2.0
        center = (MINS + MAXS) / 2.0
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set axis limits with minimal padding
        ax.set_xlim([MINS[0] - padding[0], MAXS[0] + padding[0]])
        ax.set_ylim([MINS[1] - padding[1], MAXS[1] + padding[1]])
        ax.set_zlim([MINS[2] - padding[2], MAXS[2] + padding[2]])
        
        # Ground plane helper
        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            verts = [
                [minx, miny, minz],
                [maxx, miny, minz],
                [maxx, miny, maxz],
                [minx, miny, maxz]
            ]
            ax.add_collection3d(Poly3DCollection([verts], alpha=0.1, facecolor='gray', edgecolor='none'))
        
        def update(index):
            # Clear previous frame
            for line in ax.lines[:]:
                line.remove()
            for collection in ax.collections[:]:
                collection.remove()
            
            ax.view_init(elev=120, azim=-90)
            # Reduce camera distance for tighter framing (less whitespace)
            ax.dist = max(max_range * 1.1, 4.0)  # Closer view, less padding
            
            # Plot ground plane
            plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
            
            # Plot original skeleton (blue)
            for i, chain in enumerate(t2m_kinematic_chain):
                chain_valid = [j for j in chain if j < original_keypoints.shape[1]]
                if len(chain_valid) > 1:
                    ax.plot3D(
                        original_keypoints[index, chain_valid, 0],
                        original_keypoints[index, chain_valid, 1],
                        original_keypoints[index, chain_valid, 2],
                        linewidth=3.0,
                        color='blue',
                        alpha=0.7,
                        label='Original' if i == 0 else ''
                    )
            
            # Plot reconstructed skeleton (red, dashed)
            for i, chain in enumerate(t2m_kinematic_chain):
                chain_valid = [j for j in chain if j < reconstructed_keypoints.shape[1]]
                if len(chain_valid) > 1:
                    ax.plot3D(
                        reconstructed_keypoints[index, chain_valid, 0],
                        reconstructed_keypoints[index, chain_valid, 1],
                        reconstructed_keypoints[index, chain_valid, 2],
                        linewidth=2.5,
                        color='red',
                        linestyle='--',
                        alpha=0.7,
                        label='Reconstructed' if i == 0 else ''
                    )
            
            ax.set_title(wrap_title(title), fontsize=10, pad=5)
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=min_frames, interval=1000/fps, repeat=False)
        
        # Set matplotlib parameters for minimal whitespace (same as other rendering functions)
        original_settings = {
            'savefig.dpi': plt.rcParams.get('savefig.dpi', 'figure'),
            'savefig.bbox': plt.rcParams.get('savefig.bbox', None),
            'savefig.pad_inches': plt.rcParams.get('savefig.pad_inches', 0.1),
            'figure.subplot.left': plt.rcParams.get('figure.subplot.left', 0.125),
            'figure.subplot.right': plt.rcParams.get('figure.subplot.right', 0.9),
            'figure.subplot.bottom': plt.rcParams.get('figure.subplot.bottom', 0.1),
            'figure.subplot.top': plt.rcParams.get('figure.subplot.top', 0.9),
        }
        
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.05
        plt.rcParams['figure.subplot.left'] = 0.05
        plt.rcParams['figure.subplot.right'] = 0.95
        plt.rcParams['figure.subplot.bottom'] = 0.05
        plt.rcParams['figure.subplot.top'] = 0.92
        
        # Save animation
        debug_print(f"Saving VQVAE comparison animation to {output_path}")
        ani.save(output_path, fps=fps)
        plt.close()
        
        # Restore settings
        for key, value in original_settings.items():
            plt.rcParams[key] = value
        
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Video file was not created at {output_path}")
        
        return output_path
        
    except Exception as e:
        import traceback
        error_msg = f"Error in render_vqvae_reconstruction_comparison: {str(e)}\n{traceback.format_exc()}"
        debug_print(error_msg)
        raise RuntimeError(error_msg) from e


def decode_motion_from_vq_tokens(
    vq_tokens: np.ndarray,
    motion_tokenizer,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Decode motion from VQ tokens using Motion_tokenizer.
    
    Args:
        vq_tokens: VQ tokens (may need reindexing)
        motion_tokenizer: Motion_tokenizer instance
        device: Device to use
        
    Returns:
        Reconstructed keypoints (T, 22, 3)
    """
    try:
        import torch
        
        # Convert to tensor if needed
        if isinstance(vq_tokens, np.ndarray):
            vq_tokens = torch.from_numpy(vq_tokens).long()
        
        # Reindex tokens if needed (remove LLM offset)
        # From demo.py: motion_tokens = motion_tokens - self.nb_text_tokens - 2
        # But vq_tokens from dataset might already be in VQ space
        # Check if tokens are in LLM space (typically > 1000) or VQ space (< 1000)
        if vq_tokens.max() > 1000:
            # Likely in LLM space, need to reindex
            vq_tokens = vq_tokens - motion_tokenizer.nb_text_tokens - 2
            vq_tokens = torch.clamp(vq_tokens, min=0)
        
        # Ensure on correct device
        vq_tokens = vq_tokens.to(device)
        
        # Decode
        motion = motion_tokenizer.net.forward_decoder(vq_tokens.unsqueeze(0))
        motion = motion_tokenizer.denormalize(motion.detach().cpu().numpy())
        
        # Recover keypoints
        motion_tensor = torch.from_numpy(motion).float().to(device)
        keypoints = recover_from_ric(motion_tensor, 22)
        
        return keypoints.squeeze().detach().cpu().numpy()
        
    except Exception as e:
        import traceback
        error_msg = f"Error in decode_motion_from_vq_tokens: {str(e)}\n{traceback.format_exc()}"
        debug_print(error_msg)
        raise RuntimeError(error_msg) from e


def create_video_with_audio(
    video_path: str,
    output_path: str,
    audio_path: Optional[str] = None
) -> str:
    """
    Combine video with audio using ffmpeg.
    
    Args:
        video_path: Path to video file
        output_path: Path to save the combined video
        audio_path: Path to audio file (optional)
        
    Returns:
        Path to the output file
    """
    import subprocess
    
    if audio_path and os.path.exists(audio_path):
        debug_print(f"Combining video {video_path} with audio {audio_path} -> {output_path}")
        # Combine video and audio
        # Use -map 0:v:0 to map video from first input, -map 1:a:0 to map audio from second input
        # -shortest to end when shortest stream ends
        # -strict -2 to allow experimental codecs if needed
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_path
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            debug_print(f"FFmpeg output: {result.stdout}")
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Output file was not created: {output_path}")
            debug_print(f"Successfully created video with audio: {output_path}")
        except subprocess.CalledProcessError as e:
            debug_print(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to combine video with audio: {e.stderr}") from e
    else:
        # Just copy video if no audio
        debug_print(f"No audio provided, copying video: {video_path} -> {output_path}")
        import shutil
        shutil.copy(video_path, output_path)
    
    return output_path


def decode_audio_tokens(audio_tokens: np.ndarray, wavtokenizer=None, device='cpu') -> Optional[np.ndarray]:
    """
    Decode audio tokens to audio waveform.
    
    Args:
        audio_tokens: Audio token array
        wavtokenizer: WavTokenizer instance (optional)
        device: Device to run decoding on
        
    Returns:
        Audio waveform array or None if decoding fails
    """
    if wavtokenizer is None:
        # Return None if no tokenizer provided
        return None
    
    try:
        if isinstance(audio_tokens, np.ndarray):
            audio_tokens = torch.from_numpy(audio_tokens)
        
        # Move to device
        audio_tokens = audio_tokens.to(device)
        
        # Decode using wavtokenizer: codes -> features -> audio
        features = wavtokenizer.codes_to_features(audio_tokens)
        audio = wavtokenizer.decode(features)
        
        # Convert to numpy and squeeze batch dimension if present
        audio = audio.detach().cpu().numpy()
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        
        return audio
    except Exception as e:
        debug_print(f"Error decoding audio tokens: {e}")
        import traceback
        debug_print(traceback.format_exc())
        return None


def extract_original_video_clip(
    vid_name: str,
    start_time: float,
    end_time: float,
    dataset_root: str = "/localhome/pjomeyaz/Payam_Files/Projects/Salsa_Dance/Dataset",
    output_path: Optional[str] = None
) -> Optional[str]:
    """
    Extract and crop a video clip from the original dataset video.
    
    Args:
        vid_name: Video name from aux_info (e.g., "Pair1_8_7_take1_1_leader_subject,Pair1_8_7_take1_1_follower_subject")
        start_time: Start time in seconds
        end_time: End time in seconds
        dataset_root: Root directory of the dataset
        output_path: Path to save the cropped video (optional, will create temp file if None)
        
    Returns:
        Path to the cropped video file, or None if extraction fails
    """
    try:
        # Extract the base video name from vid_name
        # Format examples:
        # "Pair1_8_7_take1_2_subject-Pair1_8_7_leader_subject_stageii,Pair1_8_7_take1_2_subject-Pair1_8_7_follower_subject_stageii"
        # "Pair1_8_7_take1_1_leader_subject,Pair1_8_7_take1_1_follower_subject"
        # We need: "Pair1_8_7_take1_2" or "Pair1_8_7_take1_1"
        
        if ',' in vid_name:
            # Take the first part before comma
            base_name = vid_name.split(',')[0]
        else:
            base_name = vid_name
        
        # Handle format with "subject-" prefix
        if 'subject-' in base_name:
            # Extract part after "subject-"
            base_name = base_name.split('subject-')[-1]
        
        # Remove suffixes like "_leader_subject", "_follower_subject", "_stageii", etc.
        parts = base_name.split('_')
        
        # Find the pattern: PairX_Y_Z_takeX_X
        # Usually: Pair + number + number + number + take + number + number
        if 'take' in base_name.lower():
            take_idx = [i for i, p in enumerate(parts) if 'take' in p.lower()][0]
            video_key = '_'.join(parts[:take_idx+2])  # PairX_Y_Z_takeX_X
            debug_print(f"Extracted video_key: {video_key} from base_name: {base_name}")
        else:
            # Fallback: use first 5 parts
            video_key = '_'.join(parts[:5])
            debug_print(f"Using fallback video_key: {video_key} from base_name: {base_name}")
        
        # Search for video file in dataset
        # Videos are in compas3d/PairX/PairX_songY_takeZ/PairX_songY_takeZ.mp4 structure
        # Try multiple naming patterns and locations
        possible_paths = []
        
        # Try exact match first
        possible_paths.extend([
            os.path.join(dataset_root, "compas3d", "**", f"{video_key}.mp4"),
            os.path.join(dataset_root, "compas3d", "**", f"{video_key}*.mp4"),
            os.path.join(dataset_root, "Animations", f"{video_key}.mp4"),
            os.path.join(dataset_root, "synced_animation", f"{video_key}.mp4"),
            os.path.join(dataset_root, "videos", f"{video_key}.mp4"),
            os.path.join(dataset_root, "**", "Animations", f"{video_key}.mp4"),
            os.path.join(dataset_root, "**", f"{video_key}.mp4"),
            os.path.join(dataset_root, f"{video_key}.mp4"),
        ])
        
        # Also try pattern matching: Pair1_8_7_take1_2 -> Pair1_song*_take*
        # Extract Pair number and try to match song/take pattern
        # The actual files are: Pair1_song1_take1.mp4, Pair1_song1_take2.mp4, etc.
        if video_key.startswith("Pair") and "_take" in video_key:
            pair_num = video_key.split("_")[0]  # "Pair1"
            take_part = video_key.split("_take")[1]  # "1_2" or "1"
            # Extract first take number (e.g., "1" from "1_2")
            take_num = take_part.split("_")[0] if "_" in take_part else take_part
            debug_print(f"Trying to match: pair={pair_num}, take={take_num}")
            
            # Add pattern matching for compas3d structure
            possible_paths.extend([
                os.path.join(dataset_root, "compas3d", pair_num, f"{pair_num}_song*_take{take_num}", f"{pair_num}_song*_take{take_num}.mp4"),
                os.path.join(dataset_root, "compas3d", pair_num, "**", f"{pair_num}_song*_take{take_num}.mp4"),
            ])
        
        # Search for video file
        video_path = None
        
        # First try direct paths (non-glob)
        for path in possible_paths:
            if '**' not in path and '*' not in path and os.path.exists(path):
                video_path = path
                debug_print(f"Found video at direct path: {video_path}")
                break
        
        # If not found, try glob patterns
        if video_path is None:
            for pattern in possible_paths:
                if '**' in pattern or '*' in pattern:
                    matches = glob(pattern, recursive=True)
                    if matches:
                        # Prefer matches with correct take number
                        if take_num:
                            for match in matches:
                                if f"_take{take_num}" in match or f"take{take_num}" in match:
                                    video_path = match
                                    debug_print(f"Found video via glob (take match): {video_path}")
                                    break
                        if not video_path:
                            video_path = matches[0]
                            debug_print(f"Found video via glob: {video_path}")
                        if video_path:
                            break
            
            # Additional glob patterns if still not found
            if video_path is None:
                for pattern in [
                    os.path.join(dataset_root, "compas3d", "**", "*.mp4"),  # All videos in compas3d
                    os.path.join(dataset_root, "**", f"{video_key}*.mp4"),
                    os.path.join(dataset_root, "**", f"*{video_key}*.mp4"),
                ]:
                    matches = glob(pattern, recursive=True)
                    if matches:
                        video_path = matches[0]
                        debug_print(f"Found video via extended glob: {video_path}")
                        break
        
        if video_path is None:
            debug_print(f"Could not find video file for {video_key}")
            return None
        
        debug_print(f"Found video: {video_path}")
        
        # Create output path if not provided
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"original_video_{int(start_time)}_{int(end_time)}.mp4")
        
        # Extract clip using ffmpeg with audio
        duration = end_time - start_time
        
        # First, check if video has audio stream
        probe_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type', '-of', 'csv=p=0',
            video_path
        ]
        has_audio = False
        try:
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
            has_audio = probe_result.returncode == 0 and 'audio' in probe_result.stdout.lower()
        except:
            pass
        
        # Build ffmpeg command
        if has_audio:
            # Video with audio
            cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',  # Encode audio to AAC
                '-b:a', '192k',  # Audio bitrate
                '-map', '0:v:0',  # Map video stream
                '-map', '0:a:0',  # Map audio stream
                '-shortest',  # End when shortest stream ends
                '-avoid_negative_ts', 'make_zero',
                output_path
            ]
        else:
            # Video without audio (just copy video)
            debug_print(f"Video has no audio stream, extracting video only")
            cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-an',  # No audio
                '-avoid_negative_ts', 'make_zero',
                output_path
            ]
        
        debug_print(f"Extracting video clip: {start_time}s to {end_time}s from {video_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            debug_print(f"FFmpeg error: {result.stderr}")
            # Try fallback without explicit mapping
            debug_print("Trying fallback command without explicit stream mapping...")
            cmd_fallback = [
                'ffmpeg', '-y', '-v', 'error',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'copy',  # Copy audio if available
                '-avoid_negative_ts', 'make_zero',
                output_path
            ]
            result = subprocess.run(cmd_fallback, capture_output=True, text=True)
            if result.returncode != 0:
                debug_print(f"FFmpeg fallback error: {result.stderr}")
                return None
        
        if os.path.exists(output_path):
            debug_print(f"Video clip saved to {output_path}")
            return output_path
        else:
            debug_print(f"Video clip was not created at {output_path}")
            return None
            
    except Exception as e:
        import traceback
        error_msg = f"Error extracting video clip: {str(e)}\n{traceback.format_exc()}"
        debug_print(error_msg)
        return None


def save_audio_waveform(audio: np.ndarray, output_path: str, sample_rate: int = 24000):
    """
    Save audio waveform to file.
    
    Args:
        audio: Audio waveform array
        output_path: Path to save audio file
        sample_rate: Sample rate of the audio
    """
    try:
        import soundfile as sf
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        sf.write(output_path, audio, sample_rate)
    except ImportError:
        # Fallback to scipy if soundfile is not available
        try:
            from scipy.io import wavfile
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            wavfile.write(output_path, sample_rate, audio)
        except ImportError:
            raise ImportError("Either soundfile or scipy is required to save audio files")

