"""
Utility functions for extracting relationship features between leader and follower dancers.
"""

import torch
import numpy as np
from typing import Union, Tuple

# Debug flag for detailed verification output (set to True for debugging)
DEBUG = False

# HumanML3D dimension indices (from analyze_humanml3d_data.py)
ROOT_ROT_VELOCITY_START = 0
ROOT_ROT_VELOCITY_END = 1
ROOT_LINEAR_VELOCITY_START = 1
ROOT_LINEAR_VELOCITY_END = 3
ROOT_Y_START = 3
ROOT_Y_END = 4


def recover_root_rot_pos(data):
    """
    Recover root rotation and position from HumanML3D data.
    This is a simplified version of the recovery function from HumanMl3D_functions.py.
    
    Args:
        data: HumanML3D data of shape (..., seq_len, 263) or (seq_len, 263)
              Expected format:
              - data[..., 0]: root_rot_velocity (angular velocity around Y-axis)
              - data[..., 1:3]: root_linear_velocity (on XZ plane)
              - data[..., 3]: root_y (height/Y position)
    
    Returns:
        r_rot_ang: Root rotation angle (Y-axis) of shape (..., seq_len)
        r_pos: Root position of shape (..., seq_len, 3) where [x, y, z]
    """
    # Convert to numpy if torch tensor
    is_torch = isinstance(data, torch.Tensor)
    original_device = None
    if is_torch:
        original_device = data.device
        data = data.detach().cpu().numpy()
    
    # Handle batch dimension
    has_batch = len(data.shape) == 3
    if not has_batch:
        data = data[np.newaxis, ...]
    
    batch_size, seq_len, _ = data.shape
    
    # Extract root rotation velocity
    rot_vel = data[..., 0]  # (batch, seq_len)
    
    # Integrate rotation velocity to get rotation angle
    # Frame 0 starts at 0, subsequent frames integrate velocity
    r_rot_ang = np.zeros_like(rot_vel)
    r_rot_ang[:, 1:] = rot_vel[:, :-1]  # Velocity at frame i affects rotation at frame i+1
    r_rot_ang = np.cumsum(r_rot_ang, axis=-1)  # Cumulative sum
    
    # Extract root linear velocity and integrate to get position
    # Root position starts at (0, root_y[0], 0) for frame 0
    r_pos = np.zeros((batch_size, seq_len, 3))
    r_pos[:, 1:, [0, 2]] = data[:, :-1, 1:3]  # Velocity at frame i affects position at frame i+1
    
    # Rotate velocity by inverse rotation to get position in world frame
    # Simplified: for small rotations, we can approximate
    # For exact calculation, we'd need quaternion rotation, but for first frame this is sufficient
    r_pos = np.cumsum(r_pos, axis=-2)  # Cumulative sum
    
    # Set Y position (height) directly from data
    r_pos[:, :, 1] = data[:, :, 3]  # root_y
    
    # Remove batch dimension if it wasn't there originally
    if not has_batch:
        r_rot_ang = r_rot_ang[0]
        r_pos = r_pos[0]
    
    # Convert back to torch if input was torch
    if is_torch:
        r_rot_ang = torch.from_numpy(r_rot_ang).float()
        r_pos = torch.from_numpy(r_pos).float()
        if original_device is not None:
            r_rot_ang = r_rot_ang.to(original_device)
            r_pos = r_pos.to(original_device)
    
    return r_rot_ang, r_pos


def extract_first_frame_relative_pose(
    leader_motion: Union[torch.Tensor, np.ndarray],
    follower_motion: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Extract first-frame relative pose (sx, sz, sθ) between leader and follower.
    
    This represents the initial spatial relationship between two motions:
    - (sx, sz) ∈ ℝ²: Relative position on the root XZ-plane at frame 0
    - sθ ∈ ℝ: Relative orientation (rotation difference) at frame 0
    
    Calculation:
    1. Recover root positions and rotations for both leader and follower
    2. At frame 0, compute:
       - sx = follower_root_x[0] - leader_root_x[0]
       - sz = follower_root_z[0] - leader_root_z[0]
       - sθ = follower_root_rotation[0] - leader_root_rotation[0]
    
    Evidence of Correctness:
    - From recover_root_rot_pos: root position at frame 0 is (0, root_y[0], 0)
    - Root rotation at frame 0 is 0 (starts from 0 and integrates velocity)
    - Therefore, at frame 0: both start at origin on XZ plane, but may have different heights
    - The relative pose (sx, sz, sθ) captures the initial offset:
      * If both truly start at (0, y, 0), then (sx, sz) = (0, 0) and sθ = 0
      * However, if there's an initial offset in the original data, this will be captured
      * The height difference is NOT included in (sx, sz) as per specification (XZ-plane only)
    
    Args:
        leader_motion: Leader motion tensor/array of shape (seq_len, 263) or (batch, seq_len, 263)
        follower_motion: Follower motion tensor/array of shape (seq_len, 263) or (batch, seq_len, 263)
    
    Returns:
        First-frame relative pose tensor/array of shape (3,) or (batch, 3)
        Features: [sx, sz, sθ]
    """
    # Recover root positions and rotations
    leader_rot_ang, leader_pos = recover_root_rot_pos(leader_motion)
    follower_rot_ang, follower_pos = recover_root_rot_pos(follower_motion)
    
    # Handle batch dimension
    if len(leader_pos.shape) == 3:  # (batch, seq_len, 3)
        # Extract frame 0 for all batches
        leader_pos_0 = leader_pos[:, 0, :]  # (batch, 3)
        follower_pos_0 = follower_pos[:, 0, :]  # (batch, 3)
        leader_rot_0 = leader_rot_ang[:, 0]  # (batch,)
        follower_rot_0 = follower_rot_ang[:, 0]  # (batch,)
        
        # Compute relative pose at frame 0
        # (sx, sz) = relative position on XZ plane
        sx = follower_pos_0[:, 0] - leader_pos_0[:, 0]  # (batch,)
        sz = follower_pos_0[:, 2] - leader_pos_0[:, 2]  # (batch,)
        # sθ = relative orientation
        sθ = follower_rot_0 - leader_rot_0  # (batch,)
        
        # Stack into (batch, 3)
        relative_pose = np.stack([sx, sz, sθ], axis=-1) if isinstance(sx, np.ndarray) else torch.stack([sx, sz, sθ], dim=-1)
    else:  # (seq_len, 3)
        # Extract frame 0
        leader_pos_0 = leader_pos[0, :]  # (3,)
        follower_pos_0 = follower_pos[0, :]  # (3,)
        leader_rot_0 = leader_rot_ang[0]  # scalar
        follower_rot_0 = follower_rot_ang[0]  # scalar
        
        # Compute relative pose at frame 0
        sx = follower_pos_0[0] - leader_pos_0[0]  # X difference
        sz = follower_pos_0[2] - leader_pos_0[2]  # Z difference
        sθ = follower_rot_0 - leader_rot_0  # Rotation difference
        
        # Stack into (3,)
        if isinstance(sx, np.ndarray):
            relative_pose = np.array([sx, sz, sθ])
        else:
            relative_pose = torch.stack([sx, sz, sθ])
    
    return relative_pose


def extract_relationship_features(
    leader_motion: Union[torch.Tensor, np.ndarray],
    follower_motion: Union[torch.Tensor, np.ndarray],
    use_first_frame_pose: bool = False
) -> Union[torch.Tensor, np.ndarray]:
    """
    Extract relationship features between leader and follower motions.
    
    If use_first_frame_pose=True:
    - First-frame relative pose (sx, sz, sθ): 3D static initial pose relationship
    - This is broadcast/repeated for all frames in the sequence
    
    If use_first_frame_pose=False (default):
    - Root translation difference (3D: x, z, y) - temporal sequence
    - Root rotation difference (1D: rotation velocity difference) - temporal sequence
    
    Args:
        leader_motion: Leader motion tensor/array of shape (seq_len, 263) or (batch, seq_len, 263)
        follower_motion: Follower motion tensor/array of shape (seq_len, 263) or (batch, seq_len, 263)
        use_first_frame_pose: If True, use first-frame relative pose (sx, sz, sθ) instead of temporal differences
    
    Returns:
        Relationship features tensor/array:
        - If use_first_frame_pose=True: shape (seq_len, 3) or (batch, seq_len, 3) with [sx, sz, sθ] repeated
        - If use_first_frame_pose=False: shape (seq_len, 4) or (batch, seq_len, 4) with [root_translation_diff_x, root_translation_diff_z, root_translation_diff_y, root_rotation_diff]
    """
    if use_first_frame_pose:
        # Extract first-frame relative pose
        first_frame_pose = extract_first_frame_relative_pose(leader_motion, follower_motion)
        
        # Handle batch dimension
        has_batch = len(leader_motion.shape) == 3
        if not has_batch:
            seq_len = leader_motion.shape[0]
            # Repeat for all frames: (3,) -> (seq_len, 3)
            if isinstance(first_frame_pose, np.ndarray):
                relationship_features = np.tile(first_frame_pose[np.newaxis, :], (seq_len, 1))
            else:
                relationship_features = first_frame_pose.unsqueeze(0).repeat(seq_len, 1)
        else:
            batch_size, seq_len, _ = leader_motion.shape
            # Repeat for all frames: (batch, 3) -> (batch, seq_len, 3)
            if isinstance(first_frame_pose, np.ndarray):
                relationship_features = np.tile(first_frame_pose[:, np.newaxis, :], (1, seq_len, 1))
            else:
                relationship_features = first_frame_pose.unsqueeze(1).repeat(1, seq_len, 1)
        
        return relationship_features
    
    # Original temporal difference features (default behavior)
    # Convert to numpy if torch tensors (store device info first)
    is_torch = isinstance(leader_motion, torch.Tensor)
    original_device = None
    if is_torch:
        original_device = leader_motion.device
        leader_motion = leader_motion.detach().cpu().numpy()
        follower_motion = follower_motion.detach().cpu().numpy()
    
    # Handle batch dimension
    has_batch = len(leader_motion.shape) == 3
    if not has_batch:
        leader_motion = leader_motion[np.newaxis, ...]
        follower_motion = follower_motion[np.newaxis, ...]
    
    batch_size, seq_len, _ = leader_motion.shape
    
    # Extract root features from HumanML3D representation
    # Root rotation velocity (dim 0)
    leader_root_rot_vel = leader_motion[:, :, ROOT_ROT_VELOCITY_START:ROOT_ROT_VELOCITY_END]  # (batch, seq_len, 1)
    follower_root_rot_vel = follower_motion[:, :, ROOT_ROT_VELOCITY_START:ROOT_ROT_VELOCITY_END]  # (batch, seq_len, 1)
    
    # Root linear velocity on xz plane (dims 1-2)
    leader_root_linear_vel = leader_motion[:, :, ROOT_LINEAR_VELOCITY_START:ROOT_LINEAR_VELOCITY_END]  # (batch, seq_len, 2)
    follower_root_linear_vel = follower_motion[:, :, ROOT_LINEAR_VELOCITY_START:ROOT_LINEAR_VELOCITY_END]  # (batch, seq_len, 2)
    
    # Root height (dim 3)
    leader_root_y = leader_motion[:, :, ROOT_Y_START:ROOT_Y_END]  # (batch, seq_len, 1)
    follower_root_y = follower_motion[:, :, ROOT_Y_START:ROOT_Y_END]  # (batch, seq_len, 1)
    
    # Compute relationship features
    # 1. Root translation difference (3D: x, z, y)
    # Note: root_linear_velocity is on xz plane, root_y is height
    root_translation_diff = np.concatenate([
        leader_root_linear_vel - follower_root_linear_vel,  # (batch, seq_len, 2) - x, z difference
        leader_root_y - follower_root_y  # (batch, seq_len, 1) - y difference
    ], axis=-1)  # (batch, seq_len, 3)
    
    # 2. Root rotation difference (1D)
    root_rotation_diff = leader_root_rot_vel - follower_root_rot_vel  # (batch, seq_len, 1)
    
    # Concatenate all relationship features
    relationship_features = np.concatenate([
        root_translation_diff,  # (batch, seq_len, 3)
        root_rotation_diff  # (batch, seq_len, 1)
    ], axis=-1)  # (batch, seq_len, 4)
    
    # Remove batch dimension if it wasn't there originally
    if not has_batch:
        relationship_features = relationship_features[0]  # (seq_len, 4)
    
    # Convert back to torch if input was torch
    if is_torch:
        relationship_features = torch.from_numpy(relationship_features).float()
        if original_device is not None:
            relationship_features = relationship_features.to(original_device)
    
    return relationship_features


def extract_relationship_features_from_keypoints3d(
    leader_keypoints3d: Union[torch.Tensor, np.ndarray],
    follower_keypoints3d: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Extract relationship features between leader and follower from original 3D keypoints.
    
    This function computes relationship features directly from keypoints3d (shape: seq_len, 22, 3),
    preserving the global spatial relationships that are lost in normalized HumanML3D representation.
    
    Features computed:
    1. Root position difference (3D: x, y, z) - temporal sequence
       - follower_root_pos - leader_root_pos for each frame
    2. Root rotation difference (1D: yaw angle difference) - temporal sequence
       - Computed from forward direction (root to neck joint)
       - follower_yaw - leader_yaw for each frame
    
    Args:
        leader_keypoints3d: Leader keypoints tensor/array of shape (seq_len, 22, 3)
                           Joint 0 is root/pelvis, joint 12 is neck
        follower_keypoints3d: Follower keypoints tensor/array of shape (seq_len, 22, 3)
    
    Returns:
        Relationship features tensor/array of shape (seq_len, 4) or (batch, seq_len, 4)
        Features: [root_translation_diff_x, root_translation_diff_z, root_translation_diff_y, root_rotation_diff]
    """
    # Convert to numpy if torch tensors (store device info first)
    is_torch = isinstance(leader_keypoints3d, torch.Tensor)
    original_device = None
    if is_torch:
        original_device = leader_keypoints3d.device
        leader_keypoints3d = leader_keypoints3d.detach().cpu().numpy()
        follower_keypoints3d = follower_keypoints3d.detach().cpu().numpy()
    
    # Handle batch dimension
    has_batch = len(leader_keypoints3d.shape) == 4  # (batch, seq_len, 22, 3)
    if not has_batch:
        leader_keypoints3d = leader_keypoints3d[np.newaxis, ...]
        follower_keypoints3d = follower_keypoints3d[np.newaxis, ...]
    
    batch_size, seq_len, num_joints, dim = leader_keypoints3d.shape
    assert num_joints == 22, f"Expected 22 joints, got {num_joints}"
    assert dim == 3, f"Expected 3D positions, got {dim}"
    
    # Extract root positions (joint 0)
    leader_root_pos = leader_keypoints3d[:, :, 0, :]  # (batch, seq_len, 3)
    follower_root_pos = follower_keypoints3d[:, :, 0, :]  # (batch, seq_len, 3)
    
    # Compute root position difference (3D: x, y, z)
    # follower - leader: positive means follower is ahead/right/above leader
    root_translation_diff = follower_root_pos - leader_root_pos  # (batch, seq_len, 3)
    
    # Compute root rotation (yaw angle) from forward direction
    # Forward direction: from root (joint 0) to neck (joint 12)
    # Yaw angle: angle of forward direction in XZ plane
    leader_neck = leader_keypoints3d[:, :, 12, :]  # (batch, seq_len, 3)
    follower_neck = follower_keypoints3d[:, :, 12, :]  # (batch, seq_len, 3)
    
    # Forward direction vector (root to neck)
    leader_forward = leader_neck - leader_root_pos  # (batch, seq_len, 3)
    follower_forward = follower_neck - follower_root_pos  # (batch, seq_len, 3)
    
    # Compute yaw angle from forward direction in XZ plane
    # yaw = atan2(forward_x, forward_z) for each frame
    leader_yaw = np.arctan2(leader_forward[:, :, 0], leader_forward[:, :, 2])  # (batch, seq_len)
    follower_yaw = np.arctan2(follower_forward[:, :, 0], follower_forward[:, :, 2])  # (batch, seq_len)
    
    # Compute rotation difference (follower_yaw - leader_yaw)
    # Normalize to [-pi, pi] range
    root_rotation_diff = follower_yaw - leader_yaw  # (batch, seq_len)
    root_rotation_diff = np.arctan2(np.sin(root_rotation_diff), np.cos(root_rotation_diff))  # Normalize to [-pi, pi]
    
    # Reshape rotation difference to (batch, seq_len, 1)
    root_rotation_diff = root_rotation_diff[:, :, np.newaxis]  # (batch, seq_len, 1)
    
    # Concatenate all relationship features
    # Order: [x_diff, z_diff, y_diff, rotation_diff]
    # Note: x and z first (XZ plane), then y (height), then rotation
    relationship_features = np.concatenate([
        root_translation_diff[:, :, [0, 2, 1]],  # (batch, seq_len, 3) - [x, z, y]
        root_rotation_diff  # (batch, seq_len, 1)
    ], axis=-1)  # (batch, seq_len, 4)
    
    # Remove batch dimension if it wasn't there originally
    if not has_batch:
        relationship_features = relationship_features[0]  # (seq_len, 4)
    
    # Convert back to torch if input was torch
    if is_torch:
        relationship_features = torch.from_numpy(relationship_features).float()
        if original_device is not None:
            relationship_features = relationship_features.to(original_device)
    
    return relationship_features


# ============================================================================
# InterHuman Representation Conversion Functions
# ============================================================================
# These functions convert Salsa data to InterHuman representation format.
# Based on the conversion pipeline developed in Salsa_InterHuman_Representation.ipynb.
# Uses in2IN's functions for consistency with their preprocessing pipeline.
#
# References:
# - in2IN: https://github.com/... (add actual repo URL if available)
# - Functions borrowed from in2IN:
#   - process_motion_interhuman: in2in.utils.utils
#   - rigid_transform: in2in.utils.utils
#   - qmul_np, qinv_np, qrot_np: in2in.utils.quaternion

import os
import sys

# Add in2IN to path if not already there
# Assuming in2IN is in Download/in2IN relative to project root
# File is at: Salsa-Agent/motion_representation/utils/relationship_features.py
# Need to go up 4 levels to get to project root (New_2025)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
in2in_root = os.path.join(project_root, "Download", "in2IN")
if in2in_root not in sys.path:
    sys.path.insert(0, in2in_root)

try:
    from in2in.utils.utils import process_motion_interhuman, rigid_transform
    from in2in.utils.quaternion import qmul_np, qinv_np, qrot_np, qbetween_np
except ImportError as e:
    raise ImportError(
        f"Failed to import in2IN functions. Make sure in2IN is available at {in2in_root}. "
        f"Original error: {e}"
    )


def rotate_keypoints_deg_x(keypoints3d, rotation_deg):
    """
    Rotate 3D keypoints by rotation_deg degrees around X-axis.
    Uses the same rotX/transf logic as salsa_utils.py and salsa_to_interhuman for consistency.
    +90° reverses the -90° preprocessing rotation (front view for mesh rendering).

    Args:
        keypoints3d: (T, N, 3) or (T, 3*N) numpy array
        rotation_deg: float - degrees (e.g. 90 for front view)

    Returns:
        Rotated keypoints, same shape as input, float64.
    """
    if rotation_deg == 0:
        return np.asarray(keypoints3d, dtype=np.float64)
    import math
    orig_shape = np.array(keypoints3d).shape
    kp = np.asarray(keypoints3d, dtype=np.float64)
    if kp.ndim == 2:
        kp = kp.reshape(kp.shape[0], -1, 3)
    T = kp.shape[0]
    rotX = lambda theta: torch.tensor(
        [[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]],
        dtype=torch.float64,
    )

    def transf(rotMat, theta_deg, values):
        theta_rad = math.pi * torch.tensor(theta_deg, dtype=torch.float64) / 180.0
        return rotMat(theta_rad).mm(values.t()).t()

    kp_t = torch.from_numpy(kp)
    for frame_i in range(T):
        kp_t[frame_i] = transf(rotX, rotation_deg, kp_t[frame_i])
    out = kp_t.numpy()
    if len(orig_shape) == 2:
        out = out.reshape(orig_shape)
    return out


def salsa_to_interhuman(keypoints3d, rotmat, n_joints=22, rotation_deg=0):
    """
    Convert Salsa keypoints3d and rotmat to InterHuman representation.
    Uses the same approach as Step 5 that produced correct results.
    
    This function is based on the conversion pipeline from Salsa_InterHuman_Representation.ipynb.
    It uses in2IN's process_motion_interhuman() for canonicalization.
    
    Args:
        keypoints3d: (T, 22, 3) numpy array - Joint positions
        rotmat: (T, 498) numpy array - Rotation matrices [trans (3) | flattened_rotmats (55*9=495)]
        n_joints: int - Number of joints (default: 22 for InterHuman)
        rotation_deg: float - Rotation angle in degrees to apply around X-axis before conversion.
                            Default 0 (no rotation). Use +90 to reverse the -90° preprocessing rotation.
    
    Returns:
        motion_interhuman: (T-1, 262) numpy array - InterHuman processed motion
                           (after process_motion_interhuman, which reduces length by 1)
        root_quat_init: (T-1, 4) numpy array - Root quaternion at each frame
        root_pos_init: (T-1, 3) numpy array - Root position at each frame
    """
    T = keypoints3d.shape[0]
    
    # Step 0: Apply rotation if specified (to reverse preprocessing rotation)
    # keypoints3d in LMDB is rotated -90° for HumanML3D compatibility (salsa_utils.py lines 797-803).
    # To convert to InterHuman, we may need to reverse this rotation (apply +90°).
    # This uses the same rotation method as salsa_utils.py for consistency.
    if rotation_deg != 0:
        import torch
        import math
        
        # Rotation function matching salsa_utils.py (lines 150-155)
        rotX = lambda theta: torch.tensor([[1, 0, 0], 
                                            [0, torch.cos(theta), -torch.sin(theta)], 
                                            [0, torch.sin(theta), torch.cos(theta)]])
        def transf(rotMat, theta_deg, values):
            theta_rad = math.pi * torch.tensor(theta_deg).float() / 180.0
            return rotMat(theta_rad).mm(values.t()).t()
        
        # Convert to torch and apply rotation frame-by-frame (matching notebook lines 31-37)
        keypoints3d_torch = torch.tensor(keypoints3d).float()
        for frame_i in range(T):
            keypoints3d_torch[frame_i] = transf(rotX, rotation_deg, keypoints3d_torch[frame_i])
        keypoints3d = keypoints3d_torch.numpy()
    
    # Step 1: Extract positions from keypoints3d (after optional rotation)
    positions = keypoints3d.reshape(T, -1).astype(np.float32)  # (T, 66)
    
    # Step 2: Extract rotations from rotmat (skip first 3 dims which are translation)
    # Note: rotmat was NOT rotated during preprocessing, use as-is
    rotmats_flat = rotmat[:, 3:].astype(np.float32)  # (T, 495) - skip trans
    rotmats = rotmats_flat.reshape(T, 55, 3, 3)  # (T, 55, 3, 3)
    
    # Extract first 21 body joints (joints 1-21, excluding root joint 0)
    body_rotmats = rotmats[:, 1:22, :, :]  # (T, 21, 3, 3)
    
    # Convert each 3x3 matrix to 6D representation (first 2 columns)
    rotations_6d = body_rotmats[:, :, :, :2].reshape(T, 21, 6)  # (T, 21, 6)
    rotations_6d_flat = rotations_6d.reshape(T, -1)  # (T, 126)
    
    # Step 3: Build InterHuman format: [22*3 positions | 21*6 rotations]
    motion_raw = np.concatenate([positions, rotations_6d_flat], axis=-1)  # (T, 192)
    
    # Step 4: Process through in2IN's canonicalization pipeline (using in2IN's function)
    motion_interhuman, root_quat_init, root_pos_init = process_motion_interhuman(
        motion_raw, 0.001, 0, n_joints=n_joints
    )
    
    return motion_interhuman, root_quat_init, root_pos_init


def extract_interhuman_relationship_features(
    motion1_proc, motion2_proc, root_quat_init1, root_pos_init1, root_quat_init2, root_pos_init2,
    root_quat_init_L_all=None, root_pos_init_L_all=None, root_quat_init_F_all=None, root_pos_init_F_all=None,
    return_aligned_follower=False
):
    """
    Extract temporal relationship features from InterHuman representation.
    
    This function:
    1. Applies rigid_transform to person2 (aligning to person1's canonical frame using frame 0 transform)
    2. Extracts temporal relative features frame-by-frame from the shared space
    
    The approach matches in2IN's preprocessing (in2in/datasets/interhuman.py lines 210-215)
    but extracts temporal features instead of using a static transform.
    
    Args:
        motion1_proc: (T, 262) - Person 1 processed InterHuman motion (canonical frame)
        motion2_proc: (T, 262) - Person 2 processed InterHuman motion (canonical frame)
        root_quat_init1: (4,) or (1, 4) - Person 1 root quaternion from frame 0 (for frame 0 transform only)
        root_pos_init1: (3,) or (1, 3) - Person 1 root position from frame 0 (for frame 0 transform only)
        root_quat_init2: (4,) or (1, 4) - Person 2 root quaternion from frame 0 (for frame 0 transform only)
        root_pos_init2: (3,) or (1, 3) - Person 2 root position from frame 0 (for frame 0 transform only)
        root_quat_init_L_all: (T, 4) or None - Unused, kept for backward compatibility
        root_pos_init_L_all: (T, 3) or None - Unused, kept for backward compatibility
        root_quat_init_F_all: (T, 4) or None - Unused, kept for backward compatibility
        root_pos_init_F_all: (T, 3) or None - Unused, kept for backward compatibility
        return_aligned_follower: bool - If True, also return motion2_rel (aligned follower motion)
    
    Returns:
        relationship_features: (T, 4) numpy array - [w, z, x, z] for each frame
            - w, z: Quaternion components [w, z] representing relative rotation (yaw) at each frame
                   (matches in2IN's approach in losses.py line 79: pred_relative_rot[..., [0, 2]])
            - x, z: Relative position on XZ plane at each frame (person2 - person1 in shared space)
        motion2_rel: (T, 262) numpy array - Person 2 motion aligned to person1's canonical frame (only if return_aligned_follower=True)
    
    Reference:
        Based on in2IN's InterHuman dataset implementation:
        - in2in/datasets/interhuman.py, lines 210-215 (rigid_transform using angle for initial alignment)
        - in2in/models/utils/losses.py, lines 76-80 (uses quaternion components [w, z] instead of angle for loss)
        - in2in/models/utils/losses.py, lines 61-81 (hip-based forward direction computation)
        - in2in/utils/paramUtil.py: FACE_JOINT_INDX = [2, 1, 17, 16] (r_hip, l_hip, sdr_r, sdr_l)
    
    Note: Using quaternion components [w, z] instead of extracting angle (arctan2) avoids discontinuity
          at ±π and matches in2IN's training approach. See losses.py line 79: pred_relative_rot[..., [0, 2]]
    """
    T = motion1_proc.shape[0]
    n_joints = 22
    
    # Ensure root_quat_init and root_pos_init are 1D (they're from frame 0 only)
    if root_quat_init1.ndim > 1:
        root_quat_init1 = root_quat_init1[0]  # (4,)
    if root_pos_init1.ndim > 1:
        root_pos_init1 = root_pos_init1[0]  # (3,)
    if root_quat_init2.ndim > 1:
        root_quat_init2 = root_quat_init2[0]  # (4,)
    if root_pos_init2.ndim > 1:
        root_pos_init2 = root_pos_init2[0]  # (3,)
    
    # Step 1: Compute initial relative transform (frame 0 only) - same as in2IN's preprocessing
    # This is the transform that aligns person2 relative to person1
    # 
    # IMPORTANT: We use root quaternions for frame 0 (matches in2IN's preprocessing in interhuman.py line 211-212)
    # but frames 1+ use forward directions (matches in2IN's loss computation in losses.py line 76)
    # This is intentional: frame 0 must match the initial transform for proper alignment,
    # while frames 1+ use forward directions for consistency with in2IN's training loss.
    # 
    # Both methods produce [w, z] quaternion components, so the representation is consistent.
    r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))  # (4,) - relative quaternion from root quats
    
    # IMPORTANT: Quaternion angle conversion
    # For a yaw-only quaternion q = [cos(θ/2), 0, sin(θ/2), 0]:
    #   q[0] = cos(θ/2), q[2] = sin(θ/2)
    #   arctan2(q[2], q[0]) = arctan2(sin(θ/2), cos(θ/2)) = θ/2
    # So angle_init_half = θ/2 (half-angle)
    # rigid_transform expects the half-angle (it does cos(angle) and sin(angle) to create [cos(θ/2), 0, sin(θ/2), 0])
    angle_init_half = np.arctan2(r_relative[2], r_relative[0])  # scalar - half of relative yaw at frame 0
    
    xz_init = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[[0, 2]]  # (2,) - relative XZ at frame 0
    relative = np.array([angle_init_half, xz_init[0], xz_init[1]])  # (3,) - [yaw_half, x, z] at frame 0
    
    # Convert to quaternion components [w, z] for frame 0
    # We store [cos(θ/2), sin(θ/2)] where θ is the full yaw angle
    # This matches the format used for frames 1+ (from forward directions via qbetween)
    rel_w_frame0 = r_relative[0]  # cos(θ/2) - directly from relative quaternion (root-based)
    rel_z_frame0 = r_relative[2]  # sin(θ/2) - directly from relative quaternion (root-based)
    rel_x_frame0 = xz_init[0]
    rel_z_pos_frame0 = xz_init[1]
    
    # Step 2: Apply rigid_transform to person2 (same as in2IN)
    # This moves person2 into person1's canonical frame (for computing frames 1+)
    motion2_rel = rigid_transform(relative, motion2_proc)  # (T, 262)
    
    # VERIFICATION: Frame 0 relationship features must match initial transform exactly
    # This is the critical check - confirms that relationship features are computed correctly
    
    # Step 3: Extract temporal relationship features from aligned motions in shared space
    # After rigid_transform, both motions are in the same canonical frame (leader's space)
    # We compute relationship features frame-by-frame from the aligned motions (time-varying)
    
    # Initialize relationship features array
    relationship_features = np.zeros((T, 4), dtype=np.float32)
    
    # Frame 0: Use initial transform (already computed above) - matches in2IN preprocessing
    relationship_features[0, :] = [rel_w_frame0, rel_z_frame0, rel_x_frame0, rel_z_pos_frame0]
    
    # Frames 1+: Extract from aligned motions using same method as process_motion_interhuman
    # This matches in2IN preprocessing: compute root quaternions from forward directions, then relative quaternion
    # Extract root positions from aligned motions (both in leader's canonical frame)
    person1_root = motion1_proc[:, :3]  # (T, 3) - leader's root in canonical frame
    person2_root = motion2_rel[:, :3]   # (T, 3) - follower's root in leader's canonical frame
    
    # Compute relative XZ position frame-by-frame (in shared space)
    relative_xz = person2_root[:, [0, 2]] - person1_root[:, [0, 2]]  # (T, 2)
    
    # Extract hip joints for forward direction computation (same as process_motion_interhuman)
    # FACE_JOINT_INDX = [2, 1, 17, 16] (r_hip, l_hip, sdr_r, sdr_l)
    r_hip, l_hip = 2, 1  # Joint indices for right and left hip
    
    # Extract hip positions from aligned motions
    person1_r_hip = motion1_proc[:, r_hip*3:(r_hip+1)*3]  # (T, 3)
    person1_l_hip = motion1_proc[:, l_hip*3:(l_hip+1)*3]  # (T, 3)
    person2_r_hip = motion2_rel[:, r_hip*3:(r_hip+1)*3]   # (T, 3)
    person2_l_hip = motion2_rel[:, l_hip*3:(l_hip+1)*3]   # (T, 3)
    
    # Compute across vector (hip-based, same as process_motion_interhuman lines 119-120)
    person1_across = person1_r_hip - person1_l_hip  # (T, 3)
    person2_across = person2_r_hip - person2_l_hip  # (T, 3)
    
    # Normalize across vectors (same as process_motion_interhuman line 120)
    person1_across = person1_across / np.sqrt((person1_across ** 2).sum(axis=-1))[..., np.newaxis]  # (T, 3)
    person2_across = person2_across / np.sqrt((person2_across ** 2).sum(axis=-1))[..., np.newaxis]  # (T, 3)
    
    # Compute forward direction (same as process_motion_interhuman lines 123-125)
    y_axis = np.zeros_like(person1_across)  # (T, 3)
    y_axis[:, 1] = 1  # Y-axis is always [0, 1, 0]
    person1_forward = np.cross(y_axis, person1_across, axis=-1)  # (T, 3)
    person2_forward = np.cross(y_axis, person2_across, axis=-1)  # (T, 3)
    
    # Normalize forward vectors (same as process_motion_interhuman line 125)
    person1_forward = person1_forward / np.sqrt((person1_forward ** 2).sum(axis=-1))[..., np.newaxis]  # (T, 3)
    person2_forward = person2_forward / np.sqrt((person2_forward ** 2).sum(axis=-1))[..., np.newaxis]  # (T, 3)
    
    # Compute per-frame root quaternions from forward directions (same as process_motion_interhuman line 128)
    # This matches preprocessing: root_quat_init = qbetween_np(forward_init, [0,0,1])
    target = np.array([[0, 0, 1]])  # Target direction (Z+)
    root_quat1_per_frame = qbetween_np(person1_forward, target)  # (T, 4)
    root_quat2_per_frame = qbetween_np(person2_forward, target)  # (T, 4)
    
    # Compute relative quaternion from root quaternions (same as interhuman.py line 211)
    # This matches preprocessing: r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
    r_relative_per_frame = qmul_np(root_quat2_per_frame, qinv_np(root_quat1_per_frame))  # (T, 4)
    
    # Extract [w, z] components from relative quaternion (frames 1+)
    relative_rot_components = r_relative_per_frame[:, [0, 2]]  # (T, 2) - [w, z]
    
    # Fill relationship features for frames 1+
    relationship_features[1:, 0:2] = relative_rot_components[1:, :]  # [w, z] from root quaternions
    relationship_features[1:, 2:4] = relative_xz[1:, :]  # [x, z] from root positions
    
    # VERIFICATION: Check that frame 0 matches initial transform exactly
    # This is CRITICAL: Frame 0's relationship features must match the initial transform from root_quat_init and root_pos_init
    # We compute relationship features using the same method as in2IN preprocessing (root quaternions)
    # Frame 0 should match exactly because we use the same root_quat_init values
    rel_w_frame0_from_per_frame = relationship_features[0, 0]
    rel_z_frame0_from_per_frame = relationship_features[0, 1]
    rel_x_frame0_from_per_frame = relationship_features[0, 2]
    rel_z_pos_frame0_from_per_frame = relationship_features[0, 3]
    
    # Compare with initial transform (from root_quat_init frame 0)
    w_diff_frame0 = abs(rel_w_frame0 - rel_w_frame0_from_per_frame)
    z_diff_frame0 = abs(rel_z_frame0 - rel_z_frame0_from_per_frame)
    x_diff_frame0 = abs(rel_x_frame0 - rel_x_frame0_from_per_frame)
    z_pos_diff_frame0 = abs(rel_z_pos_frame0 - rel_z_pos_frame0_from_per_frame)
    
    # Also verify angle conversion for rigid_transform
    angle_from_per_frame = np.arctan2(rel_z_frame0_from_per_frame, rel_w_frame0_from_per_frame)
    angle_diff_frame0 = abs(angle_init_half - angle_from_per_frame)
    angle_diff_frame0_wrapped = abs(angle_diff_frame0 - 2 * np.pi) if angle_diff_frame0 > np.pi else angle_diff_frame0
    angle_diff_frame0 = min(angle_diff_frame0, angle_diff_frame0_wrapped)
    
    # Print organized verification summary (only if DEBUG is enabled)
    if DEBUG:
        print(f"\n{'='*70}")
        print(f"VERIFICATION: Frame 0 Relationship Features")
        print(f"{'='*70}")
        print(f"Method: Frame 0 uses root quaternions (same as in2IN preprocessing: interhuman.py lines 211-213)")
        print(f"  r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))")
        print(f"  xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[[0, 2]]")
        print(f"  Frames 1+ use same approach: forward directions -> root quaternions -> relative quaternion")
        print(f"  (matches process_motion_interhuman + interhuman.py preprocessing method)")
        print(f"\nFrame 0 Values:")
        print(f"  Stored:                     w={rel_w_frame0_from_per_frame:.6f}, z={rel_z_frame0_from_per_frame:.6f}, x={rel_x_frame0_from_per_frame:.6f}, z_pos={rel_z_pos_frame0_from_per_frame:.6f}")
        print(f"  From init quat (frame 0):   w={rel_w_frame0:.6f}, z={rel_z_frame0:.6f}, x={rel_x_frame0:.6f}, z_pos={rel_z_pos_frame0:.6f}")
        print(f"  Differences:                w={w_diff_frame0:.2e}, z={z_diff_frame0:.2e}, x={x_diff_frame0:.2e}, z_pos={z_pos_diff_frame0:.2e}")
        print(f"\nAngle Verification (for rigid_transform):")
        print(f"  From per-frame: {np.degrees(angle_from_per_frame):.3f} deg (half-angle)")
        print(f"  From init:      {np.degrees(angle_init_half):.3f} deg (half-angle)")
        print(f"  Difference:     {np.degrees(angle_diff_frame0):.3f} deg")
        
        threshold = 1e-4
        all_match = (w_diff_frame0 < threshold and z_diff_frame0 < threshold and 
                     x_diff_frame0 < threshold and z_pos_diff_frame0 < threshold)
        angle_match = angle_diff_frame0 < np.deg2rad(0.1)
        
        if all_match and angle_match:
            print(f"\n✓ VERIFICATION PASSED: Frame 0 matches initial transform EXACTLY")
            print(f"  Relationship features are computed correctly using root quaternions.")
            print(f"  When rigid_transform is applied, person2 will be correctly aligned to person1's shared space.")
        else:
            print(f"\n✗ VERIFICATION FAILED:")
            if not all_match:
                print(f"  - Component mismatch (w/z/x/z_pos differences > {threshold})")
            if not angle_match:
                print(f"  - Angle mismatch ({np.degrees(angle_diff_frame0):.2f} deg difference)")
        print(f"{'='*70}\n")
    
    # COMMENTED OUT: Previous method using forward directions from hips (matches in2IN's losses.py)
    # This method computed relative features from hip-based forward directions for frames 1+
    # Kept for reference in case we want to switch back
    """
    # Extract root positions from canonical frames
    person1_root = motion1_proc[:, :3]  # (T, 3) - person1's root in canonical frame
    person2_root = motion2_rel[:, :3]   # (T, 3) - person2's root in person1's canonical frame
    
    # Compute relative XZ position frame-by-frame
    relative_xz = person2_root[:, [0, 2]] - person1_root[:, [0, 2]]  # (T, 2)
    
    # Extract hip joints for forward direction computation (same as in2IN)
    # FACE_JOINT_INDX = [2, 1, 17, 16] (r_hip, l_hip, sdr_r, sdr_l)
    r_hip, l_hip = 2, 1  # Joint indices for right and left hip
    
    # Extract hip positions from canonical frames
    person1_r_hip = motion1_proc[:, r_hip*3:(r_hip+1)*3]  # (T, 3)
    person1_l_hip = motion1_proc[:, l_hip*3:(l_hip+1)*3]  # (T, 3)
    person2_r_hip = motion2_rel[:, r_hip*3:(r_hip+1)*3]   # (T, 3)
    person2_l_hip = motion2_rel[:, l_hip*3:(l_hip+1)*3]   # (T, 3)
    
    # Compute across vector (hip-based, same as in2IN)
    person1_across = person1_r_hip - person1_l_hip  # (T, 3)
    person2_across = person2_r_hip - person2_l_hip  # (T, 3)
    
    # Normalize across vectors (same as in2IN's utils.py lines 120 and 74 in skeleton.py)
    person1_across = person1_across / np.sqrt((person1_across ** 2).sum(axis=-1))[..., np.newaxis]  # (T, 3)
    person2_across = person2_across / np.sqrt((person2_across ** 2).sum(axis=-1))[..., np.newaxis]  # (T, 3)
    
    # Compute forward direction (same as in2IN: cross(y_axis, across))
    y_axis = np.zeros_like(person1_across)  # (T, 3)
    y_axis[:, 1] = 1  # Y-axis is always [0, 1, 0]
    person1_forward = np.cross(y_axis, person1_across, axis=-1)  # (T, 3)
    person2_forward = np.cross(y_axis, person2_across, axis=-1)  # (T, 3)
    
    # Normalize forward vectors
    person1_forward = person1_forward / np.sqrt((person1_forward ** 2).sum(axis=-1))[..., np.newaxis]  # (T, 3)
    person2_forward = person2_forward / np.sqrt((person2_forward ** 2).sum(axis=-1))[..., np.newaxis]  # (T, 3)
    
    # Compute temporal relative yaw using quaternion-based approach (matches in2IN's losses.py line 76)
    relative_quat = qbetween_np(person1_forward, person2_forward)  # (T, 4)
    relative_rot_components = relative_quat[:, [0, 2]]  # (T, 2) - [w, z] components
    
    # Build relationship features sequence: [w, z, x, z] = (T, 4)
    relationship_features = np.zeros((T, 4), dtype=np.float32)
    relationship_features[0, :] = [rel_w_frame0, rel_z_frame0, rel_x_frame0, rel_z_pos_frame0]  # Frame 0: exact initial transform
    relationship_features[1:, 0:2] = relative_rot_components[1:, :]  # Frames 1+: quaternion components [w, z]
    relationship_features[1:, 2:4] = relative_xz[1:, :]  # Frames 1+: position [x, z]
    """
    
    if return_aligned_follower:
        return relationship_features, motion2_rel
    else:
        return relationship_features


# ============================================================================
# FUTURE FEATURES TO IMPLEMENT (commented for reference):
# ============================================================================
#
# Additional spatial features:
# - Distance: Euclidean distance between root positions
# - Relative orientation: Angle between facing directions (yaw)
# - Relative rotation: Full 3D rotation matrix/quaternion from leader to follower frame
# - Lateral separation: Side-to-side distance (important for salsa)
# - Forward/backward offset: Front-back positioning
#
# Temporal/velocity features:
# - Relative velocity: Difference in root velocities
# - Relative angular velocity: Difference in rotation rates
# - Velocity alignment: Cosine similarity of velocity directions
# - Acceleration correlation: How movements accelerate together
#
# Orientation/pose features:
# - Facing angle: Angle between facing directions (0° = facing each other, 180° = back-to-back)
# - Body alignment: Relative torso orientations
# - Hand connection features: If hands are connected, relative hand positions
# - Mirroring score: How much movements mirror each other
#
# Interaction-specific features:
# - Lead-follow signal: Who is initiating movement (can be derived from velocity/acceleration)
# - Synchronization offset: Temporal lag between similar movements
# - Complementary motion: How movements complement (e.g., one steps forward, other steps back)
# - Shared rotation axis: If rotating together, the axis of rotation
#
# Normalized/relative features:
# - Normalized distance: Distance divided by average arm span or body height
# - Relative frame features: All features in leader-centric or follower-centric coordinate frame
# - Dance phase features: Position in the dance pattern (beginning, middle, end of a move)
#
# ============================================================================

