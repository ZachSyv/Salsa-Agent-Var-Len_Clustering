"""
Utility functions for extracting relationship features between leader and follower dancers.
"""

import torch
import numpy as np
from typing import Union, Tuple

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

