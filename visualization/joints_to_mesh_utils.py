"""
Joints (3D keypoints) → SMPL mesh visualization for two-person dance.
Uses priorMDM joints2smpl + Rotation2xyz and MotionScript HumorRenderer.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Headless Linux: use EGL before pyrender is imported (pyrender chooses platform at import time)
if sys.platform == "linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Salsa-Agent root
_PARENT_DIR = Path(__file__).parent.parent
_PRIOR_MDM_DIR = _PARENT_DIR / "utils" / "salsa_utils" / "libs" / "priorMDM"
_MOTION_SCRIPT_DIR = _PARENT_DIR / "utils" / "salsa_utils" / "libs" / "MotionScript" / "stmc_renderer"

# Candidate SMPL model dirs (folder containing SMPL_NEUTRAL.pkl or equivalent)
_SMPL_CANDIDATES = [
    _PARENT_DIR / "body_model" / "smpl",
    _PARENT_DIR / "body_models" / "smpl",
    _PRIOR_MDM_DIR / "body_models" / "smpl",
    _PRIOR_MDM_DIR / "body_models",
]


def _resolve_smpl_model_dir() -> Optional[Path]:
    """Return first candidate path that exists and contains an SMPL .pkl file.
    Do not return a path that only exists as a directory (e.g. body_models without smpl/),
    since smplx then looks for path/smpl and fails.
    """
    for p in _SMPL_CANDIDATES:
        if not p.is_dir():
            continue
        if (p / "SMPL_NEUTRAL.pkl").exists() or (p / "basicModel_neutral.pkl").exists():
            return p
        if any(p.glob("*.pkl")):
            return p
    return None


def keypoints_to_smpl_vertices_and_faces(
    keypoints: np.ndarray,
    device_id: int = 0,
    cuda: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convert 3D keypoints (T, 22, 3) to SMPL mesh vertices (T, V, 3) and faces (F, 3).
    Uses priorMDM joints2smpl + Rotation2xyz. Requires priorMDM body_models (SMPL) and
    joints2smpl smpl_models to be set up under priorMDM.

    Returns:
        (vertices, faces) or (None, None) on failure.
    """
    if keypoints is None or keypoints.size == 0:
        return None, None
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1] != 22 or keypoints.shape[2] != 3:
        return None, None

    T = keypoints.shape[0]
    cwd = os.getcwd()
    prior_mdm_str = str(_PRIOR_MDM_DIR)
    if prior_mdm_str not in sys.path:
        sys.path.insert(0, prior_mdm_str)

    try:
        os.chdir(prior_mdm_str)
        # NumPy 2 compatibility: chumpy does "from numpy import bool, int, ..." which fails on NumPy 2.
        import numpy as _np
        if not hasattr(_np, "bool"):
            _np.bool = bool
            _np.int = int
            _np.float = float
            _np.complex = complex
            _np.object = object
            _np.str = str
            _np.unicode = str
        smpl_dir = _resolve_smpl_model_dir()
        if smpl_dir is None:
            raise FileNotFoundError(
                "SMPL model dir not found (no candidate path contains SMPL_NEUTRAL.pkl or *.pkl). "
                "Tried: " + ", ".join(str(p) for p in _SMPL_CANDIDATES) + ". "
                "Create one of these folders and put SMPL_NEUTRAL.pkl there (e.g. priorMDM/body_models/smpl/ or Salsa-Agent/body_model/smpl/). "
                "See priorMDM body_models/README.md and priorMDM README for download links."
            )
        smpl_dir_str = str(smpl_dir.resolve())
        import visualize.joints2smpl.src.config as j2s_config
        j2s_config.SMPL_MODEL_DIR = smpl_dir_str
        import utils.config as prior_utils_config
        prior_utils_config.SMPL_DATA_PATH = smpl_dir_str
        prior_utils_config.SMPL_MODEL_PATH = os.path.join(smpl_dir_str, "SMPL_NEUTRAL.pkl")
        prior_utils_config.JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(smpl_dir_str, "J_regressor_extra.npy")
        from visualize.simplify_loc2rot import joints2smpl
        from model.rotation2xyz import Rotation2xyz
        import torch

        gmm_dir = _PRIOR_MDM_DIR / "visualize" / "joints2smpl" / "smpl_models"
        gmm_file = gmm_dir / "gmm_08.pkl"
        if not gmm_file.exists():
            raise FileNotFoundError(
                f"GMM prior file not found: {gmm_file}\n"
                "Download gmm_08.pkl from VIBE/SPIN (e.g. VIBE data or SPIN smplify prior) "
                "and place in priorMDM/visualize/joints2smpl/smpl_models/"
            )
        j2s = joints2smpl(num_frames=T, device_id=device_id, cuda=cuda and torch.cuda.is_available())
        motion_tensor, _ = j2s.joint2smpl(keypoints)
        motion_tensor = motion_tensor.cpu()
        rot2xyz = Rotation2xyz(device="cpu")
        vertices = rot2xyz(
            motion_tensor,
            mask=None,
            pose_rep="rot6d",
            translation=True,
            glob=True,
            jointstype="vertices",
            vertstrans=True,
        )
        verts_np = vertices[0].permute(2, 0, 1).numpy()
        faces = rot2xyz.smpl_model.faces
        if hasattr(faces, "numpy"):
            faces_np = faces.numpy()
        else:
            faces_np = np.array(faces)
        return verts_np, faces_np
    except SystemExit as e:
        if os.getcwd() != cwd:
            os.chdir(cwd)
        raise RuntimeError(
            "keypoints_to_smpl failed (prior called sys.exit). "
            "Likely missing gmm_08.pkl in priorMDM/visualize/joints2smpl/smpl_models/. "
            "See joints2smpl README or VIBE/SPIN for GMM prior download."
        ) from e
    except Exception as e:
        if os.getcwd() != cwd:
            os.chdir(cwd)
        raise RuntimeError(f"keypoints_to_smpl failed: {e}") from e
    finally:
        if os.getcwd() != cwd:
            os.chdir(cwd)


def render_two_person_mesh(
    leader_vertices: np.ndarray,
    follower_vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str,
    fps: int = 20,
) -> Optional[str]:
    """
    Render two SMPL mesh sequences to a single video using HumorRenderer (pyrender).
    leader_vertices, follower_vertices: (T, V, 3). faces: (F, 3).
    Uses EGL for headless rendering when DISPLAY is not available (e.g. Ubuntu server).
    """
    if leader_vertices is None or follower_vertices is None or faces is None:
        return None
    if sys.platform == "linux":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    leader_vertices = np.asarray(leader_vertices, dtype=np.float64)
    follower_vertices = np.asarray(follower_vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    motion_script_dir = str(_MOTION_SCRIPT_DIR.parent)
    if motion_script_dir not in sys.path:
        sys.path.insert(0, motion_script_dir)

    from stmc_renderer.humor import HumorRenderer
    from smplx.utils import Struct
    import torch

    renderer = HumorRenderer(fps=fps, imw=720, imh=720)
    renderer.two_people(
        leader_vertices,
        faces,
        follower_vertices,
        faces,
        output=output_path,
    )
    return output_path if os.path.isfile(output_path) else None


def keypoints_to_mesh_video(
    leader_keypoints: np.ndarray,
    follower_keypoints: np.ndarray,
    output_path: str,
    fps: int = 20,
    device_id: int = 0,
    view_rotation_deg_x: float = 90,
) -> Optional[str]:
    """
    Full pipeline: (leader_kp, follower_kp) → SMPL fit → two-person mesh video.
    keypoints: (T, 22, 3) each. Returns output_path on success, None on failure.
    view_rotation_deg_x: rotation in degrees around x before SMPL fit (e.g. 90 for front view).
    """
    from motion_representation.utils.relationship_features import rotate_keypoints_deg_x

    leader_kp = rotate_keypoints_deg_x(leader_keypoints, view_rotation_deg_x)
    follower_kp = rotate_keypoints_deg_x(follower_keypoints, view_rotation_deg_x)
    verts_L, faces = keypoints_to_smpl_vertices_and_faces(
        leader_kp, device_id=device_id, cuda=True
    )
    if verts_L is None or faces is None:
        return None
    verts_F, _ = keypoints_to_smpl_vertices_and_faces(
        follower_kp, device_id=device_id, cuda=True
    )
    if verts_F is None:
        return None
    min_frames = min(verts_L.shape[0], verts_F.shape[0])
    verts_L = verts_L[:min_frames]
    verts_F = verts_F[:min_frames]
    return render_two_person_mesh(verts_L, verts_F, faces, output_path, fps=fps)
