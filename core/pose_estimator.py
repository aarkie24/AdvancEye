"""3D Head Pose Estimator using OpenCV solvePnP and canonical 3D facial landmarks."""
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger("PoseEstimator")

# Canonical 3D facial model points (in mm/arbitrary canonical frame)
# Key facial landmarks: Nose tip, Chin, Left eye outer corner, Right eye outer corner, Left mouth corner, Right mouth corner
CANONICAL_FACE_3D = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (landmark #1 or #4)
    [0.0, -330.0, -65.0],     # Chin (landmark #152)
    [-225.0, 170.0, -135.0],  # Left eye outer corner (landmark #263)
    [225.0, 170.0, -135.0],   # Right eye outer corner (landmark #33)
    [-150.0, -150.0, -125.0], # Left mouth corner (landmark #291)
    [150.0, -150.0, -125.0]   # Right mouth corner (landmark #61)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
CANONICAL_LANDMARK_INDICES = [1, 152, 263, 33, 291, 61]


@dataclass
class HeadPose:
    """Estimated 3D Head Pose orientation and translation."""
    pitch: float  # Up/Down rotation in degrees
    yaw: float    # Left/Right rotation in degrees
    roll: float   # Tilt rotation in degrees
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    rotation_matrix: np.ndarray


class PoseEstimator:
    """Estimates head pose (pitch, yaw, roll) using OpenCV solvePnP."""

    def __init__(self, camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1), dtype=np.float64)

    def _build_default_camera_matrix(self, img_w: int, img_h: int) -> np.ndarray:
        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        return np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

    def estimate_pose(self, landmarks_2d: np.ndarray, img_w: int, img_h: int) -> Optional[HeadPose]:
        """Estimate 3D head pose from 2D facial landmark coordinates.

        Args:
            landmarks_2d: (468, 2) or (N, 2) array of pixel coordinates.
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            HeadPose dataclass instance or None if solvePnP fails.
        """
        try:
            if landmarks_2d.shape[0] >= 468:
                image_points = landmarks_2d[CANONICAL_LANDMARK_INDICES].astype(np.float64)
            elif landmarks_2d.shape[0] == len(CANONICAL_LANDMARK_INDICES):
                image_points = landmarks_2d.astype(np.float64)
            else:
                logger.warning(f"Unexpected landmarks shape: {landmarks_2d.shape}")
                return None

            cam_matrix = self.camera_matrix if self.camera_matrix is not None else self._build_default_camera_matrix(img_w, img_h)

            success, rot_vec, trans_vec = cv2.solvePnP(
                CANONICAL_FACE_3D,
                image_points,
                cam_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                logger.warning("cv2.solvePnP failed to converge.")
                return None

            rot_mat, _ = cv2.Rodrigues(rot_vec)

            # Deconstruct rotation matrix into Euler angles (Pitch, Yaw, Roll)
            proj_mat = np.hstack((rot_mat, trans_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)

            pitch = float(euler_angles[0, 0])
            yaw = float(euler_angles[1, 0])
            roll = float(euler_angles[2, 0])

            return HeadPose(
                pitch=pitch,
                yaw=yaw,
                roll=roll,
                rotation_vector=rot_vec,
                translation_vector=trans_vec,
                rotation_matrix=rot_mat
            )
        except Exception as e:
            logger.error(f"Error estimating head pose: {e}")
            return None
