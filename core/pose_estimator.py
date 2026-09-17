"""3D Head Pose Estimator using OpenCV solvePnP and canonical 3D facial landmarks."""
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger("PoseEstimator")

# Canonical 3D facial model points (in mm/canonical frame with camera-aligned conventions)
# Camera frame: +X points Right, +Y points Down (towards Chin), -Z points Away from Camera (recessed features)
# Key facial landmarks:
# 1: Nose tip
# 152: Chin
# 263: Left eye outer corner (Anatomical Left = Image Right -> +X)
# 33: Right eye outer corner (Anatomical Right = Image Left -> -X)
# 291: Left mouth corner (Anatomical Left = Image Right -> +X)
# 61: Right mouth corner (Anatomical Right = Image Left -> -X)
CANONICAL_FACE_3D = np.array([
    [0.0, 0.0, 0.0],          # 1: Nose tip
    [0.0, 100.0, -25.0],      # 152: Chin
    [60.0, -40.0, -35.0],     # 263: Left eye outer corner (Image Right: +X)
    [-60.0, -40.0, -35.0],    # 33: Right eye outer corner (Image Left: -X)
    [35.0, 50.0, -20.0],      # 291: Left mouth corner (Image Right: +X)
    [-35.0, 50.0, -20.0]      # 61: Right mouth corner (Image Left: -X)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
CANONICAL_LANDMARK_INDICES = [1, 152, 263, 33, 291, 61]


@dataclass
class HeadPose:
    """Estimated 3D Head Pose orientation and translation."""
    pitch: float  # Up/Down rotation in degrees (Negative = Look Up, Positive = Look Down)
    yaw: float    # Left/Right rotation in degrees (Negative = Turn Left, Positive = Turn Right)
    roll: float   # Tilt rotation in degrees (Positive = Tilt Clockwise, Negative = Tilt Counter-Clockwise)
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

            # Deconstruct rotation matrix into Euler angles (Pitch, Yaw, Roll) via RQ decomposition
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)

            pitch = float(angles[0])
            yaw = float(angles[1])
            roll = float(angles[2])

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

    @staticmethod
    def quantize_pose_to_grid(
        yaw: float,
        pitch: float,
        n_cols: int = 5,
        n_rows: int = 3,
        yaw_range: Tuple[float, float] = (-40.0, 40.0),
        pitch_range: Tuple[float, float] = (-20.0, 20.0)
    ) -> Tuple[int, int]:
        """Maps continuous Euler angles (yaw, pitch) to discrete (row, col) grid cell indices.

        Args:
            yaw: Continuous yaw angle in degrees.
            pitch: Continuous pitch angle in degrees.
            n_cols: Number of horizontal pose columns.
            n_rows: Number of vertical pose rows.
            yaw_range: (min_yaw, max_yaw) bounds.
            pitch_range: (min_pitch, max_pitch) bounds.

        Returns:
            Tuple[int, int]: Clamped (row, col) grid coordinate.
        """
        psi_min, psi_max = yaw_range
        theta_min, theta_max = pitch_range

        # When user turns head left (yaw < 0), their right side faces the camera (screen right, higher col).
        # When user turns head right (yaw > 0), their left side faces the camera (screen left, lower col).
        norm_yaw = (psi_max - yaw) / max(psi_max - psi_min, 1e-6)
        col = int(np.clip(np.floor(norm_yaw * n_cols), 0, n_cols - 1))

        norm_pitch = (pitch - theta_min) / max(theta_max - theta_min, 1e-6)
        row = int(np.clip(np.floor(norm_pitch * n_rows), 0, n_rows - 1))

        return row, col


def get_grid_cell(
    yaw: float,
    pitch: float,
    n_cols: int = 5,
    n_rows: int = 3,
    yaw_range: Tuple[float, float] = (-40.0, 40.0),
    pitch_range: Tuple[float, float] = (-20.0, 20.0)
) -> Tuple[int, int]:
    """Convenience functional wrapper for PoseEstimator.quantize_pose_to_grid."""
    return PoseEstimator.quantize_pose_to_grid(
        yaw=yaw,
        pitch=pitch,
        n_cols=n_cols,
        n_rows=n_rows,
        yaw_range=yaw_range,
        pitch_range=pitch_range
    )

