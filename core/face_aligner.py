"""3D Pose-Aware Face Frontalizer & Homography Warping for High-Angle Invariance."""
from typing import Optional, Tuple
import cv2
import numpy as np

from config.settings import ModelSettings, get_settings
from core.pose_estimator import HeadPose
from utils.logger import get_logger

logger = get_logger("FaceAligner")


class FaceAligner:
    """Computes perspective homography transform H = K * R^-1 * K^-1 and performs canonical frontal warping."""

    def __init__(self, model_settings: Optional[ModelSettings] = None):
        self.settings = model_settings or get_settings().models
        self.crop_w, self.crop_h = self.settings.canonical_crop_size

    def compute_homography(self, pose: HeadPose, img_w: int, img_h: int) -> np.ndarray:
        """Compute the homography matrix H = K * R^-1 * K^-1 for 3D perspective rectification."""
        focal_length = float(img_w) * self.settings.focal_length_scale
        cx, cy = img_w / 2.0, img_h / 2.0
        k_mat = np.array([
            [focal_length, 0.0, cx],
            [0.0, focal_length, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        k_inv = np.linalg.inv(k_mat)
        r_inv = pose.rotation_matrix.T  # Inverse of rotation matrix R is its transpose

        h_mat = k_mat @ r_inv @ k_inv
        return h_mat

    def frontalize_and_crop(
        self,
        frame_bgr: np.ndarray,
        bbox_pixel: Tuple[int, int, int, int],
        pose: Optional[HeadPose] = None
    ) -> np.ndarray:
        """Frontalize and extract canonical 112x112 face crop.

        If head pose is available, applies 3D homography un-warping; otherwise uses similarity/bounding crop.
        """
        img_h, img_w, _ = frame_bgr.shape
        xmin, ymin, xmax, ymax = bbox_pixel

        # Bound safety padding
        pad_x = int((xmax - xmin) * 0.1)
        pad_y = int((ymax - ymin) * 0.1)
        x0 = max(0, xmin - pad_x)
        y0 = max(0, ymin - pad_y)
        x1 = min(img_w, xmax + pad_x)
        y1 = min(img_h, ymax + pad_y)

        if pose is not None:
            try:
                h_mat = self.compute_homography(pose, img_w, img_h)
                rectified_frame = cv2.warpPerspective(
                    frame_bgr,
                    h_mat,
                    (img_w, img_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                crop = rectified_frame[y0:y1, x0:x1]
                if crop.size > 0:
                    return cv2.resize(crop, (self.crop_w, self.crop_h), interpolation=cv2.INTER_AREA)
            except Exception as e:
                logger.warning(f"Frontalization warp failed: {e}. Fallback to regular crop.")

        # Fallback to direct bounding box crop
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0 or (x1 <= x0) or (y1 <= y0):
            # Emergency blank crop
            return np.zeros((self.crop_h, self.crop_w, 3), dtype=np.uint8)

        return cv2.resize(crop, (self.crop_w, self.crop_h), interpolation=cv2.INTER_AREA)
