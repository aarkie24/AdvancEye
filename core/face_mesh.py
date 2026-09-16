"""MediaPipe FaceMesh engine & Geometrically Invariant Topological Grid Mapper."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from config.settings import ModelSettings, get_settings
from utils.logger import get_logger

logger = get_logger("FaceMeshEngine")


@dataclass
class FaceDetectionResult:
    """Structured result of single face landmark extraction & grid binding."""
    landmarks_3d: np.ndarray  # Shape: (468, 3) normalized [0, 1]
    landmarks_pixel: np.ndarray  # Shape: (468, 2) in image pixel coordinates
    bbox_pixel: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    grid_cells: List[Tuple[int, int]]  # List of (row, col) per landmark index
    grid_matrix: np.ndarray  # Shape: (grid_rows, grid_cols) count/mask of landmarks


class FaceMeshEngine:
    """Extracts 468 3D landmarks and performs topological grid cell binding."""

    def __init__(self, model_settings: Optional[ModelSettings] = None):
        self.settings = model_settings or get_settings().models
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh_detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.settings.max_num_faces,
            refine_landmarks=self.settings.refine_landmarks,
            min_detection_confidence=self.settings.min_detection_confidence,
            min_tracking_confidence=self.settings.min_tracking_confidence
        )

    def process_frame(self, frame_bgr: np.ndarray) -> List[FaceDetectionResult]:
        """Process BGR image frame and return landmark & topological grid data for all detected faces."""
        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.mesh_detector.process(rgb_frame)

        if not results.multi_face_landmarks:
            return []

        face_results: List[FaceDetectionResult] = []
        for face_landmarks in results.multi_face_landmarks:
            landmarks_norm = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark], dtype=np.float32)
            landmarks_pix = np.zeros((len(face_landmarks.landmark), 2), dtype=np.int32)
            landmarks_pix[:, 0] = np.clip(landmarks_norm[:, 0] * w, 0, w - 1).astype(np.int32)
            landmarks_pix[:, 1] = np.clip(landmarks_norm[:, 1] * h, 0, h - 1).astype(np.int32)

            # Section 4.1 Bounding box computation in normalized image space
            x_min = np.min(landmarks_norm[:, 0])
            x_max = np.max(landmarks_norm[:, 0])
            y_min = np.min(landmarks_norm[:, 1])
            y_max = np.max(landmarks_norm[:, 1])

            bbox_pix = (
                int(np.clip(x_min * w, 0, w - 1)),
                int(np.clip(y_min * h, 0, h - 1)),
                int(np.clip(x_max * w, 0, w - 1)),
                int(np.clip(y_max * h, 0, h - 1))
            )

            # Section 4.1 Topological Grid Mapping
            grid_cells, grid_mat = self.map_landmarks_to_grid(
                landmarks_norm[:, :2], x_min, x_max, y_min, y_max,
                self.settings.grid_rows, self.settings.grid_cols, self.settings.epsilon
            )

            face_results.append(FaceDetectionResult(
                landmarks_3d=landmarks_norm,
                landmarks_pixel=landmarks_pix,
                bbox_pixel=bbox_pix,
                grid_cells=grid_cells,
                grid_matrix=grid_mat
            ))

        return face_results

    @staticmethod
    def map_landmarks_to_grid(
        landmarks_xy: np.ndarray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        n_rows: int,
        n_cols: int,
        epsilon: float = 1e-6
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Mathematically binds landmarks to topological (r, c) grid cells using Section 4.1 equations."""
        dx = max(x_max - x_min, epsilon)
        dy = max(y_max - y_min, epsilon)

        u = (landmarks_xy[:, 0] - x_min) / dx
        v = (landmarks_xy[:, 1] - y_min) / dy

        c = np.clip(np.floor(u * n_cols).astype(int), 0, n_cols - 1)
        r = np.clip(np.floor(v * n_rows).astype(int), 0, n_rows - 1)

        grid_cells = list(zip(r.tolist(), c.tolist()))
        grid_matrix = np.zeros((n_rows, n_cols), dtype=np.int32)
        for row, col in grid_cells:
            grid_matrix[row, col] += 1

        return grid_cells, grid_matrix

    def close(self) -> None:
        self.mesh_detector.close()
