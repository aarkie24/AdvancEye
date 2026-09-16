"""Student registration state machine and multi-pose 3D topological capture."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from config.settings import AppSettings, get_settings
from core.face_mesh import FaceDetectionResult, FaceMeshEngine
from core.pose_estimator import HeadPose, PoseEstimator
from core.face_aligner import FaceAligner
from core.face_embedder import FaceEmbedder
from storage.base import BaseRepository
from utils.logger import get_logger

logger = get_logger("RegistrationService")


class RegistrationService:
    """Manages student enrollment, multi-angle pose guidance, and cell embedding accumulation."""

    def __init__(
        self,
        repository: BaseRepository,
        face_mesh: Optional[FaceMeshEngine] = None,
        pose_estimator: Optional[PoseEstimator] = None,
        face_aligner: Optional[FaceAligner] = None,
        face_embedder: Optional[FaceEmbedder] = None,
        settings: Optional[AppSettings] = None
    ):
        self.repo = repository
        self.settings = settings or get_settings()
        self.face_mesh = face_mesh or FaceMeshEngine(self.settings.models)
        self.pose_estimator = pose_estimator or PoseEstimator()
        self.face_aligner = face_aligner or FaceAligner(self.settings.models)
        self.face_embedder = face_embedder or FaceEmbedder(self.settings.models)

        # Active enrollment session state
        self.is_enrolling: bool = False
        self.roll_no: str = ""
        self.name: str = ""
        self.collected_cells: Dict[str, List[float]] = {}
        self.canonical_cells: List[Dict[str, int]] = []

    def start_session(self, roll_no: str, name: str) -> None:
        """Initialize enrollment session for a student."""
        self.roll_no = roll_no.strip()
        self.name = name.strip()
        self.collected_cells = {}
        self.canonical_cells = []
        self.is_enrolling = True
        logger.info(f"Started enrollment session for {self.name} ({self.roll_no})")

    def process_enrollment_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[FaceDetectionResult], Optional[HeadPose], Dict[str, Any]]:
        """Process video frame during enrollment, extracting pose and capturing embeddings."""
        if not self.is_enrolling:
            return None, None, {"status": "NOT_ENROLLING"}

        h, w, _ = frame_bgr.shape
        detections = self.face_mesh.process_frame(frame_bgr)
        if not detections:
            return None, None, {"status": "NO_FACE_DETECTED", "captured_cells": len(self.collected_cells)}

        # Focus on primary detected face
        face = detections[0]
        pose = self.pose_estimator.estimate_pose(face.landmarks_pixel, w, h)

        # Frontalize and extract embedding
        crop = self.face_aligner.frontalize_and_crop(frame_bgr, face.bbox_pixel, pose)
        embedding = self.face_embedder.extract_embedding(crop)

        if embedding is not None:
            # Map canonical landmarks to cells and store embedding under key "r_c"
            for idx, (r, c) in enumerate(face.grid_cells):
                cell_key = f"{r}_{c}"
                if cell_key not in self.collected_cells:
                    self.collected_cells[cell_key] = embedding.tolist()
                    self.canonical_cells.append({"landmark_id": idx, "row": r, "col": c})

        status_info = {
            "status": "CAPTURING",
            "captured_cells": len(self.collected_cells),
            "target_cells": self.settings.models.grid_rows * self.settings.models.grid_cols,
            "pitch": pose.pitch if pose else 0.0,
            "yaw": pose.yaw if pose else 0.0,
            "roll": pose.roll if pose else 0.0
        }

        return face, pose, status_info

    def finalize_and_save(self) -> bool:
        """Persist enrolled student profile into repository."""
        if not self.is_enrolling or not self.roll_no:
            logger.error("No active enrollment session to finalize.")
            return False

        if len(self.collected_cells) == 0:
            logger.warning(f"No cell embeddings collected for {self.roll_no}.")
            return False

        profile_data = {
            "roll_no": self.roll_no,
            "name": self.name,
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
            "total_cells_captured": len(self.collected_cells),
            "canonical_landmark_cells": self.canonical_cells[:100],  # Bound length for clean JSON
            "embeddings": self.collected_cells
        }

        success = self.repo.save_profile(profile_data)
        if success:
            logger.info(f"Successfully finalized and saved profile: {self.roll_no}")
            self.is_enrolling = False
            return True

        return False
