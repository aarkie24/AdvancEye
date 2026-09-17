"""Dynamic registration service with pose node stability, Laplacian sharpness gating, and ArcFace capture."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from config.settings import AppSettings, get_settings
from core.face_aligner import FaceAligner
from core.face_embedder import FaceEmbedder
from core.face_mesh import FaceDetectionResult, FaceMeshEngine
from core.mesh_types import NodeStatus, RegistrationSessionState
from core.pose_estimator import HeadPose, PoseEstimator, get_grid_cell
from storage.base import BaseRepository
from utils.logger import get_logger

logger = get_logger("DynamicRegistrationService")


class DynamicRegistrationService:
    """Manages pose node transitions, biometric stability, and embedding capture."""

    STABLE_FRAMES_REQUIRED: int = 8
    MIN_SHARPNESS_LAPLACIAN: float = 60.0
    COMPLETION_THRESHOLD_PCT: float = 85.0

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

        # Dynamic registration configuration derived from settings
        reg_cfg = getattr(self.settings, "registration", None)
        self.stable_frames_required: int = getattr(reg_cfg, "stable_frames_required", 8)
        self.min_sharpness_laplacian: float = getattr(reg_cfg, "min_sharpness_laplacian", 60.0)
        self.completion_threshold_pct: float = getattr(reg_cfg, "completion_threshold_pct", 85.0)

        self.state: Optional[RegistrationSessionState] = None
        self.last_cell: Optional[Tuple[int, int]] = None
        self.stable_counter: int = 0
        self.is_enrolling: bool = False

    def start_session(self, roll_no: str, name: str) -> RegistrationSessionState:
        """Initialize a new registration session state."""
        reg_cfg = getattr(self.settings, "registration", None)
        model_cfg = getattr(self.settings, "models", None)

        self.state = RegistrationSessionState(
            roll_no=roll_no.strip(),
            name=name.strip(),
            grid_rows=getattr(reg_cfg, "grid_rows", getattr(model_cfg, "grid_rows", 3)),
            grid_cols=getattr(reg_cfg, "grid_cols", getattr(model_cfg, "grid_cols", 5)),
            yaw_range=getattr(reg_cfg, "yaw_range", getattr(model_cfg, "yaw_range", (-50.0, 50.0))),
            pitch_range=getattr(reg_cfg, "pitch_range", getattr(model_cfg, "pitch_range", (-30.0, 30.0)))
        )
        self.last_cell = None
        self.stable_counter = 0
        self.is_enrolling = True
        logger.info(f"Started biometric registration session for {self.state.name} ({self.state.roll_no})")
        return self.state

    def process_frame(
        self,
        frame_bgr: np.ndarray
    ) -> Tuple[Optional[FaceDetectionResult], Optional[HeadPose], RegistrationSessionState, int]:
        """Process one video frame during enrollment.

        Returns:
            Tuple[detection, pose, state, stability_counter]
        """
        if not self.is_enrolling or self.state is None:
            return None, None, self.state, 0

        h, w, _ = frame_bgr.shape
        detections = self.face_mesh.process_frame(frame_bgr)
        if not detections:
            self.state.active_cell = None
            return None, None, self.state, 0

        face = detections[0]
        pose = self.pose_estimator.estimate_pose(face.landmarks_pixel, w, h)
        if not pose:
            self.state.active_cell = None
            return face, None, self.state, 0

        # Quantize Euler angles to discrete grid cell
        cell = get_grid_cell(
            yaw=pose.yaw,
            pitch=pose.pitch,
            n_cols=self.state.grid_cols,
            n_rows=self.state.grid_rows,
            yaw_range=self.state.yaw_range,
            pitch_range=self.state.pitch_range
        )
        self.state.active_cell = cell
        node = self.state.nodes[cell]

        if node.status == NodeStatus.LOCKED:
            # Angle already captured and locked
            self.stable_counter = 0
            return face, pose, self.state, 0

        # Check stability across consecutive frames
        if cell == self.last_cell:
            self.stable_counter += 1
            node.status = NodeStatus.CAPTURING
        else:
            self.stable_counter = 1
            node.status = NodeStatus.CAPTURING

        self.last_cell = cell


        # Gate checks: Temporal stability + Laplacian Sharpness + Embedding Extraction
        if self.stable_counter >= self.stable_frames_required:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            if sharpness >= self.min_sharpness_laplacian:
                crop = self.face_aligner.frontalize_and_crop(frame_bgr, face.bbox_pixel, pose)
                embedding = self.face_embedder.extract_embedding(crop)

                if embedding is not None:
                    node.embedding = embedding
                    node.status = NodeStatus.LOCKED  # Permanently turn green
                    logger.info(f"Pose node {cell} LOCKED for {self.state.roll_no}")

                    self.state.locked_count = sum(
                        1 for n in self.state.nodes.values() if n.status == NodeStatus.LOCKED
                    )
                    self.state.coverage_pct = (self.state.locked_count / self.state.total_cells) * 100.0
                    self.state.is_ready_to_save = (self.state.coverage_pct >= self.completion_threshold_pct)
                    self.stable_counter = 0

        return face, pose, self.state, self.stable_counter

    def finalize_and_save(self) -> bool:
        """Serializes and saves profile according to Section 6 JSON schema."""
        if not self.is_enrolling or self.state is None:
            logger.error("No active session to save.")
            return False

        if self.state.locked_count == 0:
            logger.warning("Cannot save profile: no nodes locked.")
            return False

        embeddings_dict = {}
        locked_cells_list = []
        for (r, c), node in self.state.nodes.items():
            if node.status == NodeStatus.LOCKED and node.embedding is not None:
                cell_key = f"{r}_{c}"
                locked_cells_list.append(cell_key)
                embeddings_dict[cell_key] = [round(float(v), 6) for v in node.embedding]

        profile_data = {
            "schema_version": "2.0",
            "roll_no": self.state.roll_no,
            "name": self.state.name,
            "enrolled_timestamp": datetime.now(timezone.utc).isoformat(),
            "coverage_summary": {
                "total_nodes": self.state.total_cells,
                "locked_nodes": self.state.locked_count,
                "coverage_percentage": round(self.state.coverage_pct, 2)
            },
            "node_matrix": {
                "rows": self.state.grid_rows,
                "cols": self.state.grid_cols,
                "locked_cells": locked_cells_list
            },
            "embeddings": embeddings_dict
        }

        success = self.repo.save_profile(profile_data)
        if success:
            logger.info(f"Profile saved successfully for {self.state.roll_no} ({self.state.name})")
            self.is_enrolling = False
            return True

        return False


# Aliased for backward compatibility
RegistrationService = DynamicRegistrationService
