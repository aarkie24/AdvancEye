"""Real-time multi-face recognition pipeline service."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from config.settings import AppSettings, get_settings
from core.face_mesh import FaceDetectionResult, FaceMeshEngine
from core.pose_estimator import HeadPose, PoseEstimator
from core.face_aligner import FaceAligner
from core.face_embedder import FaceEmbedder
from core.face_matcher import FaceMatcher, MatchResult
from services.attendance_service import AttendanceService
from storage.base import BaseRepository
from utils.logger import get_logger

logger = get_logger("RecognitionService")


@dataclass
class RecognizedFace:
    """Consolidated state of single detected and recognized face in a frame."""
    face_detection: FaceDetectionResult
    pose: Optional[HeadPose]
    match: MatchResult
    attendance_logged: bool


class RecognitionService:
    """Coordinates live multi-face landmark extraction, pose estimation, ArcFace embedding, and matching."""

    def __init__(
        self,
        repository: BaseRepository,
        attendance_service: AttendanceService,
        face_mesh: Optional[FaceMeshEngine] = None,
        pose_estimator: Optional[PoseEstimator] = None,
        face_aligner: Optional[FaceAligner] = None,
        face_embedder: Optional[FaceEmbedder] = None,
        face_matcher: Optional[FaceMatcher] = None,
        settings: Optional[AppSettings] = None
    ):
        self.repo = repository
        self.attendance_service = attendance_service
        self.settings = settings or get_settings()

        self.face_mesh = face_mesh or FaceMeshEngine(self.settings.models)
        self.pose_estimator = pose_estimator or PoseEstimator()
        self.face_aligner = face_aligner or FaceAligner(self.settings.models)
        self.face_embedder = face_embedder or FaceEmbedder(self.settings.models)
        self.face_matcher = face_matcher or FaceMatcher(self.settings.models)

        # In-memory fast cache: roll_no -> { "name": str, "embeddings_matrix": np.ndarray }
        self.profiles_cache: Dict[str, Dict[str, any]] = {}
        self.reload_profiles()

    def reload_profiles(self) -> None:
        """Reload all registered student profiles from repository into memory cache."""
        self.profiles_cache.clear()
        profiles = self.repo.list_all_profiles()
        for p in profiles:
            roll_no = p.get("roll_no")
            name = p.get("name", "Unknown")
            embeddings_dict = p.get("embeddings", {})

            if roll_no and embeddings_dict:
                emb_list = [np.array(vec, dtype=np.float32) for vec in embeddings_dict.values()]
                if emb_list:
                    matrix = np.vstack(emb_list)
                    # Normalize rows
                    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                    matrix = matrix / np.maximum(norms, 1e-10)
                    self.profiles_cache[roll_no] = {
                        "name": name,
                        "embeddings_matrix": matrix
                    }
        logger.info(f"Loaded {len(self.profiles_cache)} student profiles into recognition cache.")

    def process_frame(self, frame_bgr: np.ndarray) -> List[RecognizedFace]:
        """Process a live frame, detect multi-faces, recognize identities, and trigger attendance events."""
        h, w, _ = frame_bgr.shape
        detections = self.face_mesh.process_frame(frame_bgr)
        recognized_faces: List[RecognizedFace] = []
        detected_roll_nos: List[str] = []

        for face in detections:
            pose = self.pose_estimator.estimate_pose(face.landmarks_pixel, w, h)
            crop = self.face_aligner.frontalize_and_crop(frame_bgr, face.bbox_pixel, pose)
            query_emb = self.face_embedder.extract_embedding(crop)

            if query_emb is not None:
                match_res = self.face_matcher.match_identity(query_emb, self.profiles_cache)
            else:
                match_res = MatchResult(roll_no="UNKNOWN", name="Unknown", similarity=0.0, is_match=False)

            logged = False
            if match_res.is_match:
                detected_roll_nos.append(match_res.roll_no)
                logged = self.attendance_service.process_match(
                    roll_no=match_res.roll_no,
                    name=match_res.name,
                    similarity=match_res.similarity
                )

            recognized_faces.append(RecognizedFace(
                face_detection=face,
                pose=pose,
                match=match_res,
                attendance_logged=logged
            ))

        # Reset consecutive counter for absent students
        self.attendance_service.reset_consecutive_for_absent(detected_roll_nos)
        return recognized_faces
