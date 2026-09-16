"""InsightFace ArcFace feature embedder with automatic execution provider fallback."""
from typing import List, Optional
import numpy as np
import insightface
from insightface.app import FaceAnalysis

from config.settings import ModelSettings, get_settings
from utils.logger import get_logger

logger = get_logger("FaceEmbedder")


class FaceEmbedder:
    """ArcFace 512D deep facial feature extractor."""

    def __init__(self, model_settings: Optional[ModelSettings] = None):
        self.settings = model_settings or get_settings().models
        self.app: Optional[FaceAnalysis] = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialize ArcFace FaceAnalysis model with fallback provider."""
        providers = [self.settings.onnx_execution_provider]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")

        try:
            logger.info(f"Initializing InsightFace ({self.settings.embedder_model_name}) with providers: {providers}")
            self.app = FaceAnalysis(
                name=self.settings.embedder_model_name,
                providers=providers
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize InsightFace with primary provider: {e}. Retrying CPU...")
            try:
                self.app = FaceAnalysis(
                    name=self.settings.embedder_model_name,
                    providers=["CPUExecutionProvider"]
                )
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("InsightFace initialized successfully with CPUExecutionProvider.")
            except Exception as cpu_e:
                logger.error(f"InsightFace CPU fallback also failed: {cpu_e}")
                self.app = None

    def extract_embedding(self, face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512-D L2-normalized ArcFace embedding from a canonical face crop.

        Args:
            face_crop_bgr: (112, 112, 3) BGR face image.

        Returns:
            Normalized (512,) float32 vector or None if extraction fails.
        """
        if self.app is None:
            logger.error("FaceEmbedder is not initialized.")
            return None

        try:
            faces = self.app.get(face_crop_bgr)
            if not faces:
                # Direct embedding fallback using recognition model directly if available
                rec_model = self.app.models.get("recognition")
                if rec_model is not None:
                    emb = rec_model.get_feat(face_crop_bgr).flatten()
                    norm = np.linalg.norm(emb)
                    return (emb / (norm + 1e-10)).astype(np.float32)
                return None

            embedding = faces[0].normed_embedding
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error during feature embedding extraction: {e}")
            return None
