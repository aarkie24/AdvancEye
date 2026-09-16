"""Vectorized Cosine Similarity Face Matcher and Identity Scorer."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from config.settings import ModelSettings, get_settings
from utils.logger import get_logger

logger = get_logger("FaceMatcher")


@dataclass
class MatchResult:
    """Result of identity matching against database."""
    roll_no: str
    name: str
    similarity: float
    is_match: bool


class FaceMatcher:
    """Vectorized cosine similarity comparison across multi-cell student profile embeddings."""

    def __init__(self, model_settings: Optional[ModelSettings] = None):
        self.settings = model_settings or get_settings().models
        self.tau_sim = self.settings.similarity_threshold

    @staticmethod
    def cosine_similarity_matrix(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between queries and gallery matrix.

        Args:
            query_embeddings: (Q, D) array of L2-normalized vectors.
            gallery_embeddings: (G, D) array of L2-normalized vectors.

        Returns:
            (Q, G) cosine similarity matrix.
        """
        # Since embeddings are L2 normalized, cosine similarity is the dot product
        return np.dot(query_embeddings, gallery_embeddings.T)

    def match_identity(
        self,
        query_embedding: np.ndarray,
        profiles_cache: Dict[str, Dict[str, any]]
    ) -> MatchResult:
        """Find best matching student profile for a given query embedding.

        Args:
            query_embedding: (512,) float32 vector.
            profiles_cache: Dictionary mapping roll_no -> {
                "name": str,
                "embeddings_matrix": np.ndarray of shape (N_cells, 512)
            }

        Returns:
            MatchResult with best score and match decision.
        """
        if not profiles_cache or query_embedding is None:
            return MatchResult(roll_no="UNKNOWN", name="Unknown", similarity=0.0, is_match=False)

        best_roll = "UNKNOWN"
        best_name = "Unknown"
        max_similarity = -1.0

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        for roll_no, data in profiles_cache.items():
            gallery = data.get("embeddings_matrix")
            if gallery is None or len(gallery) == 0:
                continue

            # Compute similarities against all multi-pose/cell embeddings for this student
            sims = np.dot(gallery, q_norm)
            max_sim_student = float(np.max(sims))

            if max_sim_student > max_similarity:
                max_similarity = max_sim_student
                best_roll = roll_no
                best_name = data.get("name", "Unknown")

        is_match = max_similarity >= self.tau_sim
        return MatchResult(
            roll_no=best_roll if is_match else "UNKNOWN",
            name=best_name if is_match else "Unknown",
            similarity=max(0.0, max_similarity),
            is_match=is_match
        )
