"""Unit tests for Phase 2 Core ML engines (FaceMesh, PoseEstimator, FaceAligner, FaceMatcher)."""
import unittest
import numpy as np

from core.face_mesh import FaceMeshEngine
from core.pose_estimator import PoseEstimator, CANONICAL_LANDMARK_INDICES, CANONICAL_FACE_3D
from core.face_aligner import FaceAligner
from core.face_matcher import FaceMatcher


class TestCoreEngines(unittest.TestCase):
    def test_topological_grid_mapping(self):
        # Synthetic landmarks in normalized space [0.2, 0.8]
        landmarks = np.array([
            [0.2, 0.2],  # Should be in top-left cell (0, 0)
            [0.8, 0.8],  # Should be in bottom-right cell (4, 4)
            [0.5, 0.5]   # Should be in center cell (2, 2)
        ], dtype=np.float32)

        grid_cells, grid_mat = FaceMeshEngine.map_landmarks_to_grid(
            landmarks, x_min=0.2, x_max=0.8, y_min=0.2, y_max=0.8, n_rows=5, n_cols=5
        )

        self.assertEqual(len(grid_cells), 3)
        self.assertEqual(grid_cells[0], (0, 0))
        self.assertEqual(grid_cells[1], (4, 4))
        self.assertEqual(grid_cells[2], (2, 2))
        self.assertEqual(grid_mat.shape, (5, 5))
        self.assertEqual(grid_mat[0, 0], 1)
        self.assertEqual(grid_mat[4, 4], 1)
        self.assertEqual(grid_mat[2, 2], 1)

    def test_pose_estimator_angles(self):
        estimator = PoseEstimator()
        # Synthetic 2D landmarks centered in a 640x480 frame
        w, h = 640, 480
        cx, cy = w / 2, h / 2
        # Use 6 canonical points projected directly
        synthetic_landmarks = np.zeros((468, 2), dtype=np.float32)
        synthetic_landmarks[1] = [cx, cy]           # Nose
        synthetic_landmarks[152] = [cx, cy + 100]   # Chin
        synthetic_landmarks[263] = [cx - 60, cy - 40] # Left eye
        synthetic_landmarks[33] = [cx + 60, cy - 40]  # Right eye
        synthetic_landmarks[291] = [cx - 40, cy + 50] # Left mouth
        synthetic_landmarks[61] = [cx + 40, cy + 50]  # Right mouth

        pose = estimator.estimate_pose(synthetic_landmarks, w, h)
        self.assertIsNotNone(pose)
        self.assertIsInstance(pose.pitch, float)
        self.assertIsInstance(pose.yaw, float)
        self.assertIsInstance(pose.roll, float)
        self.assertEqual(pose.rotation_matrix.shape, (3, 3))

    def test_face_aligner_crop(self):
        aligner = FaceAligner()
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a synthetic square
        synthetic_frame[100:300, 150:350] = 255
        bbox = (150, 100, 350, 300)

        crop = aligner.frontalize_and_crop(synthetic_frame, bbox, pose=None)
        self.assertEqual(crop.shape, (112, 112, 3))

    def test_face_matcher_vectorized_cosine(self):
        matcher = FaceMatcher()
        # Create 2 synthetic normalized embeddings
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)

        emb2 = np.random.randn(512).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)

        gallery = {
            "2026_CS_042": {
                "name": "Jane Doe",
                "embeddings_matrix": np.vstack([emb1, emb1 * 0.9])
            },
            "2026_CS_043": {
                "name": "John Smith",
                "embeddings_matrix": np.vstack([emb2])
            }
        }

        # Query matching Jane Doe
        res = matcher.match_identity(emb1, gallery)
        self.assertTrue(res.is_match)
        self.assertEqual(res.roll_no, "2026_CS_042")
        self.assertEqual(res.name, "Jane Doe")
        self.assertAlmostEqual(res.similarity, 1.0, places=4)

        # Query matching non-existing / orthogonal embedding
        random_query = np.random.randn(512).astype(np.float32)
        random_query /= np.linalg.norm(random_query)
        # Assuming random 512D vectors have ~0 cosine similarity
        res_rand = matcher.match_identity(random_query, gallery)
        self.assertIsInstance(res_rand.similarity, float)


if __name__ == "__main__":
    unittest.main()
