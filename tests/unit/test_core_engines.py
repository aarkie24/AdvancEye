"""Unit tests for Phase 2 Core ML engines (FaceMesh, PoseEstimator, FaceAligner, FaceMatcher)."""
import unittest
import cv2
import numpy as np

from config.settings import get_settings
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
        synthetic_landmarks[1] = [cx, cy]             # Nose
        synthetic_landmarks[152] = [cx, cy + 100]     # Chin
        synthetic_landmarks[263] = [cx + 60, cy - 40] # Left eye (image right)
        synthetic_landmarks[33] = [cx - 60, cy - 40]  # Right eye (image left)
        synthetic_landmarks[291] = [cx + 35, cy + 50] # Left mouth (image right)
        synthetic_landmarks[61] = [cx - 35, cy + 50]  # Right mouth (image left)

        pose = estimator.estimate_pose(synthetic_landmarks, w, h)
        self.assertIsNotNone(pose)
        self.assertIsInstance(pose.pitch, float)
        self.assertIsInstance(pose.yaw, float)
        self.assertIsInstance(pose.roll, float)
        self.assertEqual(pose.rotation_matrix.shape, (3, 3))
        # Center face should have near-zero pitch and yaw
        self.assertAlmostEqual(pose.pitch, 0.0, delta=5.0)
        self.assertAlmostEqual(pose.yaw, 0.0, delta=5.0)

    def test_pose_estimator_directional_mapping(self):
        """Verify that looking UP/DOWN/LEFT/RIGHT maps to the corresponding grid cells derived from settings."""
        settings = get_settings()
        reg_cfg = getattr(settings, "registration", settings.models)
        estimator = PoseEstimator()
        w, h = 640, 480
        cam_matrix = estimator._build_default_camera_matrix(w, h)
        tvec = np.array([[0.0], [0.0], [600.0]])
        dist = np.zeros((4, 1))

        # Synthetic landmarks for Looking UP (-15 deg Pitch)
        rvec_up = np.array([[-np.radians(15.0)], [0.0], [0.0]])
        pts_up_proj, _ = cv2.projectPoints(CANONICAL_FACE_3D, rvec_up, tvec, cam_matrix, dist)
        up_landmarks = np.zeros((468, 2), dtype=np.float32)
        up_landmarks[CANONICAL_LANDMARK_INDICES] = pts_up_proj.reshape(-1, 2)

        pose_up = estimator.estimate_pose(up_landmarks, w, h)
        self.assertIsNotNone(pose_up)
        self.assertLess(pose_up.pitch, 0.0)  # Negative pitch for looking UP
        row_up, col_up = estimator.quantize_pose_to_grid(pose_up.yaw, pose_up.pitch)
        self.assertEqual(row_up, 0)  # Row 0 = Forehead

        # Synthetic landmarks for Looking DOWN (+15 deg Pitch)
        rvec_down = np.array([[np.radians(15.0)], [0.0], [0.0]])
        pts_down_proj, _ = cv2.projectPoints(CANONICAL_FACE_3D, rvec_down, tvec, cam_matrix, dist)
        down_landmarks = np.zeros((468, 2), dtype=np.float32)
        down_landmarks[CANONICAL_LANDMARK_INDICES] = pts_down_proj.reshape(-1, 2)

        pose_down = estimator.estimate_pose(down_landmarks, w, h)
        self.assertIsNotNone(pose_down)
        self.assertGreater(pose_down.pitch, 0.0)  # Positive pitch for looking DOWN
        row_down, col_down = estimator.quantize_pose_to_grid(pose_down.yaw, pose_down.pitch)
        self.assertEqual(row_down, reg_cfg.grid_rows - 1)  # Last Row = Chin

        # Synthetic landmarks for Turning LEFT (-20 deg Yaw)
        rvec_left = np.array([[0.0], [-np.radians(20.0)], [0.0]])
        pts_left_proj, _ = cv2.projectPoints(CANONICAL_FACE_3D, rvec_left, tvec, cam_matrix, dist)
        left_landmarks = np.zeros((468, 2), dtype=np.float32)
        left_landmarks[CANONICAL_LANDMARK_INDICES] = pts_left_proj.reshape(-1, 2)

        pose_left = estimator.estimate_pose(left_landmarks, w, h)
        self.assertIsNotNone(pose_left)
        self.assertLess(pose_left.yaw, 0.0)  # Negative yaw for turning LEFT
        row_left, col_left = estimator.quantize_pose_to_grid(pose_left.yaw, pose_left.pitch)
        self.assertGreaterEqual(col_left, reg_cfg.grid_cols // 2 + 1)  # Right-side profile facing camera

        # Synthetic landmarks for Turning RIGHT (+20 deg Yaw)
        rvec_right = np.array([[0.0], [np.radians(20.0)], [0.0]])
        pts_right_proj, _ = cv2.projectPoints(CANONICAL_FACE_3D, rvec_right, tvec, cam_matrix, dist)
        right_landmarks = np.zeros((468, 2), dtype=np.float32)
        right_landmarks[CANONICAL_LANDMARK_INDICES] = pts_right_proj.reshape(-1, 2)

        pose_right = estimator.estimate_pose(right_landmarks, w, h)
        self.assertIsNotNone(pose_right)
        self.assertGreater(pose_right.yaw, 0.0)  # Positive yaw for turning RIGHT
        row_right, col_right = estimator.quantize_pose_to_grid(pose_right.yaw, pose_right.pitch)
        self.assertLessEqual(col_right, reg_cfg.grid_cols // 2 - 1)  # Left-side profile facing camera

    def test_pose_grid_thresholds(self):
        """Test exact angular thresholds for neutral deadzone and cell transitions derived from settings."""
        settings = get_settings()
        reg_cfg = getattr(settings, "registration", settings.models)
        estimator = PoseEstimator()

        mid_row = reg_cfg.grid_rows // 2
        mid_col = reg_cfg.grid_cols // 2

        # Center deadzone (neutral looking straight)
        self.assertEqual(estimator.quantize_pose_to_grid(yaw=0.0, pitch=0.0), (mid_row, mid_col))
        self.assertEqual(estimator.quantize_pose_to_grid(yaw=5.0, pitch=-5.0), (mid_row, mid_col))
        self.assertEqual(estimator.quantize_pose_to_grid(yaw=-5.0, pitch=5.0), (mid_row, mid_col))

        # Distinct movements
        # Up-Center (row 0, center col)
        self.assertEqual(estimator.quantize_pose_to_grid(yaw=0.0, pitch=reg_cfg.pitch_range[0] * 0.7), (0, mid_col))
        # Down-Center (last row, center col)
        self.assertEqual(estimator.quantize_pose_to_grid(yaw=0.0, pitch=reg_cfg.pitch_range[1] * 0.7), (reg_cfg.grid_rows - 1, mid_col))
        # Left turn (Right profile facing camera -> highest col)
        self.assertEqual(estimator.quantize_pose_to_grid(yaw=reg_cfg.yaw_range[0] * 0.8, pitch=0.0), (mid_row, reg_cfg.grid_cols - 1))
        # Right turn (Left profile facing camera -> col 0)
        self.assertEqual(estimator.quantize_pose_to_grid(yaw=reg_cfg.yaw_range[1] * 0.8, pitch=0.0), (mid_row, 0))

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
