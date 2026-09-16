"""Unit tests for Registration Session State Machine and Blue -> Gold -> Green transitions."""
import unittest
from unittest.mock import MagicMock
import numpy as np

from core.face_mesh import FaceDetectionResult
from core.mesh_types import NodeStatus, RegistrationSessionState
from core.pose_estimator import HeadPose
from services.registration_service import DynamicRegistrationService
from storage.base import BaseRepository


class TestRegistrationState(unittest.TestCase):
    def test_initial_session_state(self):
        state = RegistrationSessionState(roll_no="2026_CS_042", name="Jane Doe")
        self.assertEqual(state.total_cells, 15)
        self.assertEqual(state.locked_count, 0)
        self.assertEqual(state.coverage_pct, 0.0)
        self.assertFalse(state.is_ready_to_save)

        # All 15 nodes must initially be UNVISITED
        for cell, node in state.nodes.items():
            self.assertEqual(node.status, NodeStatus.UNVISITED)
            self.assertEqual(node.stability_count, 0)
            self.assertIsNone(node.embedding)

    def test_state_transitions_to_locked_green(self):
        mock_repo = MagicMock(spec=BaseRepository)
        mock_mesh = MagicMock()
        mock_pose = MagicMock()
        mock_aligner = MagicMock()
        mock_embedder = MagicMock()

        # Synthetic face detection result
        landmarks_pix = np.zeros((468, 2), dtype=np.int32)
        landmarks_norm = np.zeros((468, 3), dtype=np.float32)
        grid_cells = [(0, 0)] * 468
        grid_mat = np.zeros((5, 5), dtype=np.int32)
        mock_face = FaceDetectionResult(
            landmarks_3d=landmarks_norm,
            landmarks_pixel=landmarks_pix,
            bbox_pixel=(50, 50, 200, 200),
            grid_cells=grid_cells,
            grid_matrix=grid_mat
        )
        mock_mesh.process_frame.return_value = [mock_face]

        # Pose in center cell (1, 2)
        mock_pose.estimate_pose.return_value = HeadPose(
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            rotation_vector=np.zeros(3),
            translation_vector=np.zeros(3),
            rotation_matrix=np.eye(3)
        )

        mock_aligner.frontalize_and_crop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
        mock_embedder.extract_embedding.return_value = np.random.randn(512).astype(np.float32)

        service = DynamicRegistrationService(
            repository=mock_repo,
            face_mesh=mock_mesh,
            pose_estimator=mock_pose,
            face_aligner=mock_aligner,
            face_embedder=mock_embedder
        )

        service.start_session("2026_CS_042", "Jane Doe")
        # High contrast synthetic image for Laplacian sharpness
        synthetic_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Feed 7 frames (CAPTURING state, not yet locked)
        for i in range(7):
            _, _, state, stab = service.process_frame(synthetic_frame)
            target_node = state.nodes[(1, 2)]
            self.assertEqual(target_node.status, NodeStatus.CAPTURING)
            self.assertEqual(stab, i + 1)
            self.assertEqual(state.locked_count, 0)

        # 8th frame: reaches STABLE_FRAMES_REQUIRED -> Transitions to LOCKED (GREEN)
        _, _, state, stab = service.process_frame(synthetic_frame)
        target_node = state.nodes[(1, 2)]
        self.assertEqual(target_node.status, NodeStatus.LOCKED)
        self.assertEqual(state.locked_count, 1)
        self.assertAlmostEqual(state.coverage_pct, (1 / 15) * 100.0, places=2)

        # 9th frame: Node remains locked (Green), does not revert
        _, _, state, stab = service.process_frame(synthetic_frame)
        target_node = state.nodes[(1, 2)]
        self.assertEqual(target_node.status, NodeStatus.LOCKED)


if __name__ == "__main__":
    unittest.main()
