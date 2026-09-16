"""Unit tests for Phase 5 UI HUD components."""
import unittest
import numpy as np

from ui.components import (
    draw_header_banner,
    draw_face_bounding_box,
    draw_pose_indicator
)


class TestUIComponents(unittest.TestCase):
    def test_draw_header_banner(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_header_banner(frame, title="Test Title", subtitle="Test Subtitle", fps=29.9)
        # Should not throw and modify top area of the frame
        self.assertTrue(np.any(frame[:50, :] > 0))

    def test_draw_face_bounding_box(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (50, 50, 200, 200)
        draw_face_bounding_box(frame, bbox, label="Jane Doe", similarity=0.88, is_recognized=True, attendance_logged=True)
        self.assertTrue(np.any(frame[40:210, 40:210] > 0))

    def test_draw_pose_indicator(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_pose_indicator(frame, pitch=5.2, yaw=-12.4, roll=1.1, origin=(20, 100))
        self.assertTrue(np.any(frame[80:180, 10:200] > 0))


if __name__ == "__main__":
    unittest.main()
