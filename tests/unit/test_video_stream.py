"""Unit tests for video stream reader."""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from config.settings import CameraSettings
from utils.video_stream import VideoStreamReader


class TestVideoStream(unittest.TestCase):
    def test_video_stream_reader_init(self):
        cam = CameraSettings(stream_source="test.mp4")
        reader = VideoStreamReader(cam)
        self.assertEqual(reader.source, "test.mp4")
        self.assertFalse(reader._is_running)
        self.assertEqual(reader.active_codec, "NONE")

    @patch("cv2.VideoCapture")
    def test_video_stream_opencv_fallback(self, mock_cv):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_cv.return_value = mock_cap

        cam = CameraSettings(stream_source="0", preferred_codec="OPENCV_DEFAULT")
        reader = VideoStreamReader(cam)
        self.assertTrue(reader.open())
        self.assertEqual(reader.active_codec, "OPENCV_STANDARD")

        success, frame = reader.read_frame()
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (720, 1280, 3))

        reader.close()
        self.assertFalse(reader._is_running)
        self.assertIsNone(reader.cap)

    def test_video_stream_open_failure(self):
        cam = CameraSettings(stream_source="non_existent_source_9999.mp4", preferred_codec="OPENCV_DEFAULT")
        reader = VideoStreamReader(cam)
        with patch("cv2.VideoCapture") as mock_cv:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cv.return_value = mock_cap
            self.assertFalse(reader.open())


if __name__ == "__main__":
    unittest.main()
