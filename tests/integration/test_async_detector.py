"""Integration tests for Phase 4 Async Detector Worker."""
import time
import unittest
from unittest.mock import MagicMock
import numpy as np

from core.async_detector import AsyncDetector
from services.recognition_service import RecognitionService


class TestAsyncDetector(unittest.TestCase):
    def test_async_detector_lifecycle_and_nonblocking(self):
        mock_recognition_service = MagicMock(spec=RecognitionService)
        mock_recognition_service.process_frame.return_value = []

        detector = AsyncDetector(mock_recognition_service, max_queue_size=2)
        detector.start()
        self.assertTrue(detector.running)

        # Submit synthetic frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            submitted = detector.submit_frame(frame)
            self.assertTrue(submitted)

        time.sleep(0.3)
        results, fps = detector.get_latest_results()
        self.assertIsInstance(results, list)
        self.assertIsInstance(fps, float)

        detector.stop()
        self.assertFalse(detector.running)


if __name__ == "__main__":
    unittest.main()
