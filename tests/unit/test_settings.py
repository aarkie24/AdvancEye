"""Unit tests for configuration and settings."""
import unittest
from pathlib import Path
from config.settings import AppSettings, CameraSettings, ModelSettings, RegistrationSettings, AttendanceSettings, StorageSettings, get_settings


class TestSettings(unittest.TestCase):
    def test_default_settings(self):
        settings = get_settings()
        self.assertIsInstance(settings, AppSettings)
        self.assertEqual(settings.app_name, "AdvancEye 2.0")
        self.assertEqual(settings.camera.camera_id, "CAM_CLASSROOM_01")
        self.assertEqual(settings.models.similarity_threshold, 0.40)
        self.assertEqual(settings.registration.grid_rows, 3)
        self.assertEqual(settings.registration.grid_cols, 5)
        self.assertEqual(settings.registration.yaw_range, (-50.0, 50.0))
        self.assertEqual(settings.registration.pitch_range, (-30.0, 30.0))
        self.assertEqual(settings.registration.stable_frames_required, 8)
        self.assertEqual(settings.attendance.consecutive_frames_required, 5)
        self.assertEqual(settings.attendance.cooldown_period_sec, 300.0)

    def test_storage_paths(self):
        test_path = Path("tmp_test_data")
        storage = StorageSettings(base_dir=test_path)
        self.assertEqual(storage.profiles_dir, test_path / "profiles")
        self.assertEqual(storage.attendance_dir, test_path / "attendance")

    def test_custom_camera_settings(self):
        cam = CameraSettings(camera_id="CAM_TEST_99", stream_source="test.mp4", preferred_codec="H265_SW")
        self.assertEqual(cam.camera_id, "CAM_TEST_99")
        self.assertEqual(cam.stream_source, "test.mp4")
        self.assertEqual(cam.preferred_codec, "H265_SW")


if __name__ == "__main__":
    unittest.main()
