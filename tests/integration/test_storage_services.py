"""Integration tests for Phase 3 Storage and Service layer."""
import shutil
import unittest
from pathlib import Path
import numpy as np

from config.settings import AppSettings, StorageSettings, AttendanceSettings
from storage.json_repository import JSONRepository
from services.attendance_service import AttendanceService


class TestStorageAndServices(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_data_phase3")
        self.storage_settings = StorageSettings(base_dir=self.test_dir)
        self.repo = JSONRepository(self.storage_settings)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_json_repository_save_get_list(self):
        profile = {
            "roll_no": "2026_CS_042",
            "name": "Jane Doe",
            "enrolled_at": "2026-09-16T19:30:00Z",
            "total_cells_captured": 2,
            "canonical_landmark_cells": [{"landmark_id": 0, "row": 1, "col": 2}],
            "embeddings": {
                "0_0": [0.1] * 512,
                "0_1": [0.2] * 512
            }
        }
        # Save
        self.assertTrue(self.repo.save_profile(profile))

        # Get
        loaded = self.repo.get_profile("2026_CS_042")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["name"], "Jane Doe")

        # List
        all_profs = self.repo.list_all_profiles()
        self.assertEqual(len(all_profs), 1)

    def test_corrupted_profile_quarantine(self):
        # Create a corrupted JSON file
        corrupt_file = self.storage_settings.profiles_dir / "2026_CORRUPT_99.json"
        with open(corrupt_file, "w") as f:
            f.write("{ INVALID JSON CONTENT ...")

        # Attempt to load
        res = self.repo.get_profile("2026_CORRUPT_99")
        self.assertIsNone(res)

        # Verify quarantine suffix
        self.assertFalse(corrupt_file.exists())
        quarantined = self.storage_settings.profiles_dir / "2026_CORRUPT_99.json.corrupted"
        self.assertTrue(quarantined.exists())

    def test_attendance_service_gating_and_cooldown(self):
        att_settings = AttendanceSettings(consecutive_frames_required=3, cooldown_period_sec=2.0)
        service = AttendanceService(self.repo, attendance_settings=att_settings)

        # Frame 1: count=1 -> No attendance
        self.assertFalse(service.process_match("2026_CS_042", "Jane Doe", 0.75))
        # Frame 2: count=2 -> No attendance
        self.assertFalse(service.process_match("2026_CS_042", "Jane Doe", 0.76))
        # Frame 3: count=3 -> ATTENDANCE TRIGGERED
        self.assertTrue(service.process_match("2026_CS_042", "Jane Doe", 0.77))
        # Frame 4: count=4 -> Cooldown active -> No attendance
        self.assertFalse(service.process_match("2026_CS_042", "Jane Doe", 0.78))

        # Verify attendance record in storage
        records = self.repo.get_attendance_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["roll_no"], "2026_CS_042")


if __name__ == "__main__":
    unittest.main()
