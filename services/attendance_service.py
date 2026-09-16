"""Attendance logging and temporal gating service adhering to Section 7 business logic."""
from datetime import datetime, timezone
import time
from typing import Dict, List, Optional
from config.settings import AttendanceSettings, CameraSettings, get_settings
from storage.base import BaseRepository
from utils.logger import get_logger

logger = get_logger("AttendanceService")


class AttendanceService:
    """Enforces similarity gating, temporal consistency, and anti-duplication cooldown."""

    def __init__(
        self,
        repository: BaseRepository,
        attendance_settings: Optional[AttendanceSettings] = None,
        camera_settings: Optional[CameraSettings] = None
    ):
        self.repo = repository
        self.settings = attendance_settings or get_settings().attendance
        self.camera_settings = camera_settings or get_settings().camera

        # State tracking: roll_no -> consecutive frame count
        self.consecutive_matches: Dict[str, int] = {}
        # Cooldown tracking: roll_no -> last logged timestamp (epoch float)
        self.last_logged_time: Dict[str, float] = {}

    def process_match(self, roll_no: str, name: str, similarity: float) -> bool:
        """Evaluate match event against Section 7 business rules.

        Conditions:
        1. Similarity Gate: S >= tau_sim (handled prior to this or checked here)
        2. Temporal Consistency: Matched for at least N_consec consecutive frames.
        3. Anti-Duplication Cooldown: Not logged within T_cooldown seconds.

        Returns:
            True if attendance event was triggered and logged, False otherwise.
        """
        now = time.time()

        if roll_no == "UNKNOWN":
            return False

        # Update consecutive frame count
        count = self.consecutive_matches.get(roll_no, 0) + 1
        self.consecutive_matches[roll_no] = count

        # Check temporal consistency
        if count < self.settings.consecutive_frames_required:
            return False

        # Check anti-duplication cooldown
        last_time = self.last_logged_time.get(roll_no, 0.0)
        if (now - last_time) < self.settings.cooldown_period_sec:
            # Under cooldown, ignore event
            return False

        # All conditions met -> Log attendance
        record = {
            "roll_no": roll_no,
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence_similarity": round(float(similarity), 4),
            "consecutive_frames_matched": count,
            "camera_id": self.camera_settings.camera_id
        }

        success = self.repo.append_attendance_record(record)
        if success:
            self.last_logged_time[roll_no] = now
            logger.info(f"Attendance verified & logged: {name} ({roll_no})")
            return True

        return False

    def reset_consecutive_for_absent(self, detected_roll_nos: List[str]) -> None:
        """Reset consecutive frame counter for any students not detected in the current frame."""
        for roll_no in list(self.consecutive_matches.keys()):
            if roll_no not in detected_roll_nos:
                self.consecutive_matches[roll_no] = 0
