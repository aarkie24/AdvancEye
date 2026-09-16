"""JSON file repository implementation with error quarantine and schema validation."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import StorageSettings, get_settings
from storage.base import BaseRepository
from utils.logger import get_logger

logger = get_logger("JSONRepository")


class JSONRepository(BaseRepository):
    """File-based JSON storage implementation for profiles and attendance."""

    def __init__(self, storage_settings: Optional[StorageSettings] = None):
        self.settings = storage_settings or get_settings().storage
        self.profiles_dir = self.settings.profiles_dir
        self.attendance_dir = self.settings.attendance_dir
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.attendance_dir.mkdir(parents=True, exist_ok=True)

    def _validate_profile_schema(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            return False
        if not ("roll_no" in data and "name" in data and "embeddings" in data):
            return False
        # Valid if Schema 2.0 or Schema 1.0
        if "schema_version" in data or "total_cells_captured" in data or "coverage_summary" in data:
            return True
        return True


    def save_profile(self, profile_data: Dict[str, Any]) -> bool:
        """Save student profile JSON adhering to Section 6.1 schema."""
        if not self._validate_profile_schema(profile_data):
            logger.error("Profile data failed schema validation.")
            return False

        roll_no = profile_data["roll_no"]
        file_path = self.profiles_dir / f"{roll_no}.json"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)
            logger.info(f"Successfully saved profile for {roll_no} to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save profile {roll_no}: {e}")
            return False

    def get_profile(self, roll_no: str) -> Optional[Dict[str, Any]]:
        """Load single profile JSON with corrupted file quarantining."""
        file_path = self.profiles_dir / f"{roll_no}.json"
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if self._validate_profile_schema(data):
                return data
            logger.warning(f"Profile {file_path} failed schema validation.")
        except json.JSONDecodeError as jde:
            self._quarantine_file(file_path, f"JSONDecodeError: {jde}")
        except Exception as e:
            logger.error(f"Unexpected error reading profile {file_path}: {e}")

        return None

    def list_all_profiles(self) -> List[Dict[str, Any]]:
        """List all valid student profiles in the profiles directory."""
        profiles = []
        for file_path in self.profiles_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if self._validate_profile_schema(data):
                    profiles.append(data)
                else:
                    logger.warning(f"Skipping invalid profile schema: {file_path}")
            except json.JSONDecodeError as jde:
                self._quarantine_file(file_path, f"JSONDecodeError: {jde}")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        return profiles

    def _quarantine_file(self, file_path: Path, reason: str) -> None:
        """Quarantine corrupted JSON files by renaming with .corrupted suffix."""
        corrupted_path = file_path.with_suffix(".json.corrupted")
        try:
            file_path.rename(corrupted_path)
            logger.error(f"Quarantined corrupted profile {file_path} -> {corrupted_path}. Reason: {reason}")
        except Exception as e:
            logger.error(f"Failed to quarantine file {file_path}: {e}")

    def append_attendance_record(self, record: Dict[str, Any], date_str: Optional[str] = None) -> bool:
        """Append record to daily attendance JSON file (Section 6.2 schema)."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        log_file = self.attendance_dir / f"attendance_{date_str}.json"
        current_records: List[Dict[str, Any]] = []

        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    current_records = json.load(f)
            except Exception as e:
                logger.warning(f"Failed reading attendance file {log_file}: {e}. Creating new log list.")
                current_records = []

        current_records.append(record)

        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(current_records, f, indent=2)
            logger.info(f"Logged attendance for {record.get('roll_no')} in {log_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to write attendance log {log_file}: {e}")
            return False

    def get_attendance_records(self, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get attendance records for date."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        log_file = self.attendance_dir / f"attendance_{date_str}.json"
        if not log_file.exists():
            return []

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read attendance records from {log_file}: {e}")
            return []
