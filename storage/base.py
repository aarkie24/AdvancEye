"""Abstract base repository interface for student profiles and attendance logging."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRepository(ABC):
    """Abstract interface for data persistence."""

    @abstractmethod
    def save_profile(self, profile_data: Dict[str, Any]) -> bool:
        """Persist a single student profile."""
        pass

    @abstractmethod
    def get_profile(self, roll_no: str) -> Optional[Dict[str, Any]]:
        """Retrieve a student profile by roll number."""
        pass

    @abstractmethod
    def list_all_profiles(self) -> List[Dict[str, Any]]:
        """Load and return all valid student profiles."""
        pass

    @abstractmethod
    def append_attendance_record(self, record: Dict[str, Any], date_str: Optional[str] = None) -> bool:
        """Append an attendance record to the daily attendance log."""
        pass

    @abstractmethod
    def get_attendance_records(self, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve attendance records for a specified date (YYYY-MM-DD)."""
        pass
