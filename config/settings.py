"""Global application settings and configuration management for AdvancEye 2.0."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class CameraSettings:
    """Camera capture and stream decoding configurations."""
    camera_id: str = "CAM_CLASSROOM_01"
    stream_source: str = "0"  # Webcam index "0" or RTSP/file URI
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    reconnect_interval_sec: float = 2.0
    decode_timeout_sec: float = 5.0
    preferred_codec: str = "H265_HW"  # H265_HW -> H265_SW -> OPENCV_DEFAULT


@dataclass(frozen=True)
class ModelSettings:
    """ML model parameters and thresholds."""
    # Face Mesh
    max_num_faces: int = 10
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Grid mapping
    grid_rows: int = 5
    grid_cols: int = 5
    epsilon: float = 1e-6

    # 3D Pose and Frontalization
    canonical_crop_size: Tuple[int, int] = (112, 112)
    focal_length_scale: float = 1.0  # Multiplier on image width for default K
    max_pose_angle_degrees: float = 60.0

    # Face Embedder & Matcher
    embedder_model_name: str = "buffalo_l"
    embedding_dim: int = 512
    onnx_execution_provider: str = "CPUExecutionProvider"  # or CUDAExecutionProvider
    similarity_threshold: float = 0.40  # tau_sim


@dataclass(frozen=True)
class AttendanceSettings:
    """Attendance logging and temporal gating parameters."""
    consecutive_frames_required: int = 5  # N_consec
    cooldown_period_sec: float = 300.0   # T_cooldown (5 minutes)


@dataclass(frozen=True)
class StorageSettings:
    """Storage directories and format settings."""
    base_dir: Path = field(default_factory=lambda: Path(os.getenv("ADVANCEYE_DATA_DIR", "data")))
    profiles_subdir: str = "profiles"
    attendance_subdir: str = "attendance"

    @property
    def profiles_dir(self) -> Path:
        return self.base_dir / self.profiles_subdir

    @property
    def attendance_dir(self) -> Path:
        return self.base_dir / self.attendance_subdir


@dataclass(frozen=True)
class AppSettings:
    """Root configuration object for AdvancEye 2.0."""
    app_name: str = "AdvancEye 2.0"
    debug: bool = False
    camera: CameraSettings = field(default_factory=CameraSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    attendance: AttendanceSettings = field(default_factory=AttendanceSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)


_default_settings: AppSettings = AppSettings()


def get_settings() -> AppSettings:
    """Retrieve global application settings instance."""
    return _default_settings
