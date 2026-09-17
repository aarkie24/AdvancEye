"""Data types, enums, and session models for dynamic face mesh registration."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from config.settings import get_settings


class NodeStatus(str, Enum):
    """Status of a biometric pose grid node."""
    UNVISITED = "UNVISITED"
    CAPTURING = "CAPTURING"
    LOCKED = "LOCKED"


@dataclass
class PoseGridCell:
    """Represents a discrete (row, col) pose node in the angular grid."""
    row: int
    col: int
    yaw_center: float
    pitch_center: float
    status: NodeStatus = NodeStatus.UNVISITED
    stability_count: int = 0
    embedding: Optional[np.ndarray] = None


@dataclass
class RegistrationSessionState:
    """Encapsulates the complete active enrollment session state across pose nodes."""
    roll_no: str
    name: str
    grid_rows: Optional[int] = None
    grid_cols: Optional[int] = None
    yaw_range: Optional[Tuple[float, float]] = None
    pitch_range: Optional[Tuple[float, float]] = None
    nodes: Dict[Tuple[int, int], PoseGridCell] = field(default_factory=dict)
    active_cell: Optional[Tuple[int, int]] = None
    total_cells: int = 15
    locked_count: int = 0
    coverage_pct: float = 0.0
    is_ready_to_save: bool = False

    def __post_init__(self):
        settings = get_settings()
        reg_cfg = getattr(settings, "registration", None)
        model_cfg = getattr(settings, "models", None)

        if self.grid_rows is None:
            self.grid_rows = getattr(reg_cfg, "grid_rows", getattr(model_cfg, "grid_rows", 3))
        if self.grid_cols is None:
            self.grid_cols = getattr(reg_cfg, "grid_cols", getattr(model_cfg, "grid_cols", 5))
        if self.yaw_range is None:
            self.yaw_range = getattr(reg_cfg, "yaw_range", getattr(model_cfg, "yaw_range", (-50.0, 50.0)))
        if self.pitch_range is None:
            self.pitch_range = getattr(reg_cfg, "pitch_range", getattr(model_cfg, "pitch_range", (-30.0, 30.0)))

        self.total_cells = self.grid_rows * self.grid_cols

        if not self.nodes:
            psi_min, psi_max = self.yaw_range
            theta_min, theta_max = self.pitch_range
            yaw_step = (psi_max - psi_min) / max(self.grid_cols, 1)
            pitch_step = (theta_max - theta_min) / max(self.grid_rows, 1)

            for r in range(self.grid_rows):
                p_center = (theta_min + pitch_step / 2.0) + r * pitch_step
                for c in range(self.grid_cols):
                    y_center = (psi_max - yaw_step / 2.0) - c * yaw_step
                    self.nodes[(r, c)] = PoseGridCell(
                        row=r,
                        col=c,
                        yaw_center=round(y_center, 1),
                        pitch_center=round(p_center, 1),
                        status=NodeStatus.UNVISITED,
                        stability_count=0,
                        embedding=None
                    )
