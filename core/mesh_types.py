"""Data types, enums, and session models for dynamic face mesh registration."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


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
    """Encapsulates the complete active enrollment session state across 15 pose nodes."""
    roll_no: str
    name: str
    grid_rows: int = 3
    grid_cols: int = 5
    nodes: Dict[Tuple[int, int], PoseGridCell] = field(default_factory=dict)
    active_cell: Optional[Tuple[int, int]] = None
    total_cells: int = 15
    locked_count: int = 0
    coverage_pct: float = 0.0
    is_ready_to_save: bool = False

    def __post_init__(self):
        if not self.nodes:
            self.total_cells = self.grid_rows * self.grid_cols
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    self.nodes[(r, c)] = PoseGridCell(
                        row=r,
                        col=c,
                        yaw_center=32.0 - c * 16.0,
                        pitch_center=-13.3 + r * 13.3,
                        status=NodeStatus.UNVISITED,
                        stability_count=0,
                        embedding=None
                    )
