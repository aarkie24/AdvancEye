"""Vectorized real-time dynamic biometric node face mesh renderer."""
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from core.mesh_types import NodeStatus, RegistrationSessionState


class DynamicMeshRenderer:
    """Vectorized OpenCV renderer for dynamic biometric node face meshes."""

    COLOR_MAP: Dict[NodeStatus, Dict[str, Any]] = {
        NodeStatus.UNVISITED: {
            "line": (180, 130, 40),      # BGR: Slate Blue/Cyan
            "fill": (120, 80, 20),
            "alpha": 0.12,
            "vertex_glow": False
        },
        NodeStatus.CAPTURING: {
            "line": (0, 190, 255),       # BGR: Pulsing Gold
            "fill": (0, 140, 220),
            "alpha": 0.38,
            "vertex_glow": True
        },
        NodeStatus.LOCKED: {
            "line": (0, 230, 118),       # BGR: Biometric Green #00E676
            "fill": (0, 180, 70),
            "alpha": 0.35,
            "vertex_glow": True
        }
    }

    def __init__(self, triangles: List[Tuple[int, int, int]]):
        self.triangles = triangles
        # Key landmark indices for glowing vertex nodes
        self.key_vertex_indices = [
            1, 4, 33, 61, 133, 144, 152, 159, 168, 197, 263, 291, 362, 373, 386, 440
        ]

    def render(
        self,
        frame: np.ndarray,
        landmarks: Any,
        session_state: RegistrationSessionState,
        active_cell: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Renders the complete node-colored mesh onto the video frame in a single blended pass.

        Args:
            frame: (H, W, 3) BGR video frame.
            landmarks: 468 landmarks (pixel coordinates ndarray or MediaPipe landmarks list).
            session_state: Current registration state with locked nodes.
            active_cell: Currently targeted (row, col) pose node.

        Returns:
            Rendered BGR frame with biometric mesh overlay.
        """
        h, w = frame.shape[:2]

        if hasattr(landmarks, "__len__") and len(landmarks) == 0:
            return frame

        # Vectorized coordinate extraction
        if isinstance(landmarks, np.ndarray):
            if landmarks.ndim == 2 and landmarks.shape[1] >= 2:
                if landmarks.shape[0] < 468:
                    return frame
                if np.max(landmarks[:, 0]) <= 1.0 and np.max(landmarks[:, 1]) <= 1.0:
                    coords = np.empty((len(landmarks), 2), dtype=np.int32)
                    coords[:, 0] = np.clip(landmarks[:, 0] * w, 0, w - 1).astype(np.int32)
                    coords[:, 1] = np.clip(landmarks[:, 1] * h, 0, h - 1).astype(np.int32)
                else:
                    coords = landmarks[:, :2].astype(np.int32)
            else:
                return frame
        else:
            coords = np.empty((len(landmarks), 2), dtype=np.int32)
            for i, lm in enumerate(landmarks):
                coords[i, 0] = int(np.clip(getattr(lm, "x", 0.0) * w, 0, w - 1))
                coords[i, 1] = int(np.clip(getattr(lm, "y", 0.0) * h, 0, h - 1))

        overlay = frame.copy()
        t_pulse = time.time()

        # Determine active cell status
        cell_status = NodeStatus.UNVISITED
        if active_cell and active_cell in session_state.nodes:
            cell_status = session_state.nodes[active_cell].status
        elif session_state.locked_count > 0 and session_state.coverage_pct >= 85.0:
            cell_status = NodeStatus.LOCKED

        style = self.COLOR_MAP[cell_status]
        fill_color = style["fill"]
        line_color = style["line"]
        alpha = style["alpha"]

        if cell_status == NodeStatus.CAPTURING:
            alpha = float(np.clip(alpha + 0.10 * np.sin(10.0 * t_pulse), 0.10, 0.60))

        # Vectorized batch polygon rendering
        tri_pts = coords[self.triangles]  # Shape: (N_triangles, 3, 2)
        for pts in tri_pts:
            cv2.fillConvexPoly(overlay, pts, fill_color)
            cv2.polylines(frame, [pts], isClosed=True, color=line_color, thickness=1, lineType=cv2.LINE_AA)

        # Single weighted blend pass
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        # Render glowing vertex dots
        dot_color = (0, 230, 118) if cell_status == NodeStatus.LOCKED else (0, 200, 255)
        for idx in self.key_vertex_indices:
            if idx < len(coords):
                pt = (int(coords[idx, 0]), int(coords[idx, 1]))
                cv2.circle(frame, pt, 3, dot_color, -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, pt, 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        return frame
