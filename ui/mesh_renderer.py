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
            "line": (50, 50, 240),       # BGR: High-Tech Biometric Red
            "fill": (30, 30, 130),
            "alpha": 0.15,
            "vertex_glow": False
        },
        NodeStatus.CAPTURING: {
            "line": (0, 195, 255),       # BGR: Pulsing Gold / Orange
            "fill": (0, 130, 200),
            "alpha": 0.35,
            "vertex_glow": True
        },
        NodeStatus.LOCKED: {
            "line": (0, 230, 118),       # BGR: Luminous Biometric Green #00E676
            "fill": (0, 170, 60),
            "alpha": 0.30,
            "vertex_glow": True
        }
    }

    def __init__(self, triangles: List[Tuple[int, int, int]]):
        self.triangles = np.array(triangles, dtype=np.int32)
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
        """Renders per-node dynamic biometric face mesh in a single blended pass.

        At start, all unvisited nodes render in RED.
        Locked nodes turn PERMANENTLY GREEN.
        Active sampling cell pulses GOLD/ORANGE.

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

        # Face bounding box computation
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        bw = max(x_max - x_min, 1)
        bh = max(y_max - y_min, 1)

        # Pre-compute triangle points and centers
        tri_pts = coords[self.triangles]  # Shape: (N_triangles, 3, 2)
        tri_centers = np.mean(tri_pts, axis=1)  # Shape: (N_triangles, 2)

        u = np.clip((tri_centers[:, 0] - x_min) / bw, 0.0, 0.999)
        v = np.clip((tri_centers[:, 1] - y_min) / bh, 0.0, 0.999)

        cols = np.floor(u * session_state.grid_cols).astype(int)
        rows = np.floor(v * session_state.grid_rows).astype(int)

        overlay = frame.copy()
        t_pulse = time.time()
        max_alpha = 0.25

        # Single pass vectorization across all 15 pose node cells
        for r in range(session_state.grid_rows):
            for c in range(session_state.grid_cols):
                mask = (rows == r) & (cols == c)
                if not np.any(mask):
                    continue

                cell_key = (r, c)
                node = session_state.nodes.get(cell_key)

                if node and node.status == NodeStatus.LOCKED:
                    status = NodeStatus.LOCKED
                elif active_cell == cell_key and node and node.status == NodeStatus.CAPTURING:
                    status = NodeStatus.CAPTURING
                else:
                    status = NodeStatus.UNVISITED

                style = self.COLOR_MAP[status]
                fill_color = style["fill"]
                line_color = style["line"]

                if status == NodeStatus.CAPTURING:
                    cell_alpha = float(np.clip(style["alpha"] + 0.10 * np.sin(10.0 * t_pulse), 0.10, 0.50))
                    max_alpha = max(max_alpha, cell_alpha)

                cell_triangles = tri_pts[mask]
                for pts in cell_triangles:
                    cv2.fillConvexPoly(overlay, pts, fill_color)
                    cv2.polylines(frame, [pts], isClosed=True, color=line_color, thickness=1, lineType=cv2.LINE_AA)

        # SINGLE weighted blend pass for maximum performance (Rule 3)
        cv2.addWeighted(overlay, max_alpha, frame, 1.0 - max_alpha, 0, frame)

        # Render key vertex glowing dots
        for idx in self.key_vertex_indices:
            if idx < len(coords):
                pt = (int(coords[idx, 0]), int(coords[idx, 1]))
                u_pt = np.clip((coords[idx, 0] - x_min) / bw, 0.0, 0.999)
                v_pt = np.clip((coords[idx, 1] - y_min) / bh, 0.0, 0.999)
                c_pt = int(u_pt * session_state.grid_cols)
                r_pt = int(v_pt * session_state.grid_rows)

                v_node = session_state.nodes.get((r_pt, c_pt))
                if v_node and v_node.status == NodeStatus.LOCKED:
                    cv2.circle(frame, pt, 3, (0, 230, 118), -1, lineType=cv2.LINE_AA)
                elif v_node and v_node.status == NodeStatus.CAPTURING:
                    cv2.circle(frame, pt, 3, (0, 195, 255), -1, lineType=cv2.LINE_AA)
                else:
                    cv2.circle(frame, pt, 2, (50, 50, 240), -1, lineType=cv2.LINE_AA)

        return frame
