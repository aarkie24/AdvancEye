"""Vectorized real-time dynamic biometric node face mesh renderer."""
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from core.face_mesh_topology import get_canonical_landmark_uv
from core.mesh_types import NodeStatus, RegistrationSessionState
from core.pose_estimator import HeadPose


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

    def __init__(self, triangles: List[Tuple[int, int, int]], grid_rows: int = 3, grid_cols: int = 5):
        self.triangles = np.array(triangles, dtype=np.int32)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        # Pre-compute canonical anatomical cell mapping directly from canonical 468 landmark topology
        u_lm, v_lm = get_canonical_landmark_uv()
        self.u_lm = np.array(u_lm, dtype=np.float32)
        self.v_lm = np.array(v_lm, dtype=np.float32)

        # Vectorized canonical triangle centroids
        tri_u = np.mean(self.u_lm[self.triangles], axis=1)
        tri_v = np.mean(self.v_lm[self.triangles], axis=1)
        self.tri_cols = np.clip(np.floor(tri_u * self.grid_cols).astype(np.int32), 0, self.grid_cols - 1)
        self.tri_rows = np.clip(np.floor(tri_v * self.grid_rows).astype(np.int32), 0, self.grid_rows - 1)

        self.lm_cols = np.clip(np.floor(self.u_lm * self.grid_cols).astype(np.int32), 0, self.grid_cols - 1)
        self.lm_rows = np.clip(np.floor(self.v_lm * self.grid_rows).astype(np.int32), 0, self.grid_rows - 1)

        # Key landmark indices for glowing vertex nodes
        self.key_vertex_indices = [
            1, 4, 33, 61, 133, 144, 152, 159, 168, 197, 263, 291, 362, 373, 386, 440
        ]

    def render(
        self,
        frame: np.ndarray,
        landmarks: Any,
        session_state: RegistrationSessionState,
        active_cell: Optional[Tuple[int, int]] = None,
        pose: Optional[HeadPose] = None,
        landmarks_3d: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Renders per-node dynamic biometric face mesh in a single blended pass.

        At start, all unvisited nodes render in RED.
        Locked nodes turn PERMANENTLY GREEN on their anatomical facial coordinates.
        Active sampling cell pulses GOLD/ORANGE.

        Args:
            frame: (H, W, 3) BGR video frame.
            landmarks: 468 landmarks (pixel coordinates ndarray or MediaPipe landmarks list).
            session_state: Current registration state with locked nodes.
            active_cell: Currently targeted (row, col) pose node.
            pose: Optional HeadPose instance.
            landmarks_3d: Optional normalized 3D landmark coordinates.

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

        # Pre-compute 2D triangle points for drawing
        tri_pts = coords[self.triangles]  # Shape: (N_triangles, 3, 2)

        overlay = frame.copy()
        t_pulse = time.time()
        max_alpha = 0.25

        # Single pass vectorization across all pose node cells
        for r in range(session_state.grid_rows):
            for c in range(session_state.grid_cols):
                mask = (self.tri_rows == r) & (self.tri_cols == c)
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
                r_pt = self.lm_rows[idx]
                c_pt = self.lm_cols[idx]

                v_node = session_state.nodes.get((r_pt, c_pt))
                if v_node and v_node.status == NodeStatus.LOCKED:
                    cv2.circle(frame, pt, 3, (0, 230, 118), -1, lineType=cv2.LINE_AA)
                elif v_node and v_node.status == NodeStatus.CAPTURING:
                    cv2.circle(frame, pt, 3, (0, 195, 255), -1, lineType=cv2.LINE_AA)
                else:
                    cv2.circle(frame, pt, 2, (50, 50, 240), -1, lineType=cv2.LINE_AA)

        return frame
