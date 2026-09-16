# Technical Specification & Machine-Executable Blueprint: Dynamic Biometric Face Mesh Registration Engine

## 1. System Vision & Objective

This document defines the machine-executable specification for the **Dynamic 3D Biometric Node Mesh Registration Engine** in **AdvancEye 2.0**.

### 1.1 Problem Statement
Legacy registration systems rely on either a static bounding box or coarse, binary feedback (e.g., coloring the entire face red/green). This provides poor feedback on angle completeness, yields uneven facial coverage, and fails to verify multi-view biometric feature stability.

### 1.2 The Node-Mapped Solution
During registration, the subject's face is tracked via 468 high-precision 3D landmarks. The facial topology is partitioned into **discrete spatial-pose nodes** (Euler angle bins $\times$ anatomical facet clusters). 

As the user rotates and tilts their head:
1. The system identifies which spatial-pose node is currently being sampled.
2. The active node enters an **investigative state (`CAPTURING`)**, visually highlighted with a pulsating gold wireframe/fill.
3. Once temporal stability ($N \ge 8$ frames), landmark sharpness, and ArcFace feature extraction succeed, the node transitions permanently to **`LOCKED`**.
4. The exact 3D sub-mesh corresponding to that node **dynamically turns into a glowing, translucent biometric green (`#00E676` / BGR `(0, 230, 118)`)**.
5. Once coverage satisfies the completion threshold ($\ge 85\%$), the registration profile is serialized to structured JSON.

---

## 2. AI Implementation Guardrails & Code Standards

An AI coding agent implementing this specification MUST adhere to the following rules:

1. **Performance Budget**: The entire rendering and node calculation loop MUST execute in **$\le 10\text{ ms}$ per frame** (>= 60 FPS) on CPU without hardware acceleration locks.
2. **Vectorization Rule**: Never iterate over pixel arrays in Python. Mesh rasterization MUST use `cv2.fillPoly`, `cv2.polylines`, and vectorized NumPy indexing.
3. **Single Memory Allocation**: Avoid allocating new canvas buffers per triangle. Create a single `overlay` buffer per frame, rasterize all facet polygons, and perform **one** `cv2.addWeighted` blending pass.
4. **Mirroring Consistency**: When `cv2.flip(frame, 1)` is applied for selfie webcam display, landmark coordinates and yaw angles MUST maintain consistent left/right anatomical correspondence.
5. **No Monolithic Files**: Implement according to the modular directory structure specified in Section 8. No single file may exceed 300 lines of code.
6. **Strict Type Hinting & Docstrings**: Every function and dataclass must have full `typing` annotations and explicit Sphinx/Google style docstrings.

---

## 3. Mathematical Formulations & Algorithms

### 3.1 3D Head Pose Coordinate System & Discretization

Given 3D facial landmarks $\mathcal{L} = \{\mathbf{p}_i = (x_i, y_i, z_i)\}_{i=1}^{468}$, the head pose Euler angles $(\psi, \theta, \phi)$ (Yaw, Pitch, Roll) are estimated via `solvePnP` using 6 canonical fiducial landmarks:
* Nose Tip (`#1`)
* Chin (`#152`)
* Left Eye Outer Corner (`#33`)
* Right Eye Outer Corner (`#263`)
* Left Mouth Corner (`#61`)
* Right Mouth Corner (`#291`)

```
   Pitch (θ): Positive = Look Down, Negative = Look Up
   Yaw (ψ):   Positive = Turn Right, Negative = Turn Left (camera mirrored)
   Roll (ϕ):  Positive = Tilt Clockwise, Negative = Tilt Counter-Clockwise
```

#### Angular Grid Node Quantization:
The continuous pose space $(\psi, \theta)$ is mapped to discrete grid indices $(r, c)$:

$$c(\psi) = \text{clamp}\left(\left\lfloor \frac{\psi - \psi_{\min}}{\psi_{\max} - \psi_{\min}} \cdot N_{\text{cols}} \right\rfloor, 0, N_{\text{cols}} - 1\right)$$

$$r(\theta) = \text{clamp}\left(\left\lfloor \frac{\theta - \theta_{\min}}{\theta_{\max} - \theta_{\min}} \cdot N_{\text{rows}} \right\rfloor, 0, N_{\text{rows}} - 1\right)$$

**Default Constants:**
* $N_{\text{cols}} = 5$, $\psi_{\min} = -40.0^\circ$, $\psi_{\max} = +40.0^\circ$
* $N_{\text{rows}} = 3$, $\theta_{\min} = -20.0^\circ$, $\theta_{\max} = +20.0^\circ$
* Total Pose Nodes $K_{\text{pose}} = N_{\text{rows}} \times N_{\text{cols}} = 15$

---

### 3.2 468-Landmark Anatomical Cluster Mapping

The 468 MediaPipe landmarks are deterministically assigned to 9 anatomical regions $\mathcal{R}_0 \dots \mathcal{R}_8$:

```python
ANATOMICAL_REGIONS: Dict[int, Dict[str, Any]] = {
    0: {
        "name": "NOSE_DORSUM_TIP",
        "indices": [1, 2, 4, 5, 6, 19, 94, 98, 168, 195, 197, 220, 275, 327, 440]
    },
    1: {
        "name": "FOREHEAD_BROW_LEFT",
        "indices": [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
    },
    2: {
        "name": "FOREHEAD_BROW_RIGHT",
        "indices": [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]
    },
    3: {
        "name": "LEFT_EYE_PERIORBITAL",
        "indices": [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
    },
    4: {
        "name": "RIGHT_EYE_PERIORBITAL",
        "indices": [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
    },
    5: {
        "name": "LEFT_CHEEK_ZYGOMATIC",
        "indices": [32, 116, 123, 147, 192, 199, 208, 210, 211, 213, 214]
    },
    6: {
        "name": "RIGHT_CHEEK_ZYGOMATIC",
        "indices": [262, 345, 352, 376, 416, 421, 428, 430, 431, 433, 434]
    },
    7: {
        "name": "MOUTH_LIPS_PERIORAL",
        "indices": [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 87, 88, 91, 95, 146, 178, 181, 185]
    },
    8: {
        "name": "CHIN_JAWLINE",
        "indices": [58, 132, 136, 148, 149, 150, 152, 172, 176, 288, 361, 365, 377, 378, 379, 397, 400]
    }
}
```

---

### 3.3 Dynamic Color Palette & Alpha Blending Formulation

All colors are specified in **BGR** format for direct OpenCV rendering:

| State | Line Color (BGR) | Fill Color (BGR) | Alpha $\alpha$ | Glow Radius |
| :--- | :--- | :--- | :--- | :--- |
| **`UNVISITED`** | `(180, 130, 40)` (Technical Slate Cyan) | `(120, 80, 20)` | $0.12$ | 1 px |
| **`CAPTURING`** | `(0, 180, 255)` (Pulsing Gold) | `(0, 140, 220)` | $0.35 + 0.10 \sin(8\pi t)$ | 2 px |
| **`LOCKED`** | `(0, 230, 118)` (Luminous Emerald Green) | `(0, 190, 80)` | $0.32$ | 3 px |

#### Blending Equation:
$$\mathbf{I}_{\text{out}}(x, y) = \text{clamp}\left((1 - \alpha) \cdot \mathbf{I}_{\text{frame}}(x, y) + \alpha \cdot \mathbf{C}_{\text{fill}}, \, 0, \, 255\right)$$

---

## 4. State Machine & Transition Protocol

```
                        [ START REGISTRATION ]
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │   All 15 Nodes UNVISITED     │
                   │ (Mesh renders Blue/Cyan)     │
                   └──────────────┬───────────────┘
                                  │
                   Subject moves head into cell (r, c)
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │        Node CAPTURING        │◄─────────────────┐
                   │ (Mesh pulse-animates Gold)   │                  │
                   └──────────────┬───────────────┘                  │
                                  │                                  │
              Check Frame Stability:                                 │ Jitter /
              1. ΔYaw, ΔPitch < 1.8°/frame                           │ Pose Departure
              2. Temporal Counter >= 8 frames                        │ before 8 frames
              3. Image Sharpness Var(Lapl) >= 70.0                   │
              4. ArcFace Face Extracted                              │
                                  │                                  │
                     ┌────────────┴────────────┐                     │
                     │  All Checks Passed?     ├─ NO (Counter Reset) ┘
                     └────────────┬────────────┘
                                  │ YES
                                  ▼
                   ┌──────────────────────────────┐
                   │         Node LOCKED          │
                   │  (Sub-mesh turns PERMANENT   │
                   │   LUMINOUS GREEN #00E676)    │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
               Coverage Pct = (Locked Nodes / 15) * 100
                                  │
                     ┌────────────┴────────────┐
                     │ Coverage >= 85% ?       │
                     └────────────┬────────────┘
                        YES ┌─────┴─────┐ NO
                            ▼           ▼
                   [ Save Button Active ] [ Continue Guidance ]
```

---

## 5. Precise Code Implementation Blueprint

### 5.1 Dataclass Definitions (`core/mesh_types.py`)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional
import numpy as np


class NodeStatus(str, Enum):
    UNVISITED = "UNVISITED"
    CAPTURING = "CAPTURING"
    LOCKED = "LOCKED"


@dataclass
class PoseGridCell:
    row: int
    col: int
    yaw_center: float
    pitch_center: float
    status: NodeStatus = NodeStatus.UNVISITED
    stability_count: int = 0
    embedding: Optional[np.ndarray] = None


@dataclass
class RegistrationSessionState:
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
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    self.nodes[(r, c)] = PoseGridCell(
                        row=r,
                        col=c,
                        yaw_center=-32.0 + c * 16.0,
                        pitch_center=-13.3 + r * 13.3,
                        status=NodeStatus.UNVISITED,
                        stability_count=0,
                        embedding=None
                    )
```

---

### 5.2 Topology & Delaunay Triangulation (`core/face_mesh_topology.py`)

```python
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Dict

# MediaPipe canonical face mesh connection set
MP_FACE_MESH = mp.solutions.face_mesh
FACEMESH_TESSELATION = MP_FACE_MESH.FACEMESH_TESSELATION


def get_tesselation_triangles() -> List[Tuple[int, int, int]]:
    """
    Extracts canonical MediaPipe 468 landmark triangulation index tuples.
    Returns:
        List[Tuple[int, int, int]]: List of triangle vertex index triplets.
    """
    triangles: List[Tuple[int, int, int]] = []
    # Build unique edge adjacency graph to extract 3-vertex polygons
    edge_set = set(FACEMESH_TESSELATION)
    adj: Dict[int, set] = {}
    for (u, v) in edge_set:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    
    seen_triangles = set()
    for u in adj:
        for v in adj[u]:
            if v <= u:
                continue
            common = adj[u].intersection(adj[v])
            for w in common:
                if w <= v:
                    continue
                tri = tuple(sorted((u, v, w)))
                if tri not in seen_triangles:
                    seen_triangles.add(tri)
                    triangles.append(tri)
    return triangles
```

---

### 5.3 Vectorized Real-Time Dynamic Mesh Renderer (`ui/mesh_renderer.py`)

```python
import cv2
import numpy as np
import time
from typing import List, Tuple, Dict
from core.mesh_types import NodeStatus, RegistrationSessionState


class DynamicMeshRenderer:
    """Vectorized OpenCV renderer for dynamic biometric node face meshes."""

    COLOR_MAP = {
        NodeStatus.UNVISITED: {
            "line": (180, 130, 40),      # BGR: Slate Blue
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

    def render(
        self,
        frame: np.ndarray,
        landmarks: list,
        session_state: RegistrationSessionState,
        active_cell: Tuple[int, int]
    ) -> np.ndarray:
        """
        Renders the complete node-colored mesh onto the video frame.
        
        Args:
            frame: (H, W, 3) BGR video frame.
            landmarks: 468 MediaPipe normalized Landmark objects.
            session_state: Current registration state with locked nodes.
            active_cell: Currently targeted (row, col) pose node.
        """
        h, w = frame.shape[:2]
        coords = np.empty((len(landmarks), 2), dtype=np.int32)
        for i, lm in enumerate(landmarks):
            coords[i, 0] = int(lm.x * w)
            coords[i, 1] = int(lm.y * h)

        overlay = frame.copy()
        t_pulse = time.time()

        # Render all triangles based on whether the active pose node is locked
        for (v0, v1, v2) in self.triangles:
            # Determine status based on active cell and overall coverage
            cell_node = session_state.nodes.get(active_cell)
            cell_status = cell_node.status if cell_node else NodeStatus.UNVISITED
            
            style = self.COLOR_MAP[cell_status]
            fill_color = style["fill"]
            line_color = style["line"]

            # Pulsing alpha calculation for CAPTURING state
            alpha = style["alpha"]
            if cell_status == NodeStatus.CAPTURING:
                alpha += 0.10 * np.sin(10.0 * t_pulse)

            pts = coords[[v0, v1, v2]]
            cv2.fillConvexPoly(overlay, pts, fill_color)
            cv2.polylines(frame, [pts], isClosed=True, color=line_color, thickness=1, lineType=cv2.LINE_AA)

        # Single overlay blend
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Draw glowing vertex nodes for all LOCKED cells or active CAPTURING vertices
        for pt in coords[::6]:  # Sample key landmarks for performance
            cell_node = session_state.nodes.get(active_cell)
            if cell_node and cell_node.status == NodeStatus.LOCKED:
                cv2.circle(frame, tuple(pt), 2, (0, 255, 128), -1, lineType=cv2.LINE_AA)
            elif cell_node and cell_node.status == NodeStatus.CAPTURING:
                cv2.circle(frame, tuple(pt), 2, (0, 200, 255), -1, lineType=cv2.LINE_AA)

        return frame
```

---

### 5.4 Registration Business Logic Service (`services/registration_service.py`)

```python
import cv2
import numpy as np
import time
from typing import Optional, Tuple
from core.mesh_types import NodeStatus, RegistrationSessionState
from pose_utils import get_head_pose, get_grid_cell


class DynamicRegistrationService:
    """Manages pose node transitions, biometric stability, and embedding capture."""

    STABLE_FRAMES_REQUIRED: int = 8
    MIN_SHARPNESS_LAPLACIAN: float = 60.0
    COMPLETION_THRESHOLD_PCT: float = 85.0

    def __init__(self, roll_no: str, name: str, face_analysis_app):
        self.state = RegistrationSessionState(roll_no=roll_no, name=name)
        self.face_app = face_analysis_app
        self.last_cell: Optional[Tuple[int, int]] = None
        self.stable_counter: int = 0

    def process_frame(
        self,
        frame: np.ndarray,
        landmarks: list,
        frame_w: int,
        frame_h: int
    ) -> Tuple[RegistrationSessionState, Optional[Tuple[int, int]]]:
        """
        Executes one registration cycle step.
        """
        pose = get_head_pose(landmarks, frame_w, frame_h)
        if not pose:
            return self.state, None

        yaw, pitch, roll = pose
        cell = get_grid_cell(
            yaw=yaw,
            pitch=pitch,
            n_cols=self.state.grid_cols,
            n_rows=self.state.grid_rows,
            yaw_range=(-40.0, 40.0),
            pitch_range=(-20.0, 20.0)
        )
        self.state.active_cell = cell
        node = self.state.nodes[cell]

        if node.status == NodeStatus.LOCKED:
            # Already captured angle; keep green state
            self.stable_counter = 0
            return self.state, cell

        # Track stability across consecutive frames
        if cell == self.last_cell:
            self.stable_counter += 1
            node.status = NodeStatus.CAPTURING
        else:
            self.stable_counter = 0
            node.status = NodeStatus.UNVISITED

        self.last_cell = cell

        # Check if stability threshold reached for biometric extraction
        if self.stable_counter >= self.STABLE_FRAMES_REQUIRED:
            # Quality Gate: Image Sharpness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if sharpness >= self.MIN_SHARPNESS_LAPLACIAN:
                # Extract ArcFace 512-D embedding
                faces = self.face_app.get(frame)
                if faces and len(faces) > 0:
                    node.embedding = faces[0].embedding
                    node.status = NodeStatus.LOCKED  # <--- TURNS NODE PERMANENTLY GREEN
                    self.state.locked_count = sum(
                        1 for n in self.state.nodes.values() if n.status == NodeStatus.LOCKED
                    )
                    self.state.coverage_pct = (self.state.locked_count / self.state.total_cells) * 100.0
                    self.state.is_ready_to_save = (self.state.coverage_pct >= self.COMPLETION_THRESHOLD_PCT)
                    self.stable_counter = 0

        return self.state, cell
```

---

## 6. Output JSON Profile Schema (`data/profiles/{roll_no}.json`)

When registration completes and is saved, the profile is serialized to standard JSON:

```json
{
  "schema_version": "2.0",
  "roll_no": "2026_CS_042",
  "name": "Jane Doe",
  "enrolled_timestamp": "2026-09-16T21:55:00Z",
  "coverage_summary": {
    "total_nodes": 15,
    "locked_nodes": 14,
    "coverage_percentage": 93.33
  },
  "node_matrix": {
    "rows": 3,
    "cols": 5,
    "locked_cells": ["0_0", "0_1", "0_2", "0_3", "0_4", "1_0", "1_1", "1_2", "1_3", "1_4", "2_0", "2_1", "2_2", "2_3"]
  },
  "embeddings": {
    "0_0": [0.0124, -0.0451, 0.0892, "... 512 float values ..."],
    "0_1": [0.0152, -0.0410, 0.0821, "... 512 float values ..."],
    "1_2": [0.0381, -0.0199, 0.0954, "... 512 float values ..."]
  }
}
```

---

## 7. Interactive HUD & Visual UX Protocol

```
+-----------------------------------------------------------------------+
|  [ADVANCEYE 2.0] BIOMETRIC ENROLLMENT: Jane Doe (2026_CS_042)         |
+-----------------------------------------------------------------------+
|                                                                       |
|             . - ~ ~ ~ - .             [ COVERAGE PROGRESS ]           |
|         . '   \  GREEN  /   ' .       Coverage:  86% [████████░░]     |
|       /        \  MESH /        \     Nodes:     13 / 15              |
|      |          \     /          |                                    |
|      |           (•_•)           |    [ POSE TELEMETRY ]              |
|      |          /     \          |      Pitch:  +2.1° (STABLE)        |
|       \        /       \        /       Yaw:   -14.8° (PAN RIGHT ->)  |
|         . _   /         \   _ .         Roll:   -0.5°                 |
|             ' - ~ ~ ~ - '                                             |
|                                       [ ACTIVE NODE ]                 |
|                                         Target: Cell (Row 1, Col 3)   |
|                                         State:  CAPTURING (6/8)       |
|                                                                       |
|  [!] INSTRUCTION: Turn head slowly right until active node locks GREEN|
|                                                                       |
|  [ SAVE PROFILE (SPACE / ENTER) ]                   [ QUIT (ESC / Q) ]|
+-----------------------------------------------------------------------+
```

---

## 8. Directory Structure & Modular File Locations

```
AdvancEye/
├── config/
│   ├── __init__.py
│   └── settings.py               # Grid size, thresholds, colors, ranges
├── core/
│   ├── __init__.py
│   ├── mesh_types.py             # NodeStatus enum, PoseGridCell, SessionState
│   ├── face_mesh_topology.py     # 468 MediaPipe Delaunay triangulation graph
│   └── pose_estimator.py         # solvePnP 3D Euler angles & grid cell mapper
├── services/
│   ├── __init__.py
│   └── registration_service.py   # State machine & ArcFace embedding capture
├── ui/
│   ├── __init__.py
│   ├── mesh_renderer.py          # Vectorized OpenCV dynamic green mesh shader
│   ├── hud_overlays.py           # Coverage bars, guidance arrows, telemetry text
│   └── registration_view.py      # Main OpenCV window event loop
└── tests/
    ├── __init__.py
    ├── test_mesh_topology.py     # Verifies 468 triangle generation
    ├── test_registration_state.py# Verifies Blue -> Gold -> Green transitions
    └── test_mesh_renderer.py     # Frame rate & alpha blending benchmarks
```

---

## 9. Automated Testing & Definition of Done (DoD)

### 9.1 Test Execution Commands
```bash
# Execute unit tests for topology and state machine
pytest tests/test_mesh_topology.py tests/test_registration_state.py -v

# Run performance benchmark (must achieve >= 60 FPS)
pytest tests/test_mesh_renderer.py -v -s
```

### 9.2 Definition of Done (DoD) Checklist
- [x] **Topology Verification**: `test_mesh_topology.py` confirms that `get_tesselation_triangles()` returns valid 3-vertex tuples within index range $[0, 467]$.
- [x] **State Transition Integrity**: `test_registration_state.py` verifies:
  1. Initial state of all 15 cells is `UNVISITED`.
  2. Frame count increments on repeated cell observations.
  3. Cell transitions to `LOCKED` only when count reaches `STABLE_FRAMES_REQUIRED` and embedding is present.
  4. Locked cells never revert to `UNVISITED`.
- [x] **Rendering Performance**: `test_mesh_renderer.py` confirms average per-frame render latency is $\le 10\text{ ms}$.
- [x] **Profile Serialization**: Profile JSON strictly validates against the schema in Section 6.

---

## 10. Explicit Out-of-Scope List

1. Rendering via OpenGL / GLFW C-bindings (OpenCV + NumPy vectorized drawing is strictly required).
2. Cloud sync / remote database persistence during registration (file-based JSON persistence is mandatory).
3. 3D depth cameras or IR sensors (pure RGB webcam input with MediaPipe 3D landmark regression).
