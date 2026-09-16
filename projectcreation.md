# Project Blueprint & System Specification: AdvancEye 2.0

## 1. Executive Summary & System Vision

**AdvancEye 2.0** is an enterprise-grade, real-time multi-face biometric registration and attendance monitoring system. It replaces legacy fixed-frame grid recognition with **Geometrically Invariant 3D Landmark Sampling**, **Pose-Aware 3D Face Frontalization**, and **Resilient Multithreaded Video Ingestion**.

This specification defines the strict technical standards, data schemas, mathematical formulations, application service layers, and phase-by-phase development protocols required to build the codebase.

---

## 2. AI Agent Guidelines & Development Protocol

### 2.1 AI Agent Rules
1. **Strict Modular Isolation**: Code must be split cleanly across `config`, `core`, `services`, `storage`, `ui`, and `utils`. No monolithic files exceeding 300 lines of code.
2. **Type Safety & Documentation**: All public functions and classes must include Python type hints (`typing`) and clear docstrings defining arguments, return types, and exceptions.
3. **Zero Hardcoded Configuration**: Magic numbers, thresholds, dimensions, and file paths must reside in `config/settings.py` or environment overrides.
4. **No Masking Exceptions**: All errors must be explicitly caught, logged via `utils/logger.py`, and handled gracefully. Silently swallowing exceptions is forbidden.
5. **No Scope Creep**: Implement only the features explicitly specified in this blueprint and roadmap.

### 2.2 Phase-by-Phase Development Protocol
* The development agent must construct the system **one phase at a time**.
* Before proceeding from Phase $N$ to Phase $N+1$, the agent MUST execute the automated test suite for Phase $N$ and verify all Definition of Done (DoD) criteria.

---

## 3. System Architecture & Layer Separation

The application enforces a strict 4-layer architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer (ui/)             │
│   Components, HUD Overlays, OpenCV Windows, Dialogs     │
└────────────────────────────┬────────────────────────────┘
                             │ Calls
┌────────────────────────────▼────────────────────────────┐
│                Application Layer (services/)            │
│  RegistrationService, RecognitionService, Attendance    │
└────────────────────────────┬────────────────────────────┘
                             │ Coordinates
┌────────────────────────────▼────────────────────────────┐
│                    Core Engine Layer (core/)            │
│ FaceMesh, PoseEstimator, FaceAligner, Embedder, Matcher │
└────────────────────────────┬────────────────────────────┘
                             │ Reads/Writes
┌────────────────────────────▼────────────────────────────┐
│                Infrastructure Layer (storage/ & utils/) │
│  JSONProfileRepository, H265StreamReader, Logger        │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Landmark-Grid Correspondence & 3D Frontalization Math

### 4.1 Landmark-Grid Topological Mapping
To eliminate frame-fixed grid vulnerability under translation, scale, and head tilt:

1. Let $\mathcal{L} = \{\mathbf{p}_i = (x_i, y_i, z_i)\}_{i=1}^{468}$ be the set of 3D facial landmarks extracted by MediaPipe Face Mesh.
2. Compute the bounding box of the detected face in normalized image space:
   $$x_{\min} = \min_{i} x_i, \quad x_{\max} = \max_{i} x_i, \quad y_{\min} = \min_{i} y_i, \quad y_{\max} = \max_{i} y_i$$
3. For each landmark $i$, compute its normalized coordinate within the face box:
   $$u_i = \frac{x_i - x_{\min}}{\max(x_{\max} - x_{\min}, \epsilon)}, \quad v_i = \frac{y_i - y_{\min}}{\max(y_{\max} - y_{\min}, \epsilon)}$$
4. Map landmark $i$ to a discrete grid cell $(r_i, c_i)$ in an $N_{\text{rows}} \times N_{\text{cols}}$ matrix:
   $$c_i = \min\left(N_{\text{cols}} - 1, \, \lfloor u_i \cdot N_{\text{cols}} \rfloor\right), \quad r_i = \min\left(N_{\text{rows}} - 1, \, \lfloor v_i \cdot N_{\text{rows}} \rfloor\right)$$
5. **Topological Binding**: This initial mapping binds landmark index $i$ to cell $(r_i, c_i)$. During subsequent motion, vertex index $i$ remains bound to its semantic facial feature (e.g., eye corner landmark #33 always tracks the left eye corner), providing full spatial invariance.

### 4.2 3D Pose-Aware Frontalization Algorithm
For high-angle overhead cameras (up to $\pm 60^\circ$ pitch/yaw):

1. **3D Pose Estimation**: Estimate 3D rotation vector $\mathbf{r}$ and translation vector $\mathbf{t}$ via OpenCV `solvePnP` using 2D landmarks and canonical 3D facial model points $\mathbf{P}_{3D}$. Convert $\mathbf{r}$ to rotation matrix $\mathbf{R} \in \mathbb{R}^{3 \times 3}$ using Rodrigues transformation.
2. **Perspective Warping Matrix**: Compute the homography/perspective transform $\mathbf{H} = \mathbf{K} \mathbf{R}^{-1} \mathbf{K}^{-1}$, where $\mathbf{K}$ is the camera intrinsic matrix.
3. **Canonical Alignment**: Warp the cropped facial image via $\mathbf{H}$ (or affine similarity alignment based on pose-projected eye/nose centers) to generate a canonical frontalized crop $\mathbf{I}_{\text{frontal}} \in \mathbb{R}^{112 \times 112 \times 3}$ prior to passing to the ArcFace embedder.

---

## 5. Technology Stack & Codec Fallback Chain

| Component | Primary Option | Fallback 1 | Fallback 2 |
| :--- | :--- | :--- | :--- |
| **Video Decoding** | H.265 (HEVC) HW (NVDEC/CUDA) | H.265 SW (PyAV / FFmpeg) | H.264 / OpenCV V4L2 USB Webcam |
| **Face Mesh** | MediaPipe FaceMesh (468 3D) | OpenCV DNN Face Detector | N/A |
| **Pose Estimator** | OpenCV `solvePnP` | EPnP / Iterative solvePnP | Facial ratio heuristic |
| **Face Embedder** | InsightFace ArcFace (`buffalo_l`) | ONNX Runtime CPU | MobileFaceNet |
| **Storage Engine** | JSON File Repository | SQLite Database | In-Memory Dictionary |

---

## 6. Data Schemas

### 6.1 Student Profile JSON Schema (`data/profiles/{roll_no}.json`)
```json
{
  "roll_no": "2026_CS_042",
  "name": "Jane Doe",
  "enrolled_at": "2026-09-16T19:30:00Z",
  "total_cells_captured": 15,
  "canonical_landmark_cells": [
    {"landmark_id": 0, "row": 1, "col": 2},
    {"landmark_id": 33, "row": 0, "col": 1}
  ],
  "embeddings": {
    "0_0": [0.012, -0.045, 0.089, "... 512 float values ..."],
    "0_1": [0.015, -0.041, 0.082, "..."]
  }
}
```

### 6.2 Attendance Log Record Schema (`data/attendance/attendance_{YYYY-MM-DD}.json`)
```json
[
  {
    "roll_no": "2026_CS_042",
    "name": "Jane Doe",
    "timestamp": "2026-09-16T09:15:32Z",
    "confidence_similarity": 0.742,
    "consecutive_frames_matched": 5,
    "camera_id": "CAM_CLASSROOM_01"
  }
]
```

---

## 7. Attendance Business Logic

An identity match transitions to a logged **Attendance Event** if and only if all the following conditions are met:

1. **Similarity Gate**: Cosine similarity $S \ge \tau_{\text{sim}}$ (default $\tau_{\text{sim}} = 0.40$).
2. **Temporal Consistency**: The subject is recognized with $S \ge \tau_{\text{sim}}$ for at least $N_{\text{consec}}$ consecutive frames (default $N_{\text{consec}} = 5$).
3. **Anti-Duplication Cooldown**: No attendance record has been logged for `roll_no` within the cooldown period $T_{\text{cooldown}}$ (default $T_{\text{cooldown}} = 300$ seconds / 5 minutes).

---

## 8. Error Handling & Resiliency Matrix

| Failure Mode | Detection Mechanism | System Recovery Strategy |
| :--- | :--- | :--- |
| **Video Feed Disconnection** | `cap.read()` returns `False` / Stream timeout | Attempt stream auto-reconnect every 2 seconds; display HUD warning banner. |
| **H.265 HW Decoder Failure** | CUDA/NVDEC init exception | Fallback gracefully to PyAV software H.265 decoder $\rightarrow$ OpenCV standard decoder. |
| **Landmark Occlusion / Loss** | `multi_face_landmarks` is `None` | Skip pose/grid processing for current frame; retain previous bounding box bounding box. |
| **Corrupted Profile JSON** | `json.JSONDecodeError` on startup | Log error, quarantine file (`.corrupted`), notify admin, continue loading valid profiles. |
| **InsightFace Model Init Crash** | ExecutionProvider fail | Fallback to CPU execution provider (`CPUExecutionProvider`). |

---

## 9. Target Directory Structure

```
AdvancEye/
├── config/
│   ├── __init__.py
│   └── settings.py          # AppSettings dataclass & constants
├── core/
│   ├── __init__.py
│   ├── face_mesh.py         # MediaPipe 468 landmark engine
│   ├── pose_estimator.py    # solvePnP 3D pose estimator
│   ├── face_aligner.py      # High-Angle 3D Pose-Aware Frontalization
│   ├── face_embedder.py     # InsightFace ArcFace wrapper
│   ├── face_matcher.py      # Vectorized Cosine Similarity matcher
│   └── async_detector.py    # Non-blocking async detection worker
├── services/
│   ├── __init__.py
│   ├── registration_service.py # Enrollment state machine logic
│   ├── recognition_service.py  # Live recognition pipeline logic
│   └── attendance_service.py   # Attendance triggering & log management
├── storage/
│   ├── __init__.py
│   ├── base.py              # Abstract repository interface
│   └── json_repository.py   # Student profile & attendance JSON storage
├── ui/
│   ├── __init__.py
│   ├── components.py        # OpenCV HUD rendering components
│   ├── registration_view.py # Enrollment UI view controller
│   └── recognition_view.py  # Live stream UI view controller
├── utils/
│   ├── __init__.py
│   ├── video_stream.py      # H.265/H.264 stream reader with fallbacks
│   └── logger.py            # Structured system logger
├── tests/
│   ├── unit/                # Unit tests for math, aligner, matcher
│   ├── integration/         # Integration tests for storage & services
│   └── e2e/                 # End-to-end CLI execution tests
├── data/
│   ├── profiles/            # Profile storage directory
│   └── attendance/          # Attendance logs directory
├── main.py                  # CLI application entry point
└── requirements.txt         # Project dependencies
```

---

## 10. Phase-by-Phase Roadmap & Definition of Done (DoD)

### Phase 1: Infrastructure, Config & Video Stream Reader
- **Deliverables**: `config/settings.py`, `utils/logger.py`, `utils/video_stream.py` with H.265 $\rightarrow$ H.264 $\rightarrow$ USB webcam fallback.
- **DoD**: Unit tests verify configuration loading and `video_stream.py` successfully opens video streams and returns valid frames.

### Phase 2: Core ML Engines & 3D Alignment
- **Deliverables**: `pose_estimator.py`, `face_mesh.py`, `face_aligner.py`, `face_embedder.py`, `face_matcher.py`.
- **DoD**: Unit tests verify 3D landmark cell mapping, pose estimation angles, 3D frontalization matrix computation, embedding extraction, and cosine matching.

### Phase 3: Storage & Service Layer
- **Deliverables**: `storage/json_repository.py`, `services/registration_service.py`, `services/recognition_service.py`, `services/attendance_service.py`.
- **DoD**: Integration tests verify profile saving/loading, JSON schema validity, attendance triggering rules, and anti-duplication cooldown timers.

### Phase 4: Non-Blocking Async Pipeline
- **Deliverables**: `core/async_detector.py`.
- **DoD**: Integration tests verify asynchronous thread execution, non-blocking frame buffer updates, and thread-safe results retrieval.

### Phase 5: UI Layer & Presentation Views
- **Deliverables**: `ui/components.py`, `ui/registration_view.py`, `ui/recognition_view.py`.
- **DoD**: Manual execution confirms crisp HUD rendering, interactive key commands, multi-pose registration card progress, and bounding box drawing.

### Phase 6: Unified CLI & End-to-End Verification
- **Deliverables**: `main.py` CLI (`python main.py run`, `python main.py register`, `python main.py list`).
- **DoD**: All unit and end-to-end tests pass cleanly (`pytest`). Complete enrollment and recognition workflow runs without error.

---

## 11. Explicit Out-of-Scope List

To prevent scope creep, the following features are explicitly **OUT OF SCOPE**:
1. Cloud web dashboards, REST APIs, or microservices (system is a desktop/edge application).
2. Mobile application clients (iOS/Android).
3. Advanced deep-learning anti-spoofing / liveness detection beyond MediaPipe mesh landmark metrics.
4. Database migrations to PostgreSQL / Cloud databases (JSON file repository is the required storage mechanism).
