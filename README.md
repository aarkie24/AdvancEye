# AdvancEye 2.0: Enterprise Biometric Registration & Attendance System

AdvancEye 2.0 is an enterprise-grade, real-time multi-face biometric registration and attendance monitoring system. It replaces legacy fixed-frame grid recognition with **Geometrically Invariant 3D Landmark Sampling**, **Pose-Aware 3D Face Frontalization**, and **Resilient Multithreaded Video Ingestion**.

---

## Key Features

- **4-Layer Modular Architecture**: Clean separation across `config/`, `core/`, `services/`, `storage/`, `ui/`, and `utils/`.
- **MediaPipe 468 3D Face Mesh**: Full 3D facial landmark mesh extraction and spatial topological grid mapping ($5 \times 5$).
- **Pose-Aware 3D Frontalization**: Perspective homography rectification ($\mathbf{H} = \mathbf{K} \mathbf{R}^{-1} \mathbf{K}^{-1}$) using OpenCV `solvePnP` head pose estimation (Pitch, Yaw, Roll up to $\pm 60^\circ$).
- **InsightFace ArcFace (512-D Embeddings)**: Deep facial representation with vectorized cosine similarity matching and automatic execution provider fallback.
- **Resilient Video Ingestion**: Multi-codec fallback hierarchy ($H.265 \text{ HW} \rightarrow H.265 \text{ PyAV SW} \rightarrow \text{OpenCV Webcam}$) with auto-reconnection.
- **Attendance Business Logic**: Gated with similarity thresholds ($\tau_{\text{sim}} \ge 0.40$), temporal consistency ($N_{\text{consec}} \ge 5$ consecutive frames), and anti-duplication cooldowns ($T_{\text{cooldown}} = 300\text{s}$).
- **Non-Blocking Inference Pipeline**: Dedicated background worker thread ensuring high framerate UI rendering without detection lag.
- **Modern OpenCV HUD**: High-contrast biometric bounding boxes, 3D pose indicators, and attendance badges.

---

## Directory Structure

```
AdvancEye/
├── config/
│   ├── __init__.py
│   └── settings.py               # AppSettings dataclass & constants
├── core/
│   ├── __init__.py
│   ├── face_mesh.py              # MediaPipe 468 3D landmark engine & grid mapper
│   ├── pose_estimator.py         # solvePnP 3D pose estimator (pitch, yaw, roll)
│   ├── face_aligner.py           # 3D Pose-Aware Frontalization & Warping
│   ├── face_embedder.py          # InsightFace ArcFace embedder wrapper
│   ├── face_matcher.py           # Vectorized Cosine Similarity matcher
│   └── async_detector.py         # Non-blocking async detection worker
├── services/
│   ├── __init__.py
│   ├── registration_service.py   # Enrollment state machine logic
│   ├── recognition_service.py    # Live recognition pipeline logic
│   └── attendance_service.py     # Attendance triggering & log management
├── storage/
│   ├── __init__.py
│   ├── base.py                   # Abstract repository interface
│   └── json_repository.py        # Student profile & attendance JSON storage
├── ui/
│   ├── __init__.py
│   ├── components.py             # OpenCV HUD rendering components
│   ├── registration_view.py      # Enrollment UI view controller
│   └── recognition_view.py       # Live stream UI view controller
├── utils/
│   ├── __init__.py
│   ├── video_stream.py           # Stream reader with H.265/H.264 fallbacks
│   └── logger.py                 # Structured system logger
├── tests/
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── e2e/                      # End-to-end CLI tests
├── data/                         # Student profiles & daily attendance logs
├── main.py                       # CLI application entry point
├── projectcreation.md            # System blueprint and specification
└── requirements.txt              # Project dependencies
```

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aarkie24/AdvancEye.git
   cd AdvancEye
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Test Suite**:
   ```bash
   pytest tests/ -v
   ```

---

## CLI Usage

### 1. Student Enrollment
Launch interactive enrollment window to register a student:
```bash
python main.py register --roll 2026_CS_042 --name "Jane Doe"
```
*Follow on-screen instructions to rotate your head across different angles. Press `S` to save and finish.*

### 2. Live Multi-Face Attendance Monitoring
Start real-time attendance stream monitoring:
```bash
python main.py run
# Or specify a camera / RTSP stream source:
python main.py run --source 0
```

### 3. List Registered Profiles
```bash
python main.py list
```

### 4. View Attendance Records
```bash
python main.py attendance
# Or for a specific date:
python main.py attendance --date 2026-09-16
```

---

## License
MIT License
