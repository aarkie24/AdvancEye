"""OpenCV HUD rendering primitives, bounding boxes, pose badges, and card overlays."""
from typing import List, Optional, Tuple
import cv2
import numpy as np

# Color Palette (BGR)
COLOR_BG_DARK = (24, 24, 28)
COLOR_TEXT_WHITE = (245, 245, 245)
COLOR_ACCENT_BLUE = (255, 178, 50)
COLOR_GREEN = (60, 220, 80)
COLOR_RED = (60, 60, 240)
COLOR_YELLOW = (50, 215, 255)
COLOR_GRAY = (120, 120, 120)


def draw_header_banner(
    frame: np.ndarray,
    title: str,
    subtitle: Optional[str] = None,
    fps: Optional[float] = None
) -> None:
    """Render semi-transparent top HUD status banner."""
    h, w, _ = frame.shape
    banner_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), COLOR_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, title, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_TEXT_WHITE, 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(frame, subtitle, (300, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ACCENT_BLUE, 1, cv2.LINE_AA)
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 130, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2, cv2.LINE_AA)


def draw_face_bounding_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    similarity: float,
    is_recognized: bool,
    attendance_logged: bool = False
) -> None:
    """Render high-contrast biometric bounding box and info badge."""
    xmin, ymin, xmax, ymax = bbox
    color = COLOR_GREEN if is_recognized else COLOR_RED
    if attendance_logged:
        color = COLOR_YELLOW

    # Bounding Box Corners
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    corner_len = 15
    # Top-left
    cv2.line(frame, (xmin, ymin), (xmin + corner_len, ymin), color, 4)
    cv2.line(frame, (xmin, ymin), (xmin, ymin + corner_len), color, 4)
    # Bottom-right
    cv2.line(frame, (xmax, ymax), (xmax - corner_len, ymax), color, 4)
    cv2.line(frame, (xmax, ymax), (xmax, ymax - corner_len), color, 4)

    # Label Badge
    badge_text = f"{label} ({similarity:.2f})"
    if attendance_logged:
        badge_text += " [ATTENDANCE LOGGED]"

    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    badge_y = max(ymin - 10, th + 10)
    cv2.rectangle(frame, (xmin, badge_y - th - 5), (xmin + tw + 10, badge_y + 5), COLOR_BG_DARK, -1)
    cv2.rectangle(frame, (xmin, badge_y - th - 5), (xmin + tw + 10, badge_y + 5), color, 1)
    cv2.putText(frame, badge_text, (xmin + 5, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)


def draw_pose_indicator(
    frame: np.ndarray,
    pitch: float,
    yaw: float,
    roll: float,
    origin: Tuple[int, int] = (15, 90)
) -> None:
    """Render 3D head pose telemetry overlay."""
    x, y = origin
    lines = [
        f"Pitch: {pitch:+5.1f} deg",
        f"Yaw:   {yaw:+5.1f} deg",
        f"Roll:  {roll:+5.1f} deg"
    ]
    cv2.rectangle(frame, (x - 5, y - 20), (x + 160, y + 60), COLOR_BG_DARK, -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
