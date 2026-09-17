"""Interactive HUD overlay components, telemetry cards, and progress bars."""
from typing import Optional, Tuple
import cv2
import numpy as np

from config.settings import get_settings
from core.mesh_types import NodeStatus, RegistrationSessionState
from core.pose_estimator import HeadPose

# Dark futuristic HUD color palette (BGR)
HUD_BG = (20, 20, 24)
HUD_BORDER = (55, 55, 65)
HUD_TEXT_WHITE = (245, 245, 245)
HUD_TEXT_MUTED = (160, 160, 170)
HUD_ACCENT_BLUE = (255, 178, 50)
HUD_GREEN = (60, 230, 118)
HUD_GOLD = (0, 195, 255)
HUD_RED = (60, 60, 240)


def draw_hud_panel(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.82) -> None:
    """Draw semi-transparent dark panel with high-tech outline."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), HUD_BG, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), HUD_BORDER, 1, cv2.LINE_AA)


def draw_enrollment_hud(
    frame: np.ndarray,
    session_state: RegistrationSessionState,
    pose: Optional[HeadPose] = None,
    stability_count: int = 0,
    required_stability: int = 8
) -> None:
    """Renders the comprehensive HUD visual protocol specified in Section 7."""
    h, w, _ = frame.shape

    # 1. Top Header Banner
    draw_hud_panel(frame, 0, 0, w, 52, alpha=0.90)
    title_text = f"ADVANCEYE 2.0 | BIOMETRIC ENROLLMENT: {session_state.name} ({session_state.roll_no})"
    cv2.putText(frame, title_text, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, HUD_TEXT_WHITE, 2, cv2.LINE_AA)

    # 2. Right Side: Coverage Progress Card
    card_w = 270
    card_x = w - card_w - 20
    draw_hud_panel(frame, card_x, 65, card_w, 115)

    cv2.putText(frame, "[ COVERAGE PROGRESS ]", (card_x + 15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_ACCENT_BLUE, 1, cv2.LINE_AA)
    pct = session_state.coverage_pct
    nodes_text = f"Nodes: {session_state.locked_count} / {session_state.total_cells}"
    cv2.putText(frame, f"Coverage: {pct:5.1f}%", (card_x + 15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, nodes_text, (card_x + 15, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_TEXT_MUTED, 1, cv2.LINE_AA)

    # Progress Bar
    bar_x = card_x + 15
    bar_y = 150
    bar_w = card_w - 30
    bar_h = 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 45), -1)
    fill_w = int(bar_w * np.clip(pct / 100.0, 0.0, 1.0))
    bar_color = HUD_GREEN if session_state.is_ready_to_save else HUD_GOLD
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), HUD_BORDER, 1)

    # 3. Right Side: Pose Telemetry Card
    draw_hud_panel(frame, card_x, 195, card_w, 135)
    cv2.putText(frame, "[ POSE TELEMETRY ]", (card_x + 15, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_ACCENT_BLUE, 1, cv2.LINE_AA)
    
    settings = get_settings()
    reg_cfg = getattr(settings, "registration", settings.models)

    pitch = pose.pitch if pose else 0.0
    yaw = pose.yaw if pose else 0.0
    roll = pose.roll if pose else 0.0

    yaw_range = session_state.yaw_range or reg_cfg.yaw_range
    pitch_range = session_state.pitch_range or reg_cfg.pitch_range
    grid_cols = session_state.grid_cols or reg_cfg.grid_cols
    grid_rows = session_state.grid_rows or reg_cfg.grid_rows

    yaw_deadzone = (yaw_range[1] - yaw_range[0]) / (2.0 * max(grid_cols, 1))
    pitch_deadzone = (pitch_range[1] - pitch_range[0]) / (2.0 * max(grid_rows, 1))

    yaw_hint = "CENTER"
    if yaw < -yaw_deadzone:
        yaw_hint = "<- LOOK LEFT"
    elif yaw > yaw_deadzone:
        yaw_hint = "LOOK RIGHT ->"

    pitch_hint = "LEVEL"
    if pitch > pitch_deadzone:
        pitch_hint = "LOOK DOWN"
    elif pitch < -pitch_deadzone:
        pitch_hint = "LOOK UP"

    cv2.putText(frame, f"Pitch: {pitch:+5.1f} deg ({pitch_hint})", (card_x + 15, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Yaw:   {yaw:+5.1f} deg ({yaw_hint})", (card_x + 15, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Roll:  {roll:+5.1f} deg", (card_x + 15, 299), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_TEXT_WHITE, 1, cv2.LINE_AA)

    # 4. Right Side: Active Node Status Card
    draw_hud_panel(frame, card_x, 345, card_w, 95)
    cv2.putText(frame, "[ ACTIVE NODE ]", (card_x + 15, 368), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_ACCENT_BLUE, 1, cv2.LINE_AA)
    
    active_cell = session_state.active_cell
    if active_cell:
        r, c = active_cell
        node = session_state.nodes.get(active_cell)
        status_name = node.status.value if node else "UNVISITED"
        target_text = f"Target: Cell (Row {r}, Col {c})"
        state_text = f"State:  {status_name} ({stability_count}/{required_stability})"
    else:
        target_text = "Target: No Face Detected"
        state_text = "State:  WAITING"

    cv2.putText(frame, target_text, (card_x + 15, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, state_text, (card_x + 15, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_GOLD if stability_count > 0 else HUD_TEXT_MUTED, 1, cv2.LINE_AA)

    # 5. Bottom Instruction & Action Bar
    draw_hud_panel(frame, 0, h - 60, w, 60, alpha=0.90)

    completion_pct = getattr(reg_cfg, "completion_threshold_pct", 85.0)
    if session_state.is_ready_to_save:
        instruction = f"[*] COVERAGE COMPLETE (>={completion_pct:.0f}%)! Press [SPACE] or [S] to Save Profile."
        inst_color = HUD_GREEN
    else:
        instruction = f"[!] INSTRUCTION: Rotate and tilt head slowly until active node turns GREEN."
        inst_color = HUD_GOLD

    cv2.putText(frame, instruction, (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, inst_color, 1, cv2.LINE_AA)

    save_btn = "[ SAVE (SPACE/S) ]" if session_state.is_ready_to_save else "[ IN PROGRESS... ]"
    save_col = HUD_GREEN if session_state.is_ready_to_save else HUD_TEXT_MUTED
    cv2.putText(frame, save_btn, (w - 360, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, save_col, 2, cv2.LINE_AA)
    cv2.putText(frame, "[ QUIT (ESC/Q) ]", (w - 160, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, HUD_TEXT_MUTED, 1, cv2.LINE_AA)
