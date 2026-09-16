"""Enrollment UI view controller for interactive student registration."""
import cv2
import numpy as np

from config.settings import AppSettings, get_settings
from services.registration_service import RegistrationService
from ui.components import COLOR_ACCENT_BLUE, COLOR_GREEN, COLOR_TEXT_WHITE, draw_header_banner, draw_pose_indicator
from utils.logger import get_logger
from utils.video_stream import VideoStreamReader

logger = get_logger("RegistrationView")


class RegistrationView:
    """Controls multi-pose interactive registration window."""

    def __init__(self, registration_service: RegistrationService, video_stream: VideoStreamReader, settings: AppSettings = None):
        self.service = registration_service
        self.stream = video_stream
        self.settings = settings or get_settings()

    def run_registration(self, roll_no: str, name: str) -> bool:
        """Run interactive registration window until student enrollment finishes."""
        self.service.start_session(roll_no, name)
        if not self.stream.open():
            logger.error("Could not open video stream for registration.")
            return False

        window_name = f"AdvancEye 2.0 - Enrollment: {name} ({roll_no})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        saved = False

        try:
            for frame in self.stream.frames():
                render_frame = frame.copy()
                face, pose, status = self.service.process_enrollment_frame(frame)

                draw_header_banner(
                    render_frame,
                    title=f"Enrollment: {name} [{roll_no}]",
                    subtitle=f"Status: {status.get('status')} | Cells: {status.get('captured_cells', 0)}"
                )

                if pose:
                    draw_pose_indicator(render_frame, pose.pitch, pose.yaw, pose.roll)

                if face:
                    xmin, ymin, xmax, ymax = face.bbox_pixel
                    cv2.rectangle(render_frame, (xmin, ymin), (xmax, ymax), COLOR_ACCENT_BLUE, 2)

                # Instruction footer
                cv2.putText(
                    render_frame,
                    "Instructions: Slowly rotate head (Left/Right/Up/Down). Press 'S' to Save & Exit, 'Q' to Cancel",
                    (15, render_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLOR_TEXT_WHITE,
                    1,
                    cv2.LINE_AA
                )

                cv2.imshow(window_name, render_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') or key == ord('S'):
                    saved = self.service.finalize_and_save()
                    break
                elif key == ord('q') or key == ord('Q') or key == 27:
                    logger.info("Registration cancelled by user.")
                    break
        finally:
            self.stream.close()
            cv2.destroyWindow(window_name)

        return saved
