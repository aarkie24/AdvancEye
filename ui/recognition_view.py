"""Real-time recognition UI view controller for classroom/multi-face attendance stream."""
import cv2
import numpy as np

from config.settings import AppSettings, get_settings
from core.async_detector import AsyncDetector
from services.recognition_service import RecognitionService
from ui.components import (
    COLOR_RED,
    COLOR_TEXT_WHITE,
    draw_face_bounding_box,
    draw_header_banner,
    draw_pose_indicator
)
from utils.logger import get_logger
from utils.video_stream import VideoStreamReader

logger = get_logger("RecognitionView")


class RecognitionView:
    """Controls real-time multi-face attendance recognition UI window."""

    def __init__(
        self,
        recognition_service: RecognitionService,
        video_stream: VideoStreamReader,
        settings: AppSettings = None
    ):
        self.service = recognition_service
        self.stream = video_stream
        self.settings = settings or get_settings()
        self.async_detector = AsyncDetector(self.service)

    def run_live_recognition(self) -> None:
        """Run the non-blocking real-time attendance recognition view."""
        if not self.stream.open():
            logger.error("Could not open video stream for live recognition.")
            return

        self.service.reload_profiles()
        self.async_detector.start()

        window_name = "AdvancEye 2.0 - Real-Time Multi-Face Attendance Monitoring"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            for frame in self.stream.frames():
                render_frame = frame.copy()
                self.async_detector.submit_frame(frame)

                recognized_faces, fps = self.async_detector.get_latest_results()

                draw_header_banner(
                    render_frame,
                    title="AdvancEye 2.0 - Classroom Attendance",
                    subtitle=f"Monitored: {len(recognized_faces)} face(s)",
                    fps=fps
                )

                for rf in recognized_faces:
                    face = rf.face_detection
                    match = rf.match
                    draw_face_bounding_box(
                        render_frame,
                        bbox=face.bbox_pixel,
                        label=match.name if match.is_match else "UNKNOWN",
                        similarity=match.similarity,
                        is_recognized=match.is_match,
                        attendance_logged=rf.attendance_logged
                    )
                    if rf.pose:
                        draw_pose_indicator(render_frame, rf.pose.pitch, rf.pose.yaw, rf.pose.roll)

                cv2.putText(
                    render_frame,
                    "Press 'Q' or ESC to exit monitoring",
                    (15, render_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLOR_TEXT_WHITE,
                    1,
                    cv2.LINE_AA
                )

                cv2.imshow(window_name, render_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    logger.info("Live recognition closed by user.")
                    break
        finally:
            self.async_detector.stop()
            self.stream.close()
            cv2.destroyWindow(window_name)
