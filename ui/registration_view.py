"""Enrollment UI view controller with Dynamic Biometric Face Mesh Shader and HUD."""
import cv2
import numpy as np

from config.settings import AppSettings, get_settings
from core.face_mesh_topology import get_tesselation_triangles
from services.registration_service import DynamicRegistrationService
from ui.hud_overlays import draw_enrollment_hud
from ui.mesh_renderer import DynamicMeshRenderer
from utils.logger import get_logger
from utils.video_stream import VideoStreamReader

logger = get_logger("RegistrationView")


class RegistrationView:
    """Controls dynamic node-mesh interactive registration window."""

    def __init__(
        self,
        registration_service: DynamicRegistrationService,
        video_stream: VideoStreamReader,
        settings: AppSettings = None
    ):
        self.service = registration_service
        self.stream = video_stream
        self.settings = settings or get_settings()
        self.triangles = get_tesselation_triangles()
        self.mesh_renderer = DynamicMeshRenderer(self.triangles)

    def run_registration(self, roll_no: str, name: str) -> bool:
        """Run interactive registration window with real-time dynamic mesh shader."""
        session_state = self.service.start_session(roll_no, name)
        if not self.stream.open():
            logger.error("Could not open video stream for registration.")
            return False

        window_name = f"AdvancEye 2.0 - Biometric Node Mesh Registration: {name} ({roll_no})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        saved = False

        try:
            for frame in self.stream.frames():
                # Mirror frame horizontally for natural selfie webcam experience
                display_frame = cv2.flip(frame, 1)

                face, pose, state, stability_count = self.service.process_frame(display_frame)

                # Render dynamic biometric node face mesh
                if face is not None:
                    display_frame = self.mesh_renderer.render(
                        frame=display_frame,
                        landmarks=face.landmarks_pixel,
                        session_state=state,
                        active_cell=state.active_cell,
                        pose=pose,
                        landmarks_3d=face.landmarks_3d
                    )

                # Render comprehensive HUD telemetry, progress bar, guidance
                draw_enrollment_hud(
                    frame=display_frame,
                    session_state=state,
                    pose=pose,
                    stability_count=stability_count,
                    required_stability=self.service.STABLE_FRAMES_REQUIRED
                )

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF

                # Save when ready or explicitly requested
                if key in (ord('s'), ord('S'), 32, 13):  # S, Space, Enter
                    if state.is_ready_to_save or state.locked_count > 0:
                        saved = self.service.finalize_and_save()
                        break
                    else:
                        logger.info("Coverage below threshold. Continue moving head or press Q to exit.")
                elif key in (ord('q'), ord('Q'), 27):  # Q or ESC
                    logger.info("Registration cancelled by user.")
                    break
        finally:
            self.stream.close()
            cv2.destroyWindow(window_name)

        return saved
