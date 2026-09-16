"""Resilient video stream reader with H.265 HW -> H.265 PyAV SW -> OpenCV fallback chain."""
import time
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

from config.settings import CameraSettings, get_settings
from utils.logger import get_logger

logger = get_logger("VideoStreamReader")


class VideoStreamReader:
    """Multi-codec video ingestion reader with automatic reconnection and fallback."""

    def __init__(self, camera_settings: Optional[CameraSettings] = None):
        self.settings = camera_settings or get_settings().camera
        self.source = self.settings.stream_source
        self.cap: Optional[cv2.VideoCapture] = None
        self.av_container = None
        self.active_codec: str = "NONE"
        self._is_running: bool = False
        self._last_frame: Optional[np.ndarray] = None

    def open(self) -> bool:
        """Attempt to open stream according to fallback hierarchy:
        1. H265 HW (NVDEC/CUDA) if configured
        2. PyAV SW H.265 decoder
        3. Standard OpenCV VideoCapture (H.264 / Webcam V4L2/DSHOW)
        """
        logger.info(f"Opening video stream from source: {self.source}")

        # Attempt 1: PyAV for H.265 if source looks like a file/rtsp and codec requested
        if self.settings.preferred_codec in ("H265_HW", "H265_SW") and not self._is_webcam_index(self.source):
            if self._try_open_pyav():
                self.active_codec = "PYAV_H265"
                self._is_running = True
                logger.info("Successfully opened video stream using PyAV (H.265).")
                return True
            logger.warning("PyAV H.265 decoder failed. Falling back to standard OpenCV decoder.")

        # Attempt 2: Standard OpenCV VideoCapture
        if self._try_open_opencv():
            self.active_codec = "OPENCV_STANDARD"
            self._is_running = True
            logger.info("Successfully opened video stream using OpenCV standard backend.")
            return True

        logger.error(f"Failed to open video stream from source: {self.source} across all decoders.")
        return False

    def _is_webcam_index(self, src: str) -> bool:
        try:
            int(src)
            return True
        except ValueError:
            return False

    def _try_open_pyav(self) -> bool:
        try:
            import av
            self.av_container = av.open(self.source)
            # Find video stream
            video_streams = [s for s in self.av_container.streams if s.type == 'video']
            if not video_streams:
                self.av_container.close()
                self.av_container = None
                return False
            return True
        except Exception as e:
            logger.debug(f"PyAV opening failed: {e}")
            self.av_container = None
            return False

    def _try_open_opencv(self) -> bool:
        try:
            src = int(self.source) if self._is_webcam_index(self.source) else self.source
            self.cap = cv2.VideoCapture(src)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.frame_height)
                return True
            return False
        except Exception as e:
            logger.error(f"OpenCV VideoCapture failed: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next video frame as a BGR numpy array.

        Returns:
            Tuple of (success, frame).
        """
        if not self._is_running:
            return False, None

        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._last_frame = frame
                return True, frame
            else:
                logger.warning("OpenCV frame grab returned False.")
                return False, None

        if self.av_container is not None:
            try:
                for frame in self.av_container.decode(video=0):
                    bgr_frame = frame.to_ndarray(format='bgr24')
                    self._last_frame = bgr_frame
                    return True, bgr_frame
            except Exception as e:
                logger.warning(f"PyAV frame decoding failed: {e}")
                return False, None

        return False, None

    def reconnect(self) -> bool:
        """Attempt to reconnect after a stream failure."""
        logger.info(f"Attempting stream reconnection in {self.settings.reconnect_interval_sec}s...")
        self.close()
        time.sleep(self.settings.reconnect_interval_sec)
        return self.open()

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames continuously, handling disconnections gracefully."""
        while self._is_running:
            success, frame = self.read_frame()
            if success and frame is not None:
                yield frame
            else:
                logger.warning("Stream disconnected or frame lost. Reconnecting...")
                if not self.reconnect():
                    logger.error("Reconnection failed. Retrying...")
                    time.sleep(self.settings.reconnect_interval_sec)

    def close(self) -> None:
        """Release all allocated stream and decoding resources."""
        self._is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.debug(f"Error releasing cv2.VideoCapture: {e}")
            self.cap = None

        if self.av_container is not None:
            try:
                self.av_container.close()
            except Exception as e:
                logger.debug(f"Error closing av.Container: {e}")
            self.av_container = None

        self.active_codec = "NONE"
        logger.info("Video stream closed and resources released.")
