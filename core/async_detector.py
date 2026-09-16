"""Non-blocking asynchronous multi-face detection and recognition worker thread."""
from queue import Empty, Queue
import threading
import time
from typing import List, Optional, Tuple
import numpy as np

from services.recognition_service import RecognitionService, RecognizedFace
from utils.logger import get_logger

logger = get_logger("AsyncDetector")


class AsyncDetector:
    """Runs heavy face mesh, frontalization, and ArcFace inference on a dedicated background worker thread."""

    def __init__(self, recognition_service: RecognitionService, max_queue_size: int = 2):
        self.recognition_service = recognition_service
        self.input_queue: Queue = Queue(maxsize=max_queue_size)
        self.lock = threading.Lock()

        self._latest_results: List[RecognizedFace] = []
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._start_time: float = time.time()

        self.running: bool = False
        self.worker_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background inference worker thread."""
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("AsyncDetector background inference worker thread started.")

    def stop(self) -> None:
        """Gracefully stop the worker thread."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        logger.info("AsyncDetector background inference worker thread stopped.")

    def submit_frame(self, frame_bgr: np.ndarray) -> bool:
        """Submit a new frame to the queue without blocking the rendering/ingestion loop."""
        if not self.running:
            return False

        # Drop older frames if worker is busy to guarantee lowest latency
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except Empty:
                pass

        try:
            self.input_queue.put_nowait(frame_bgr.copy())
            return True
        except Exception:
            return False

    def get_latest_results(self) -> Tuple[List[RecognizedFace], float]:
        """Thread-safe retrieval of latest recognition results and calculated FPS."""
        with self.lock:
            return list(self._latest_results), self._fps

    def _worker_loop(self) -> None:
        """Continuous worker execution loop."""
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.2)
            except Empty:
                continue

            t0 = time.time()
            try:
                results = self.recognition_service.process_frame(frame)
                with self.lock:
                    self._latest_results = results
                    self._frame_count += 1
                    elapsed = time.time() - self._start_time
                    if elapsed >= 1.0:
                        self._fps = self._frame_count / elapsed
                        self._frame_count = 0
                        self._start_time = time.time()
            except Exception as e:
                logger.error(f"Error in async detector worker loop: {e}")
            finally:
                self.input_queue.task_done()
