"""Performance benchmark and unit tests for Vectorized Dynamic Mesh Renderer."""
import time
import unittest
import numpy as np

from core.face_mesh_topology import get_tesselation_triangles
from core.mesh_types import NodeStatus, RegistrationSessionState
from ui.mesh_renderer import DynamicMeshRenderer


class TestMeshRenderer(unittest.TestCase):
    def setUp(self):
        self.triangles = get_tesselation_triangles()
        self.renderer = DynamicMeshRenderer(self.triangles)
        self.state = RegistrationSessionState(roll_no="2026_CS_042", name="Jane Doe")

    def test_rendering_performance_budget(self):
        """Verify the rendering loop executes within the <= 10 ms (>= 60 FPS) performance budget."""
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Synthetic 468 landmark coordinates
        landmarks = np.zeros((468, 2), dtype=np.int32)
        cx, cy = w // 2, h // 2
        for i in range(468):
            landmarks[i] = [cx + int(100 * np.cos(i)), cy + int(100 * np.sin(i))]

        # Warm up
        for _ in range(5):
            self.renderer.render(frame, landmarks, self.state, active_cell=(1, 2))

        # Benchmark 100 frames
        num_frames = 100
        start_time = time.perf_counter()
        for _ in range(num_frames):
            rendered = self.renderer.render(frame.copy(), landmarks, self.state, active_cell=(1, 2))
        total_time = time.perf_counter() - start_time

        avg_latency_ms = (total_time / num_frames) * 1000.0
        fps = num_frames / total_time

        print(f"\n[BENCHMARK] DynamicMeshRenderer Latency: {avg_latency_ms:.2f} ms/frame | Throughput: {fps:.1f} FPS")

        # Must satisfy Section 2 rule 1 (<= 10 ms per frame)
        self.assertLessEqual(avg_latency_ms, 10.0, f"Render time {avg_latency_ms:.2f} ms exceeded 10 ms budget!")
        self.assertEqual(rendered.shape, (h, w, 3))


if __name__ == "__main__":
    unittest.main()
