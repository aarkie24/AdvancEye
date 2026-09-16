"""Unit tests for 468-landmark MediaPipe Delaunay triangulation topology."""
import unittest
from core.face_mesh_topology import get_tesselation_triangles, ANATOMICAL_REGIONS


class TestMeshTopology(unittest.TestCase):
    def test_get_tesselation_triangles(self):
        triangles = get_tesselation_triangles()
        self.assertIsInstance(triangles, list)
        self.assertGreater(len(triangles), 100)

        # Verify all triangles have 3 valid vertex indices in [0, 467]
        for tri in triangles:
            self.assertEqual(len(tri), 3)
            u, v, w = tri
            self.assertGreaterEqual(u, 0)
            self.assertLess(u, 468)
            self.assertGreaterEqual(v, 0)
            self.assertLess(v, 468)
            self.assertGreaterEqual(w, 0)
            self.assertLess(w, 468)
            # Ensure no degenerate vertices
            self.assertNotEqual(u, v)
            self.assertNotEqual(v, w)
            self.assertNotEqual(u, w)

    def test_anatomical_regions(self):
        self.assertEqual(len(ANATOMICAL_REGIONS), 9)
        for r_id, region in ANATOMICAL_REGIONS.items():
            self.assertIn("name", region)
            self.assertIn("indices", region)
            self.assertGreater(len(region["indices"]), 0)


if __name__ == "__main__":
    unittest.main()
