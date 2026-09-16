"""MediaPipe FaceMesh 468 landmark topology, triangulation graph, and anatomical clusters."""
from typing import Any, Dict, List, Set, Tuple
import mediapipe as mp

MP_FACE_MESH = mp.solutions.face_mesh
FACEMESH_TESSELATION = MP_FACE_MESH.FACEMESH_TESSELATION

# 9 Anatomical facial clusters according to Section 3.2
ANATOMICAL_REGIONS: Dict[int, Dict[str, Any]] = {
    0: {
        "name": "NOSE_DORSUM_TIP",
        "indices": [1, 2, 4, 5, 6, 19, 94, 98, 168, 195, 197, 220, 275, 327, 440]
    },
    1: {
        "name": "FOREHEAD_BROW_LEFT",
        "indices": [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
    },
    2: {
        "name": "FOREHEAD_BROW_RIGHT",
        "indices": [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]
    },
    3: {
        "name": "LEFT_EYE_PERIORBITAL",
        "indices": [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
    },
    4: {
        "name": "RIGHT_EYE_PERIORBITAL",
        "indices": [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
    },
    5: {
        "name": "LEFT_CHEEK_ZYGOMATIC",
        "indices": [32, 116, 123, 147, 192, 199, 208, 210, 211, 213, 214]
    },
    6: {
        "name": "RIGHT_CHEEK_ZYGOMATIC",
        "indices": [262, 345, 352, 376, 416, 421, 428, 430, 431, 433, 434]
    },
    7: {
        "name": "MOUTH_LIPS_PERIORAL",
        "indices": [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 87, 88, 91, 95, 146, 178, 181, 185]
    },
    8: {
        "name": "CHIN_JAWLINE",
        "indices": [58, 132, 136, 148, 149, 150, 152, 172, 176, 288, 361, 365, 377, 378, 379, 397, 400]
    }
}

_cached_triangles: List[Tuple[int, int, int]] = []


def get_tesselation_triangles() -> List[Tuple[int, int, int]]:
    """Extracts canonical MediaPipe 468 landmark triangulation index triplets.

    Returns:
        List[Tuple[int, int, int]]: List of unique 3-vertex polygon triplets.
    """
    global _cached_triangles
    if _cached_triangles:
        return _cached_triangles

    edge_set = set(FACEMESH_TESSELATION)
    adj: Dict[int, Set[int]] = {}
    for u, v in edge_set:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    seen_triangles: Set[Tuple[int, int, int]] = set()
    triangles: List[Tuple[int, int, int]] = []

    for u in adj:
        for v in adj[u]:
            if v <= u:
                continue
            common = adj[u].intersection(adj[v])
            for w in common:
                if w <= v:
                    continue
                tri = (u, v, w)
                if tri not in seen_triangles:
                    seen_triangles.add(tri)
                    triangles.append(tri)

    _cached_triangles = triangles
    return _cached_triangles
