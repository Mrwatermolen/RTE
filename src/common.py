import numpy as np

FRONT_FACE_OFFSET = np.array([1, 0, 0])
BACK_FACE_OFFSET = np.array([-1, 0, 0])
RIGHT_FACE_OFFSET = np.array([0, 1, 0])
LEFT_FACE_OFFSET = np.array([0, -1, 0])
TOP_FACE_OFFSET = np.array([0, 0, 1])
BOTTOM_FACE_OFFSET = np.array([0, 0, -1])

FACE_ORDER_INDEX_OFFSET = np.array([
    FRONT_FACE_OFFSET,
    BACK_FACE_OFFSET,
    RIGHT_FACE_OFFSET,
    LEFT_FACE_OFFSET,
    TOP_FACE_OFFSET,
    BOTTOM_FACE_OFFSET
])

def closure_face_order(i, j, k):
    return np.array([o + [i, j, k] for o in FACE_ORDER_INDEX_OFFSET])

class CoefficientA():
    def __init__(self, a_center: float, a_face_arr: np.array([float])) -> None:
        self.a_center = a_center
        self.a_face_arr = a_face_arr