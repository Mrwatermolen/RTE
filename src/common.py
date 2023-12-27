from enum import Enum
import numpy as np
import random
import string

# the maximum number of grids per dimension
MAX_GRID_NUM_PER_DIMENSION = 1e4

# these offset can be treated as the normal vector of the face
FRONT_FACE_OFFSET = np.array([1, 0, 0])
BACK_FACE_OFFSET = np.array([-1, 0, 0])
RIGHT_FACE_OFFSET = np.array([0, 1, 0])
LEFT_FACE_OFFSET = np.array([0, -1, 0])
TOP_FACE_OFFSET = np.array([0, 0, 1])
BOTTOM_FACE_OFFSET = np.array([0, 0, -1])

# define the order of the faces closuring the center of grid
FACE_ORDER_INDEX_OFFSET = np.array([
    FRONT_FACE_OFFSET,
    BACK_FACE_OFFSET,
    RIGHT_FACE_OFFSET,
    LEFT_FACE_OFFSET,
    TOP_FACE_OFFSET,
    BOTTOM_FACE_OFFSET
])


def closure_face_order(i: int, j: int, k: int) -> np.array([int]):
    """get the around faces of the grid (i, j, k)

    Args:
        i,j,k (int): the index of grid

    Returns:
        np.array([int]): the index of around faces of the grid (i, j, k)
    """
    return np.array([o + [i, j, k] for o in FACE_ORDER_INDEX_OFFSET])


class CoefficientA():
    """encapsulates the row vectors of the coefficient matrix A in the FVM——RTE system of linear equations Ax=b.
    a_face_arr obey the order of the faces closuring the center of grid.
    """

    def __init__(self, a_center: float, a_face_arr: np.array([float])) -> None:
        self.a_center = a_center
        self.a_face_arr = a_face_arr


class FloatCompareOperator(Enum):
    EQUAL = 0
    LESS_THAN = 1
    GREATER_THAN = 2
    LESS_THAN_OR_EQUAL = 3
    GREATER_THAN_OR_EQUAL = 4


def compare_float(a: np.array([float]), op: FloatCompareOperator, b: np.array([float]), epsilon: float = 1e-9) -> bool:
    """compare two float array. If the difference between two float is less than epsilon, we think they are equal.

    Args:
        a (np.array): _description_
        op (FloatCompareOperator): _description_
        b (np.array): _description_
        epsilon (float, optional): _description_. Defaults to 1e-9.

    Raises:
        Exception: if op is invalid

    Returns:
        bool: a op b
    """
    if op == FloatCompareOperator.EQUAL:
        return (np.abs(a - b) < epsilon).all()
    if op == FloatCompareOperator.LESS_THAN:
        return (a < b).all()
    if op == FloatCompareOperator.GREATER_THAN:
        return (a > b).all()
    if op == FloatCompareOperator.LESS_THAN_OR_EQUAL:
        return compare_float(a, FloatCompareOperator.EQUAL, b, epsilon) or compare_float(a, FloatCompareOperator.LESS_THAN, b, epsilon)
    if op == FloatCompareOperator.GREATER_THAN_OR_EQUAL:
        return compare_float(a, FloatCompareOperator.EQUAL, b, epsilon) or compare_float(a, FloatCompareOperator.GREATER_THAN, b, epsilon)

    raise Exception("Invalid FloatCompareOperator")


def random_string(length: int):
    return ''.join(random.choice(string.ascii_letters) for i in range(length))
