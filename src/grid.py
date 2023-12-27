import numpy as np
from common import *


class GridFace():
    def __init__(self, delta_a: float, delta_b: float, norm_vec) -> None:
        self.area = delta_a * delta_b
        self.norm_vec = norm_vec

    def get_area(self):
        return self.area

    def get_norm_vec(self):
        return self.norm_vec

    def calculate_d(self, s_vec: np.array([float]), omega: float) -> float:
        """D^m_j = {vec{s} \cdot vec{n}} \cdot \Omega^m
        """
        return np.dot(self.norm_vec, s_vec) * omega
        # return integrate.quad(lambda x: np.dot(self.norm_vec, s_vec) * x, omega - 0.5*delta_omega, omega + 0.5*delta_omega)[0]

    def calculate_a(self, s_vec: np.array([float]), omega: float) -> float:
        """a^m_j = area_j * D^m_j
        """
        return self.calculate_d(s_vec, omega) * self.area


class Grid():
    """Finite control volume cell\n
    V_{i,j,k}
    """

    def __init__(self, i: int, j: int, k: int, delta_x: float, delta_y: float, delta_z: float) -> None:
        """
        Args:
            i (int): index of x
            j (int): index of y
            k (int): index of z
            delta_x (float): grid size of x
            delta_y (float): grid size of y
            delta_z (float): grid size of z
        """
        self.index = np.array([i, j, k])
        # NOTE: This order isn't associated with the function closure_face_order,
        # so it is needed to be changed by manually while face_order is changed.
        self.closure_faces = [
            GridFace(delta_y, delta_z, FRONT_FACE_OFFSET),
            GridFace(delta_y, delta_z, BACK_FACE_OFFSET),
            GridFace(delta_x, delta_z, RIGHT_FACE_OFFSET),
            GridFace(delta_x, delta_z, LEFT_FACE_OFFSET),
            GridFace(delta_x, delta_y, TOP_FACE_OFFSET),
            GridFace(delta_x, delta_y, BOTTOM_FACE_OFFSET),
        ]
        self.dx = delta_x
        self.dy = delta_y
        self.dz = delta_z
        self.k_eta = 0  # default value
        self.intensity = np.array([])  # spectral radiation energy

    def _get_coff_initial_a_array(self, s_vec: np.array([float]), omega: float) -> np.array([float]):
        """the face of grid: a_j = area_j * D^m_j
        """
        return np.array([face.calculate_a(s_vec, omega)
                         for face in self.closure_faces])

    def get_coff_a_array(self, s_vec, omega) -> CoefficientA:
        """the center of grid: a_p = coff_a[coff_a > 0] + k_eta * V_p * \Omega^m
            the face of grid: a_j = coff_a[coff_a < 0] (the face of grid is also called other center of grid which around the grid_{i,j,k})
        """
        coff_a = self._get_coff_initial_a_array(s_vec, omega)
        a_p = np.sum(coff_a[coff_a > 0]) + self.k_eta * self.volume * omega
        coff_a[coff_a > 0] = 0
        return CoefficientA(a_p, coff_a)

    def get_index(self) -> np.array([int]):
        return self.index

    @property
    def volume(self) -> float:
        return self.dx * self.dy * self.dz

    def _read_k_eta(self):
        pass

    def __repr__(self) -> str:
        return f"The property of grid ( index: {self.index} volume: {self.volume} k_eta: {self.k_eta} intensity: {self.intensity} )"
