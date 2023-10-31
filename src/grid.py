import numpy as np
from discretization_angle import DiscretizationAngle
from common import *


class Face():
    def __init__(self, delta_a: float, delta_b: float, norm_vec) -> None:
        self.area = delta_a * delta_b
        self.norm_vec = norm_vec

    def get_area(self):
        return self.area

    def get_norm_vec(self):
        return self.norm_vec

    def calculate_d(self, s_vec: np.array([float]), omega: float) -> float:
        return np.dot(self.norm_vec, s_vec) * omega

    def calculate_a(self, s_vec: np.array([float]), omega: float) -> float:
        return self.calculate_d(s_vec, omega) * self.area


class Grid():
    def __init__(self, i: int, j: int, k: int, delta_x, delta_y, delta_z, k_eta:float,) -> None:
        self.index = np.array([i, j, k])
        self.discretization_angle = DiscretizationAngle(20)
        # NOTE: This order isn't associated with the function closure_face_order,
        # so it is needed to be changed by manually while face_order is changed.
        self.closure_faces = [
            Face(delta_y, delta_z, FRONT_FACE_OFFSET),
            Face(delta_y, delta_z, BACK_FACE_OFFSET),
            Face(delta_x, delta_z, RIGHT_FACE_OFFSET),
            Face(delta_x, delta_z, LEFT_FACE_OFFSET),
            Face(delta_x, delta_y, TOP_FACE_OFFSET),
            Face(delta_x, delta_y, BOTTOM_FACE_OFFSET),
        ]
        self.volume = delta_x * delta_y * delta_z
        self.k_eta = k_eta # spectral absorption coefficient
        self.intensity = 0

    def _get_coff_initial_a_array(self, s_vec: np.array([float]), omega: float) -> np.array([float]):
        return np.array([face.calculate_a(s_vec, omega)
                         for face in self.closure_faces])

    def get_coff_a_array(self, s_vec, omega) -> CoefficientA:
        coff_a = self._get_coff_initial_a_array(s_vec, omega)
        a_p = np.sum(coff_a[coff_a > 0]) + self.k_eta * self.volume * omega
        coff_a[coff_a > 0] = 0
        return CoefficientA(a_p, coff_a)

    def get_index(self) -> np.array([int]):
        return self.index
    
    def get_k_eta(self, lambda_min:float, lambda_max:float) -> float:
        return self.k_eta
    
    def get_volume(self) -> float:
        return self.volume
    
    def _read_k_eta(self):
        pass

    def __repr__(self) -> str:
        return f"The property of grid ( index: {self.index} volume: {self.volume} k_eta: {self.k_eta} intensity: {self.intensity} )"
