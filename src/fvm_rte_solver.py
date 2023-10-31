from black_body import BlackBody
from discretization_angle import DiscretizationAngle
from grid import Grid
from grid_coordinate import GridCoordinate
from common import *
import numpy as np


class FvmRteSolver():
    """To solve the problem "Ax = b"
    """

    def __init__(self, grid_coordinate: GridCoordinate, theta_num: int) -> None:

        self.grid_coord = grid_coordinate
        self.grid_num = [self.grid_coord.nx,
                         self.grid_coord.ny, self.grid_coord.nz]

        self.discretization_angle = DiscretizationAngle(theta_num)

        self.lambda_max = 1e-4
        self.lambda_min = 1e-6
        self.T = 1000
        self.black_body = BlackBody(self.T)
        self.B = self.calculate_Black_Body_Radiant_exitance_band()

    def calculate_Black_Body_Radiant_exitance_band(self) -> float:
        return self.black_body.integrate_radiant_exitance_band(self.lambda_min, self.lambda_max)

    def run(self):
        num_theta = self.discretization_angle.num_theta
        for t in range(num_theta):
            for p in range(self.discretization_angle.num_phi[t]):
                s_vec = self.discretization_angle.get_vec_s(t, p)
                omega = self.discretization_angle.get_omega(t, p)
                self.solver_per_omega(s_vec, omega)
                return

    def solver_per_omega(self, s_vec, omega):
        a, b = self.calculate_coff_matrix(s_vec, omega)
        self.coff_a_matrix = a
        self.coff_b_matrix = b

    def calculate_coff_matrix(self, s_vec, omega):
        total_grid_num = self.grid_coord.total_grid_num
        coff_a_matrix = np.zeros((total_grid_num, total_grid_num))
        coff_b_matrix = np.zeros((total_grid_num, 1))
        for i in range(1, self.grid_num[0]-1):
            for j in range(1, self.grid_num[1]-1):
                for k in range(1, self.grid_num[2]-1):
                    g = self.grid_coord.get_grid(i, j, k)
                    g_index = self.grid_coord.get_grid_flatten_index(
                        *g.get_index())
                    face_index = np.array(
                        [self.grid_coord.transform_grid_index_to_flatten_index(index=l) for l in closure_face_order(i, j, k)])
                    a = self.calculate_a(g, s_vec, omega)
                    b = self.calculate_b(g, omega)
                    coff_a_matrix[g_index, g_index] = a.a_center
                    coff_a_matrix[g_index, face_index] = a.a_face_arr
                    coff_b_matrix[g_index] = b
        return coff_a_matrix, coff_b_matrix

    def calculate_b(self, grid: Grid, omega: float):
        v = grid.get_volume()
        k_eta = grid.get_k_eta(self.lambda_min, self.lambda_max)
        return k_eta * self.B * v * omega

    def calculate_a(self, grid: Grid, s_vec: np.array, omega: float) -> CoefficientA:
        return grid.get_coff_a_array(s_vec, omega)


def test_fvm_rte_solver():
    g = GridCoordinate(0, 0, 0, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5)
    f = FvmRteSolver(g, 20)
    f.run()
    view_grid_file = "view_grid.txt"
    view_a_matrix_file = "view_a_matrix.txt"
    view_b_matrix_file = "view_b_matrix.txt"
    with open(view_grid_file, "w") as file:
        file.write(str(g.grid))
    with open(view_a_matrix_file, "w") as file:
        file.write(str(f.coff_a_matrix))
    with open(view_b_matrix_file, "w") as file:
        file.write(str(f.coff_b_matrix))


if __name__ == "__main__":
    test_fvm_rte_solver()
