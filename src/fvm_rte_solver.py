from matplotlib import pyplot as plt
from black_body import BlackBody
from discretization_angle import DiscretizationAngle
from grid import GridFace, Grid
from grid_coordinate import GridCoordinate
from common import *
import numpy as np
from scipy.sparse import linalg
from tqdm import tqdm
from material_object import MaterialObject
from shape import Cube, Sphere


class FvmRteSolver():
    """To solve the problem "Ax = b"
    TODO: handle boundary condition
    """

    def __init__(self) -> None:
        self.black_body = BlackBody(0)

    def set_config_before_running(self, lambda_min: float, lambda_max: float, temperature: float, grid_coordinate: GridCoordinate, discretization_angle: DiscretizationAngle):
        """_summary_

        Args:
            lambda_min (float): unit: m
            lambda_max (float): unit: m
            temperature (float): unit: K
            grid_coordinate (GridCoordinate): _description_
            discretization_angle (DiscretizationAngle): _description_
        """
        self.set_wavelength_band(np.array([lambda_min, lambda_max]))
        self.set_temperature(temperature)
        self.set_grid_coordinate(grid_coordinate)
        self.set_discretization_angle(discretization_angle)

    def set_discretization_angle(self, discretization_angle: DiscretizationAngle):
        self.discretization_angle = discretization_angle

    def set_temperature(self, T: float):
        self.T = T
        # TODO: set black body

    def set_grid_coordinate(self, grid_coordinate: GridCoordinate):
        self.grid_coord = grid_coordinate

    def set_wavelength_band(self, wavelength_band: np.array([float])):
        self.lambda_min = wavelength_band[0]
        self.lambda_max = wavelength_band[-1]
        self.lambda_band = wavelength_band

    def addObject(self, obj):
        self.grid_coord.addObject(obj)

    def calculate_black_body_radiant_exitance_band(self) -> float:
        return self.black_body.integrate_radiant_exitance_band(self.lambda_min, self.lambda_max, self.T)

    def run(self, lambda_min: float = None, lambda_max: float = None):
        lambda_min = self.lambda_min if lambda_min is None else lambda_min
        lambda_max = self.lambda_max if lambda_max is None else lambda_max

        # init
        self.B = self.calculate_black_body_radiant_exitance_band()
        self.grid_coord.generateGridSpace()
        self.grid_coord.initObject(lambda_min, lambda_max)
        index = self.grid_coord.get_grid_by_point(np.array([0, 0, 0]))
        num_omega = self.discretization_angle.num_omega
        for grid in self.grid_coord.grid:
            grid.intensity = np.zeros((num_omega))

        self._solver_RTE()

    def _solver_RTE(self):
        num_theta = self.discretization_angle.num_theta
        pbar = tqdm(total=self.discretization_angle.num_omega)
        for t in range(num_theta):
            for p in range(self.discretization_angle.num_phi_arr[t]):
                s_vec = self.discretization_angle.get_vec_s(t, p)
                omega = self.discretization_angle.get_omega_by_theta_index(t)
                omega_index = self.discretization_angle.get_omega_index(t, p)
                x = self._solve_per_omega(s_vec, omega)
                for grid in self.grid_coord.grid:
                    grid.intensity[omega_index] = x[self.grid_coord.get_grid_flatten_index(
                        *grid.get_index())]
                pbar.update(1)
        pbar.close()

    def _solve_per_omega(self, s_vec, omega):
        """solve RTE for one omega

        Args:
            s_vec (_type_): solid angle vector
            omega (_type_): solid angle

        Raises:
            Exception: Bicgstab can't converge

        Returns:
            x: RTE: Ax=b
        """
        self.coff_a_matrix, self.coff_b_matrix = self._calculate_coff_matrix(
            s_vec, omega)  # get the A and b
        x, exit_code = linalg.bicgstab(
            self.coff_a_matrix, self.coff_b_matrix, tol=1e-3)  # solve the equation
        # print(self.coff_a_matrix[22,:])
        # print(self.coff_b_matrix)
        # print(x)
        # if exit_code != 0:
            # print("Solver failed to converge")
            # raise Exception("Solver failed to converge")
        return x

    def _calculate_coff_matrix(self, s_vec, omega):
        # NOTE: Indeed, we don't need to calculate the coefficient which is on the boundary.
        total_grid_num = self.grid_coord.grid_num
        grid_space_shape = self.grid_coord.grid_space_shape
        coff_a_matrix = np.zeros((total_grid_num, total_grid_num))
        coff_b_matrix = np.zeros((total_grid_num, 1))

        for i in range(1, grid_space_shape[0]-1):
            for j in range(1, grid_space_shape[1]-1):
                for k in range(1, grid_space_shape[2]-1):
                    g = self.grid_coord.get_grid(i, j, k)
                    g_index = self.grid_coord.get_grid_flatten_index(
                        i, j, k)
                    face_index_ijk = np.array(
                        [ijk for ijk in closure_face_order(i, j, k)])
                    face_index = np.array(
                        [self.grid_coord.transform_grid_index_to_flatten_index(index=f) for f in face_index_ijk])
                    a = self._calculate_a(g, s_vec, omega)
                    b = self._calculate_b(g, omega)
                    coff_a_matrix[g_index, g_index] = a.a_center
                    coff_a_matrix[g_index, face_index] = a.a_face_arr
                    coff_b_matrix[g_index] = b

        return coff_a_matrix, coff_b_matrix

    def _calculate_a(self, grid: Grid, s_vec: np.array, omega: float) -> CoefficientA:
        return grid.get_coff_a_array(s_vec, omega)

    def _calculate_b(self, grid: Grid, omega: float):
        v = grid.volume
        k_eta = grid.k_eta
        return k_eta * self.B * v * omega / np.pi

    def _get_radiative_flux_density_for_one(self, intensity: np.array([float]), norm_vec: np.array([float])) -> float:
        num_omega = len(intensity)
        s_vec_arr = self.discretization_angle.get_vec_s_array()
        d_omega = np.array(
            [self.discretization_angle.get_omega(i) for i in range(num_omega)])
        res = np.sum(intensity * d_omega * np.dot(s_vec_arr, norm_vec))
        # TODO: This is a temporary solution
        while res.shape != ():
            res = np.sum(res)
        return res

    def get_radiative_flux_density(self, grid_arr: np.array([Grid]), norm_vec) -> np.array([float]):
        return np.array([self._get_radiative_flux_density_for_one(grid.intensity, norm_vec) for grid in grid_arr])

    def only_for_debug_plot_radiative_flux_density(self):
        # TODO: unfinished
        norm_x = np.array([1, 0, 0])
        # get all grid in X = 0 plane
        grid_arr = np.array(
            [grid for grid in self.grid_coord.grid if grid.get_index()[0] == 0])
        yz_shape = (self.grid_coord.ny, self.grid_coord.nz)
        v = self.get_radiative_flux_density(grid_arr, norm_x).reshape(yz_shape)
        plt.imshow(v)
        print(np.unique(v))
        plt.show()


def test_fvm_rte_solver():
    # Calculate domain size is 5*5*5
    # Background is transparent
    # Create a black body with a cube in the center
    size = 0.5
    g = GridCoordinate(size, size, size)
    d = DiscretizationAngle(20)
    f = FvmRteSolver()
    f.set_config_before_running(lambda_min=1e-7, lambda_max=1e-2,
                                temperature=500, grid_coordinate=g, discretization_angle=d)

    # You have to define the spectral absorption coefficient function like this
    # def function_name(lambda_min, lambda_max, r_vec): { function_body }
    # IMPORTANT AND TODO: The name of args of the function must be (lambda_min, lambda_max, r_vec)
    def black_k(lambda_min, lambda_max, r_vec): return 1
    def transparent_k(lambda_min, lambda_max, r_vec): return 0
    f.addObject(MaterialObject(
        shape=Cube(origin=np.array([-2, -2, -2]), end=np.array([2, 2, 10])), k=transparent_k, name="background"
    ))
    f.addObject(MaterialObject(
        shape=Sphere(center=np.array([0, 0, 0]), radius=1), k=black_k, name="black_body"
    ))
    f.run()
    index = f.grid_coord.get_grid_by_point(np.array([0, 0, 0]))
    print(index.get_index())
    s_arr = f.discretization_angle.get_vec_s_array()
    w_arr = f.discretization_angle.get_omega_array()
    # save data
    out_dir =  "data_" + random_string(5)
    import os
    os.mkdir(out_dir)
    np.save(f"{out_dir}/s_arr.npy", s_arr)
    np.save(f"{out_dir}/w_arr.npy", w_arr)
    g_data = np.array([g.intensity for g in f.grid_coord.grid])
    np.save(f"{out_dir}/g_data.npy", g_data)
    np.save(f"{out_dir}/grid_space_shape.npy", f.grid_coord.grid_space_shape)

def plot(data_dir:str):
    s_arr = np.load(f"{data_dir}/s_arr.npy")
    w_arr = np.load(f"{data_dir}/w_arr.npy")
    g_data = np.load(f"{data_dir}/g_data.npy")
    grid_space_shape = np.load(f"{data_dir}/grid_space_shape.npy")
    index = Grid(4,4,4,0,0,0)
    plot_heat_flux_with_r(s_arr,w_arr,g_data, index, 0.5, grid_space_shape)
    
def plot_heat_flux_with_r(s_arr,w_arr,g_data, index, size, shape):
    y = np.array([])
    for i in range(shape[2]):
        ii = index.get_index()[0] * shape[1] * shape[2] + index.get_index()[1] * shape[2] + i
        g = g_data[ii]
        q_x =np.sum(g * w_arr * np.dot(s_arr, np.array([1, 0, 0]))) 
        q_y = np.sum(g * w_arr * np.dot(s_arr, np.array([0, 1, 0])))
        q_z = np.sum(g * w_arr * np.dot(s_arr, np.array([0, 0, 1])))
        print(q_x, q_y, q_z)
        y = np.append(y, np.sqrt(q_x**2 + q_y**2 + q_z**2))
    x = np.arange(-2 + 0.5 * size, 10 + 0.5 * size, size)
    x = x - x[index.get_index()[2]]
    plt.figure()
    plt.plot(x,y, 'o-')
    plt.grid()
    plt.xlim([1, 8])
    plt.ylim([0,1000])
    plt.show()
    

if __name__ == "__main__":
    # test_fvm_rte_solver()
    plot("data_hZHDA")
