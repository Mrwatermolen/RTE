from fvm_rte_solver import FvmRteSolver
from grid_coordinate import GridCoordinate
from grid import Grid
from discretization_angle import DiscretizationAngle
from black_body import BlackBody
from material_object import MaterialObject
from shape import Cube, Sphere
from matplotlib import pyplot as plt
import numpy as np
import os
import random
import string
import time


def random_string(length: int):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))


def test_black_body():
    b = BlackBody(298)
    print(b.c1)
    print(b.c2)


def make_fvm_solver(grid_size, discretization_angle, z_min, z_max):
    g_coord = GridCoordinate(grid_size, grid_size, grid_size)
    d_angle = DiscretizationAngle(discretization_angle)
    f = FvmRteSolver()
    f.set_config_before_running(lambda_min=1e-7, lambda_max=1e-2, temperature=500,
                                grid_coordinate=g_coord, discretization_angle=d_angle)

    def black_k(lambda_min, lambda_max, r_vec): return 1
    def transparent_k(lambda_min, lambda_max, r_vec): return 0
    f.addObject(MaterialObject(
        shape=Cube(origin=np.array([-4, -4, z_min]), end=np.array([4, 4, z_max])), k=transparent_k, name="background"
    ))
    f.addObject(MaterialObject(
        shape=Sphere(center=np.array([0, 0, 0]), radius=1), k=black_k, name="black_body"
    ))
    f.run()
    return f


def fvm_result_z(size, s_arr, w_arr, g_data, shape, index: Grid, z_min, z_max):
    y = np.array([])
    for i in range(shape[2]):
        ii = index.get_index()[0] * shape[1] * shape[2] + \
            index.get_index()[1] * shape[2] + i
        g = g_data[ii]
        q_x = np.sum(g * w_arr * np.dot(s_arr, np.array([1, 0, 0])))
        q_y = np.sum(g * w_arr * np.dot(s_arr, np.array([0, 1, 0])))
        q_z = np.sum(g * w_arr * np.dot(s_arr, np.array([0, 0, 1])))
        y = np.append(y, np.sqrt(q_x**2 + q_y**2 + q_z**2))
    x = np.arange(z_min + 0.5 * size, z_max + 0.5 * size, size)
    x = x - x[index.get_index()[2]]
    return x, y


def test_fvm_3d(out_dir: str, grid_size=0.5, discretization_angle=4, z_min=-2, z_max=12):
    f = make_fvm_solver(grid_size, discretization_angle, z_min, z_max)
    s_arr = f.discretization_angle.get_vec_s_array()
    w_arr = f.discretization_angle.get_omega_array()
    g_data = np.array([g.intensity for g in f.grid_coord.grid])
    # out_dir = os.path.join(out_dir, "data_" + time.strftime(
    # "%Y%m%d%H%M%S") + "_" + random_string(5))
    # os.mkdir(out_dir)
    # np.save(f"{out_dir}/s_arr.npy", s_arr)
    # np.save(f"{out_dir}/w_arr.npy", w_arr)
    # np.save(f"{out_dir}/g_data.npy", g_data)
    # np.save(f"{out_dir}/grid_space_shape.npy", f.grid_coord.grid_space_shape)

    x, y = fvm_result_z(grid_size, s_arr, w_arr, g_data, f.grid_coord.grid_space_shape,
                        f.grid_coord.get_grid_by_point(np.array([0, 0, 0])), z_min, z_max)
    plt.figure()
    plt.plot(x, y, 'o-', label="numerical")
    plt.grid()
    plt.xlim([1, z_max])
    plt.ylim([0, 1400])
    T = 500
    b = BlackBody(T)
    I_b = b.integrate_radiant_exitance_band(1e-7, 1e-2) / np.pi
    a = np.pi / (x**2)
    I = a * I_b
    plt.plot(x, I, 'k--', label="exact")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    pwd = os.getcwd()
    test_fvm_3d(os.path.join(pwd, "tmp"), grid_size=0.25,
                discretization_angle=4, z_min=-4, z_max=4)
