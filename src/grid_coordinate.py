import numpy as np
from grid import Grid

class GridCoordinate():
    def __init__(self, origin_x: float, origin_y: float, origin_z: float, size_x, size_y, size_z, delta_x: float, delta_y: float, delta_z: float) -> None:
        self.origin_point = np.array([origin_x, origin_y, origin_z])
        self.size = np.array([size_x, size_y, size_z])
        self.end_point = self.origin_point + self.size
        self.grid_size = np.array([delta_x, delta_y, delta_z])
        self.generateGridSpace()

    def generateGridSpace(self):
        self.nx = int(self.size[0] / self.grid_size[0])
        self.ny = int(self.size[1] / self.grid_size[1])
        self.nz = int(self.size[2] / self.grid_size[2])
        self.size = np.array([self.nx, self.ny, self.nz]) * self.grid_size
        self.end_point = self.origin_point + self.size
        self.grid = np.array([Grid(i, j, k, self.grid_size[0], self.grid_size[1], self.grid_size[2], 1)
                             for i in range(self.nx) for j in range(self.ny) for k in range(self.nz)])
        assert self.grid.shape == (self.nx * self.ny * self.nz,)
        # self.grid = np.reshape(self.grid, (self.nx, self.ny, self.nz))

    def print_all_grid(self):
        print(str(self.grid))

    def get_grid(self, i, j, k) -> Grid:
        return self.grid[self.get_grid_flatten_index(i, j, k)]

    def get_grid_by_flatten_index(self, flatten_index) -> Grid:
        return self.grid[flatten_index]

    def get_grid_flatten_index(self, i: int, j: int, k: int) -> int:
        return i * self.ny * self.nz + j * self.nz + k

    def transform_grid_index_to_flatten_index(self, index: np.array([int])) -> int:
        return self.get_grid_flatten_index(index[0], index[1], index[2])
    
    @property
    def total_grid_num(self):
        return self.nx * self.ny * self.nz