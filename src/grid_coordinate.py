import numpy as np
from grid import Grid
from material_object import MaterialObject


class GridCoordinate():
    """discretization of the space
    """

    def __init__(self, delta_x: float, delta_y: float, delta_z: float) -> None:
        self.grid_size = np.array([delta_x, delta_y, delta_z])
        self.object_list = None

    def generateGridSpace(self):
        if self.object_list is None or len(self.object_list) == 0:
            raise Exception("No object in the grid space")
        self._calculate_grid_space()

        self.grid = np.array([Grid(i, j, k, self.grid_size[0], self.grid_size[1], self.grid_size[2])
                             for i in range(self.nx) for j in range(self.ny) for k in range(self.nz)])
        # assert (self.grid.shape == (self.grid_num,))
        self.grid_space_shape = np.array([self.nx, self.ny, self.nz])

    def _calculate_grid_space(self):
        origin_p = np.array([np.inf, np.inf, np.inf])
        end_p = np.array([-np.inf, -np.inf, -np.inf])
        for obj in self.object_list:
            origin_p = np.minimum(origin_p, obj.shape.origin)
            end_p = np.maximum(end_p, obj.shape.end)
        self.origin_point = origin_p
        self.size = end_p - origin_p
        self.nx, self.ny, self.nz = np.ceil(
            self.size / self.grid_size).astype(int)
        self.grid_num = self.nx * self.ny * self.nz
        self.end_point = self.origin_point + self.grid_num * self.grid_size

    def initObject(self, lambda_min: float, lambda_max: float):
        for obj in self.object_list:
            for g in self.grid:
                g_index = g.get_index()
                g_center = self.origin_point + self.grid_size * (g_index + 0.5)
                if obj.shape.isPointInside(g_center):
                    # TODO: add r_vec
                    g.k_eta = obj.k_eta(
                        lambda_min, lambda_max, None)

    def addObject(self, obj: MaterialObject):
        if self.object_list is None:
            self.object_list = [obj]
            return
        self.object_list.append(obj)

    def get_grid(self, i, j, k) -> Grid:
        return self.grid[self.get_grid_flatten_index(i, j, k)]

    def get_grid_by_flatten_index(self, flatten_index) -> Grid:
        return self.grid[flatten_index]

    def get_grid_flatten_index(self, i: int, j: int, k: int) -> int:
        return i * self.ny * self.nz + j * self.nz + k

    def transform_grid_index_to_flatten_index(self, index: np.array([int])) -> int:
        return self.get_grid_flatten_index(index[0], index[1], index[2])
