import numpy as np

from common import *


class Cube(object):
    def __init__(self, origin, end) -> None:
        self.origin = origin
        self.end = end
        self.size = end - origin
        self.center = (origin + end) / 2
        self.eps = np.min(self.size) / (MAX_GRID_NUM_PER_DIMENSION * 10)

    # @property
    # def origin(self):
    #     return self.origin

    # @property
    # def end(self):
    #     return self.end

    # @property
    # def size(self):
    #     return self.size

    # @property
    # def center(self):
    #     return self.center

    def isPointInside(self, point) -> bool:
        return compare_float(self.origin, FloatCompareOperator.LESS_THAN_OR_EQUAL, point, self.eps)


class Sphere(Cube):
    def __init__(self, center, radius: float) -> None:
        super(Sphere, self).__init__(center - radius, center + radius)
        self.center = center
        self.radius = radius

    # @property
    # def center(self):
    #     return self.center

    # @property
    # def radius(self):
    #     return self.radius

    def isPointInside(self, point) -> bool:
        return compare_float(np.linalg.norm(point - self.center), FloatCompareOperator.LESS_THAN_OR_EQUAL, self.radius, self.eps)
