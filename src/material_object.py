from common import random_string
from shape import *
    

class MaterialObject:
    def __init__(self, shape: Cube, k, name=random_string(6)):
        """

        Args:
            shape (Cube): Object Shape
            k (lambda): absorption coefficient function
            name (_type_, optional): _description_. Defaults to random_string(6).
        """
        self.shape = shape
        self.k = k
        self.name = name

    def k_eta(self,  lambda_min: float, lambda_max: float, r_vec: np.array([float])):
        """spectral absorption coefficient
        """
        return self.k(lambda_min=lambda_min, lambda_max=lambda_max, r_vec=r_vec)

    def __str__(self):
        return f"MaterialObject: name={self.name}, k={self.k}"

    def __repr__(self):
        return self.__str__()
