import numpy as np
import matplotlib.pyplot as plt


class DiscretizationAngle():
    def __init__(self, n) -> None:
        if (n % 2 != 0):
            raise ValueError("n must be even")
        self.num_theta = n
        self.delta_theta = np.pi / n
        self.num_phi = self._get_num_phi_array(n)
        self.num_omega = self.num_theta * (self.num_theta + 2)
        assert self.num_omega == np.sum(self.num_phi)

    def _get_num_phi_array(self, n: int) -> np.array([int]):
        if (n % 2 != 0):
            raise ValueError("n must be even")
        num = int(n / 2)
        num_phi = np.array(
            [int(4 * (i+1)) for i in range(num)] + [int(4 * (i)) for i in range(num, 0, -1)])
        return num_phi

    def get_omega(self, t: int, p: int) -> float:
        theta = self.get_theta(t)
        phi = self.get_phi(t, p)
        delta_phi = 2 * np.pi / self.num_phi[t]
        return delta_phi * (
            np.cos(theta - (self.delta_theta) / 2) -
            np.cos(theta + (self.delta_theta) / 2)
        )

    def get_vec_s(self, t: int, p: int) -> np.array([float]):
        theta_s = self.get_theta(t)
        phi_s = self.get_phi(t, p)
        return np.array([
            np.sin(theta_s) * np.cos(phi_s),
            np.sin(theta_s) * np.sin(phi_s),
            np.cos(theta_s)
        ])

    def get_theta(self, t) -> float:
        assert t < self.num_theta
        return (t+0.5) * self.delta_theta

    def get_phi(self, t, p) -> float:
        assert p < self.num_phi[t]
        delta_phi = self.get_delta_phi(t)
        return (p+0.5) * delta_phi

    def get_delta_phi(self, t) -> float:
        assert t < self.num_theta
        return 2 * np.pi / self.num_phi[t]

    def get_vec_s_array(self) -> np.array([float]):
        return np.array([self.get_vec_s(t, p) for t in range(self.num_theta) for p in range(self.num_phi[t])])

    def get_omega_array(self) -> np.array([float]):
        return np.array([self.get_omega(t, p) for t in range(self.num_theta) for p in range(self.num_phi[t])])

    # Tool for debug
    def _validate_omega_array(omega_array: np.array([float])) -> bool:
        return np.sum(omega_array) == 4 * np.pi
