import numpy as np
import matplotlib.pyplot as plt


class DiscretizationAngle():
    """A class to discretize the angle space.
    the discretization scheme ref: https://doi.org/10.1016/S0017-9310(99)00211-2
    """

    def __init__(self, num_theta: int) -> None:
        """

        Args:
            num_theta (int): the number of discretization in theta direction. And the number of phi is 4, 8, 12, ..., 2N - 4, 2N, 2N , 2N - 4, ..., 8, 4

        Raises:
            ValueError: if num_theta is odd
        """
        if (num_theta % 2 != 0):
            raise ValueError("n must be even")
        self.num_theta = num_theta
        self.delta_theta = np.pi / num_theta
        self.num_phi_arr = self._get_num_phi_array(num_theta)
        self.num_omega = self.num_theta * (self.num_theta + 2)
        assert self.num_omega == np.sum(self.num_phi_arr)

    def _get_num_phi_array(self, n: int) -> np.array([int]):
        if (n % 2 != 0):
            raise ValueError("n must be even")
        num = int(n / 2)
        num_phi = np.array(
            [int(4 * (i+1)) for i in range(num)] + [int(4 * (i)) for i in range(num, 0, -1)])
        return num_phi

    def get_omega_by_theta_index(self, theta_index: int) -> float:
        theta = self.get_theta(theta_index)
        delta_phi = self.get_delta_phi(theta_index)
        return delta_phi * (
            np.cos(theta - (self.delta_theta) / 2) -
            np.cos(theta + (self.delta_theta) / 2)
        )

    def get_omega(self, omega_index: int) -> float:
        theta_index = self.get_theta_index_from_omega_index(omega_index)
        return self.get_omega_by_theta_index(theta_index)

    def get_vec_s(self, theta_index: int, phi_index: int) -> np.array([float]):
        """get the solid angle vector
        """
        theta_s = self.get_theta(theta_index)
        phi_s = self.get_phi(theta_index, phi_index)
        return np.array([
            np.sin(theta_s) * np.cos(phi_s),
            np.sin(theta_s) * np.sin(phi_s),
            np.cos(theta_s)
        ])

    def get_theta(self, t: int) -> float:
        assert t < self.num_theta
        return (t+0.5) * self.delta_theta

    def get_phi(self, t: int, p: int) -> float:
        assert p < self.num_phi_arr[t]
        delta_phi = self.get_delta_phi(t)
        return (p+0.5) * delta_phi

    def get_delta_phi(self, t: int) -> float:
        assert t < self.num_theta
        return 2 * np.pi / self.num_phi_arr[t]

    def get_theta_array(self) -> np.array([float]):
        return np.array([self.get_theta(t) for t in range(self.num_theta)])

    def get_omega_array(self) -> np.array([float]):
        return np.array([self.get_omega(t, p) for t in range(self.num_theta) for p in range(self.num_phi_arr[t])])

    def get_vec_s_array(self) -> np.array([float]):
        return np.array([self.get_vec_s(t, p) for t in range(self.num_theta) for p in range(self.num_phi_arr[t])])

    def get_omega_array(self) -> np.array([float]):
        return np.array([self.get_omega_by_theta_index(t) for t in range(self.num_theta) for p in range(self.num_phi_arr[t])])

    def get_omega_index(self, theta_index: int, phi_index: int) -> int:
        return np.sum(self.num_phi_arr[:theta_index]) + phi_index

    def get_theta_index_from_omega_index(self, omega_index: int) -> int:
        upper_index = 0
        for res, add_index in enumerate(self.num_phi_arr):
            upper_index += add_index
            if (omega_index < upper_index):
                break
        return res

    def get_phi_index_from_omega_index(self, omega_index: int) -> int:
        theta_index = self.get_theta_index_from_omega_index(omega_index)
        return (omega_index - np.sum(self.num_phi_arr[:theta_index]))

    def get_phi_index_from_omega_index_and_theta_index(self, omega_index, theta_index) -> int:
        return omega_index - np.sum(self.num_phi_arr[:theta_index])

    # Tool for debug
    def _validate_omega_array(omega_array: np.array([float])) -> bool:
        return np.sum(omega_array) == 4 * np.pi


def test_d():
    import scipy.integrate as integrate
    def f(x): return np.sin(x)
    d = DiscretizationAngle(10)
    o_arr = d.get_omega_array()
    for t in range(d.num_theta):
        for p in range(d.num_phi_arr[t]):
            o_i = d.get_omega_index(t, p)
            dt = d.delta_theta
            dp = d.get_delta_phi(t)
            theta = d.get_theta(t)
            phi = d.get_phi(t, p)

            o_1 = 2 * dp * np.sin(theta) * np.sin(dt / 2)
            o_2 = integrate.quad(f, theta - dt/2, theta + dt/2)[0] * dp
            assert np.isclose(o_arr[o_i], o_1).all()
            assert np.isclose(o_arr[o_i], o_2).all()


if __name__ == "__main__":
    test_d()
