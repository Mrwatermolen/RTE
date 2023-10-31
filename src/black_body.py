import numpy as np
import matplotlib.pyplot as plt
from scipy import constants, integrate


class BlackBody:
    def __init__(self, T: float) -> None:
        self.T = T  # Temperature (unit: K)

        self.k_b = constants.Boltzmann  # Boltzmann constant (unit: J/K)
        self.h = constants.Planck  # Planck constant (unit: J*s)
        self.c = constants.speed_of_light  # Speed of light (unit: m/s)
        # Stefan–Boltzmann constant (unit: J s^-1 m^-2 K^-4)
        self.sigma = constants.Stefan_Boltzmann
        self.b = constants.Wien  # Wien's displacement constant (unit: m*K)

        self.c1 = 2 * np.pi * self.h * self.c ** 2  # First radiation constant
        self.c2 = self.h * self.c / self.k_b  # Second radiation constant

    @property
    def temperature(self):
        """Temperature (unit: K)
        """
        return self.T

    @property
    def lambda_max(self):
        """Wavelength of maximum radiance (unit: m)
        """
        return self.get_lambda_max(self.T)

    @property
    def boltzmann_constant(self):
        """Boltzmann constant (unit: J/K)
        """
        return self.k_b

    @property
    def planck_constant(self):
        """Planck constant (unit: J*s)
        """
        return self.h

    @property
    def speed_of_light(self):
        """Speed of light (unit: m/s)
        """
        return self.c

    @property
    def stefan_boltzmann_constant(self):
        """Stefan-Boltzmann constant (unit: J s^-1 m^-2 K^-4)
        """
        return self.sigma

    @property
    def first_radiation_constant(self):
        return self.c1

    @property
    def second_radiation_constant(self):
        return self.c2

    @property
    def wien_displacement_constant(self):
        return self.b

    def get_lambda_max(self, T=None) -> float:
        if T is None:
            T = self.T
        return self.b / T

    def get_radiant_exitance(self, lambda_, T=None):
        """Get radiant exitance (unit: W m^-2 m^-1)
        see more: https://en.wikipedia.org/wiki/Radiant_exitance
        """
        if T is None:
            T = self.T
        return self.c1 / (lambda_ ** 5 * (np.exp(self.c2 / (lambda_ * T)) - 1))

    def get_radiance(self, lambda_):
        """Get radiance (unit: W m^-2 sr^-1 m^-1)
        see more: https://en.wikipedia.org/wiki/Radiance
        """
        return self.get_radiant_exitance(lambda_) / np.pi

    def get_max_radiant_exitance(self, T=None):
        """Get max radiance exitance at this temperature (unit: W m^-2 m^-1)
        """
        if T is None:
            T = self.T
        B = self.c1 * self.b**-5 / (np.exp(self.c2 / self.b) - 1)
        return B * self.T ** 5

    def get_maximum_radiance(self):
        """Get max radiance at this temperature (unit: W m^-2 sr^-1 m^-1)
        """
        return self.get_max_radiant_exitance() / np.pi

    def radiant_exitance_with_all_lambda(self):
        """Get integrate radiance (unit: W m^-2 sr^-1)
        """
        return self.sigma * self.T ** 4

    def integrate_radiant_exitance_band(self, lambda_min, lambda_max, T=None) -> float:
        """Integrate radiance (unit: W m^-2 sr^-1)
        """
        if (T is None):
            T = self.T
        return integrate.quad(self.get_radiant_exitance, lambda_min, lambda_max, args=(T))[0]


def test_black_body():
    T_min = 200
    T_max = 6000
    T_step = 10
    T = np.logspace(np.log10(T_min), np.log10(T_max), T_step)
    T = np.around(T)

    l_min = 0.1e-6
    l_max = 100e-6
    l = np.logspace(np.log10(l_min), np.log10(l_max), 100)

    max_radiance = np.zeros((T_step, 2))
    max_radiance_err = np.zeros((T_step))

    stefan_boltzmann_law_data = np.zeros((T_step, 2))

    for i in range(len(T)):
        t = T[i]
        black = BlackBody(t)
        radiance = black.get_radiance(l)

        max_radiance[i, 0] = black.lambda_max
        max_radiance[i, 1] = black.get_radiance(max_radiance[i, 0])
        max_radiance_err[i] = np.abs(
            max_radiance[i, 1] - black.get_maximum_radiance())

        stefan_boltzmann_law_data[i, 0] = black.integrate_radiant_exitance_band(
            l_min, l_max)
        stefan_boltzmann_law_data[i, 1] = np.abs(
            black.radiant_exitance_with_all_lambda() - stefan_boltzmann_law_data[i, 0])

        plt.loglog(l * 1e6, radiance, label=str(t) + "K")
        plt.text(max_radiance[i, 0] * 1e6, max_radiance[i, 1], str(t) + "K")

    plt.loglog(max_radiance[:, 0] * 1e6, max_radiance[:, 1], "o-")
    plt.ylim([1e-4, 1e15])
    plt.ylabel("Radiance (W m^-2 sr^-1 m^-1)")
    plt.xlim([0.1, 100])
    plt.xlabel("Wavelength (um)")
    plt.grid(True)
    # plt.legend()

    plt.figure()
    plt.plot(T, max_radiance_err, "o-")
    plt.title("The abs error of the max radiance at each temperature")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Error (W m^-2 sr^-1 m^-1)")

    plt.figure()
    plt.plot(T, max_radiance_err / max_radiance[:, 1], "o-")
    plt.title("The relative error of the max radiance at each temperature")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Relative error")

    plt.figure()
    plt.plot(T, stefan_boltzmann_law_data[:, 1], "o-")
    plt.title("The abs error of the Stefan-Boltzmann law at each temperature")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Error (W m^-2)")

    plt.figure()
    plt.plot(T, stefan_boltzmann_law_data[:, 1] /
             stefan_boltzmann_law_data[:, 0], "o-")
    plt.title("The relative error of the Stefan-Boltzmann law at each temperature")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Relative error")

    plt.show()


if __name__ == "__main__":
    test_black_body()
