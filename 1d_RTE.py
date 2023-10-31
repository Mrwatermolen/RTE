import numpy as np
from matplotlib import pyplot as plt

def main():
    n = 200

    def k(x):
        if (0 <= x < 100):
            return 0.3
        return 0.6
    I0 = 1
    dz = 1e-5
    I = np.zeros(n)
    I[0] = I0
    for i in range(1, n):
        I[i] = I[i-1] * (1 - k(i-1) * dz)
    X = np.arange(0, n, 1)
    # exact

    def exact(x):
        if (x <= 100):
            return I0 * np.exp(-0.3 * x * dz)
        return I0 * np.exp(-0.3 * 100 * dz) * np.exp(-0.6 * (x - 100) * dz)
    plt.plot(X, I, label='approx')
    X_1 = np.arange(0, n, 10)
    plt.scatter(X_1, [exact(x) for x in X_1],  label='exact', c='r')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
