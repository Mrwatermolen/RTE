import grid
import discretization_angle
from common import *
from scipy import integrate

def test_grid():
    # validate the d
    d = discretization_angle.DiscretizationAngle(10)
    delta_l = 1
    g = grid.Grid(0, 0, 0, delta_l, delta_l, delta_l)
    face = g.closure_faces
    num_theta = d.num_theta
    for t in range(num_theta):
        for p in range(d.num_phi_arr[t]):
            theta = d.get_theta(t)
            phi = d.get_phi(t, p)
            s_vec = d.get_vec_s(t, p)
            omega = d.get_omega(t)
            dt = d.delta_theta
            dp = d.get_delta_phi(t)
            for norm_vec in FACE_ORDER_INDEX_OFFSET:
                d11 = np.dot(norm_vec, s_vec) * omega
                t_min = theta - dt/2
                t_max = theta + dt/2
                p_min = phi - dp/2
                p_max = phi + dp/2

                
                d2 = 0
                # if norm_vec == RIGHT_FACE_OFFSET:
                if (norm_vec == TOP_FACE_OFFSET).all():
                    def l(tt): return np.cos(theta) * np.sin(theta)
                    d1 = integrate.quad(l, t_min, t_max)
                    d2 = 0.5 * np.sin(2*theta) * np.sin(dt) * dp
                    d3 = 0.25 * (np.sin(2*t_min) - np.sin(2*t_max)) * dp
                    print(d11, d1[0] * dp)
                    


def main():
    test_grid()


if __name__ == "__main__":
    main()
