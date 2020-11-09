from ilqr.costs.finite_diff import FiniteDiffCost
from ilqr.costs.exact import ExactCost
import numpy as np

def finite_diff_cost_fn():
	cost = FiniteDiffCost(lambda x, u, i: 2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2 + u[0] ** 2,
	                      lambda x, i: (2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2),
	                      4, 1, use_multiprocessing = True)
	return cost


def l(x, u, i, terminal=False):
        if terminal:
            return (2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2)
        return 2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2 + u[0] ** 2
def l_x(x, u, i, terminal=False):
        if terminal:
            return np.array([4, 20, 2, 2]) * x
        return np.array([4, 20, 2, 2]) * x
def l_u(x, u, i, terminal = False):
    if terminal:
        return np.zeros(1)
    return np.array([2]) * u
def l_xx(x, u, i, terminal=False):
    deriv = np.zeros((4, 4))
    deriv[0][0] = 4
    deriv[1][1] = 20
    deriv[2][2] = 2
    deriv[3][3] = 2
    return deriv
def l_ux(x, u, i, terminal=False):
    return np.zeros((1, 4))
def l_uu(x, u, i, terminal=False):
    if terminal:
        return np.zeros((1, 1))
    return np.array([[2]])

def exact_cost_fn():
    cost = ExactCost(l, l_x, l_u, l_xx, l_ux, l_uu, use_multiprocessing = True)
    return cost

class Config:
	xmlpath = 'ilqr/xmls/inverted_pendulum.xml'
	action_bounds = [-1, 1]
	cost_fn = exact_cost_fn()
