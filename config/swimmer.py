from ilqr.costs.finite_diff import FiniteDiffCost
from ilqr.costs.exact import ExactCost
import numpy as np

def l(x, u, i):
    action_cost = np.square(u).sum()
    vel_cost = 100 * ((x[5] - 10) ** 2)
    return action_cost + vel_cost


def l_exact(x, u, i, terminal=False):
        if terminal:
            return l(x, np.array([0, 0]), i)
        return l(x, u, i)
def l_x(x, u, i, terminal=False):
        deriv = np.zeros(10)
        deriv[5] = 200 * (x[5] - 10)
        return deriv
def l_u(x, u, i, terminal = False):
        if terminal:
            return np.zeros(2)
        return 2 * u
def l_xx(x, u, i, terminal=False):
        deriv = np.zeros((10, 10))
        deriv[5][5] = 200
        return deriv
def l_ux(x, u, i, terminal=False):
    return np.zeros((2, 10))
def l_uu(x, u, i, terminal=False):
    if terminal:
        return np.zeros((2, 2))
    return 2 * np.eye(2)

def exact_cost_fn():
    cost = ExactCost(l_exact, l_x, l_u, l_xx, l_ux, l_uu, use_multiprocessing = True)
    return cost

class Config:
	xmlpath = 'ilqr/xmls/swimmer.xml'
	action_bounds = None
	cost_fn = exact_cost_fn()
