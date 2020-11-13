from ilqr.costs.finite_diff import FiniteDiffCost
from ilqr.costs.exact import ExactCost
import numpy as np

def l(x, u, i):
    action_cost = np.square(u).sum()
    vel_cost = 10 * (x[9] - 4) ** 2
    steady_cost = 200 * (x[10] ** 2)
    return action_cost + vel_cost + steady_cost

cost2 = FiniteDiffCost(l, lambda x, i: l(x, np.array([0, 0, 0, 0, 0, 0]), i), 18, 6, use_multiprocessing = True)

def l_exact(x, u, i, terminal=False):
        if terminal:
            return l(x, np.array([0, 0, 0, 0, 0, 0]), i)
        return l(x, u, i)
def l_x(x, u, i, terminal=False):
        deriv = np.zeros(18)
        deriv[9] = 20 * (x[9] - 4)
        deriv[10] = 400 * x[10]
        return deriv
def l_u(x, u, i, terminal = False):
        if terminal:
            return np.zeros(6)
        return 2 * u
def l_xx(x, u, i, terminal=False):
        deriv = np.zeros((18, 18))
        deriv[9][9] = 20
        deriv[10][10] = 400
        return deriv
def l_ux(x, u, i, terminal=False):
    return np.zeros((6, 18))
def l_uu(x, u, i, terminal=False):
    if terminal:
        return np.zeros((6, 6))
    return 2 * np.eye(6)

def exact_cost_fn():
    cost = ExactCost(l_exact, l_x, l_u, l_xx, l_ux, l_uu, use_multiprocessing = True)
    return cost

class Config:
	xmlpath = 'ilqr/xmls/half_cheetah.xml'
	action_bounds = None
	cost_fn = exact_cost_fn()
