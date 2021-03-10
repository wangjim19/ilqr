from ilqr.costs.finite_diff import FiniteDiffCost
from ilqr.costs.exact import ExactCost
from ilqr.costs.jax_cost import JaxCost
import numpy as np
import pickle
import haiku as hk
import jax.numpy as jnp
import jax
from jax_models.layers.ensemble_linear import EnsembleLinear

ensemble_size = 5
state_size = 18
action_size = 6

def l(x, u, i):
    action_cost = np.square(u).sum()
    vel_cost = 10 * (x[9] - 4) ** 2
    steady_cost = 200 * (x[10] ** 2)
    return action_cost + vel_cost + steady_cost

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


def cost_func(x, u):
    action_cost = jnp.square(u).sum()
    vel_cost = 10 * (x[9] - 4) ** 2
    steady_cost = 200 * (x[10] ** 2)
    return action_cost + vel_cost + steady_cost
def terminal_cost_func(x):
    vel_cost = 10 * (x[9] - 4) ** 2
    steady_cost = 200 * (x[10] ** 2)
    return vel_cost + steady_cost
def jax_cost_fn():
    cost = JaxCost(cost_func, terminal_cost_func)
    return cost


def model_fn(inputs):
    mlp = hk.Sequential([
        EnsembleLinear(ensemble_size, 200), jax.nn.swish,
        EnsembleLinear(ensemble_size, 200), jax.nn.swish,
        EnsembleLinear(ensemble_size, 200), jax.nn.swish,
        EnsembleLinear(ensemble_size, 200), jax.nn.swish,
        EnsembleLinear(ensemble_size, state_size),
    ])
    return mlp(inputs)

class Config:
    params = pickle.load(open('jax_models/saved-models/halfcheetah-ensemble/params.pkl', 'rb'))
    ensemble_size = ensemble_size
    state_size = state_size
    action_size = action_size
    action_bounds = None
    cost_fn = jax_cost_fn()
    model_fn = model_fn
