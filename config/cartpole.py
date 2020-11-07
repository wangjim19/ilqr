from ilqr.costs.finite_diff import FiniteDiffCost
from ilqr.costs.exact import ExactCost
from ilqr.costs.jax import JaxCost

def finite_diff_cost_fn():
	cost = FiniteDiffCost(lambda x, u, i: 2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2 + u[0] ** 2,
	                      lambda x, i: (2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2),
	                      4, 1, use_multiprocessing = True)
	return cost

def exact_cost_fn():
    cost = ExactCost(lambda x, u, i: 2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2 + u[0] ** 2,
                  lambda x, i: (2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2),
                  4, 1, use_multiprocessing = True)
    return cost

def jax_cost_fn():
	cost = JaxCost()
	return cost
    
class Config:
	xmlpath = 'ilqr/xmls/inverted_pendulum.xml'
	action_bounds = [-1, 1]
	cost_fn = jax_cost_fn()