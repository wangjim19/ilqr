from ilqr.cost import FiniteDiffCost

def finite_diff_cost_fn():
	cost = FiniteDiffCost(lambda x, u, i: 2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2 + u[0] ** 2,
	                      lambda x, i: (2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2),
	                      4, 1, use_multiprocessing = True)
	return cost

class Config:
	xmlpath = 'ilqr/xmls/inverted_pendulum.xml'
	action_bounds = [-1, 1]
	cost_fn = finite_diff_cost_fn()