import pdb

def finite_diff_cost_fn():
    from ilqr.costs.finite_diff import FiniteDiffCost

    cost = FiniteDiffCost(lambda x, u, i: 2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2 + u[0] ** 2,
                          lambda x, i: (2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2),
                          4, 1, use_multiprocessing = True)
    return cost

def exact_cost_fn():
    from ilqr.costs.exact import ExactCost

    cost = ExactCost(lambda x, u, i: 2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2 + u[0] ** 2,
                  lambda x, i: (2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2),
                  4, 1, use_multiprocessing = True)
    return cost

def jax_cost_fn():
    import numpy as onp
    import jax.numpy as jnp
    from ilqr.costs.jax import JaxCost

    def cost_fn(xu):
        coeffs = jnp.array([2, 10, 1, 1, 1])
        xu_squared = xu ** 2
        cost = jnp.dot(coeffs,  xu_squared)
        return cost

    def terminal_cost_fn(x):
        coeffs = jnp.array([2, 10, 1, 1])
        x_squared = x ** 2
        cost = jnp.dot(coeffs,  x_squared)
        return cost

    def batch_cost_fn_np(xs, us):
        """
            faster than jax versions, but not able to get derivatives
        """
        coeffs = onp.array([2, 10, 1, 1])
        xs_squared = onp.square(xs)
        cost = onp.dot(xs_squared, coeffs)
        cost += onp.square(us).sum(axis=-1)
        return cost

    cost = JaxCost(cost_fn, terminal_cost_fn, batch_cost_fn_np)
    return cost
    
class Config:
    xmlpath = 'ilqr/xmls/inverted_pendulum.xml'
    action_bounds = [-1, 1]
    cost_fn = jax_cost_fn()