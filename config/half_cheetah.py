import pdb

STATE_DIM = 18
ACT_DIM = 6

def jax_cost_fn():
    import numpy as onp
    import jax.numpy as jnp
    from ilqr.costs.jax import JaxCost

    def _state_cost(x):
        vel_cost = 10 * (x[9] - 4) ** 2
        steady_cost = 200 * (x[10] ** 2)

        total_cost = vel_cost + steady_cost
        return total_cost

    def cost_fn(xu):
        x = xu[:STATE_DIM]
        u = xu[STATE_DIM:]

        action_cost = jnp.square(u).sum()
        state_cost = _state_cost(x)

        total_cost = action_cost + state_cost
        return total_cost

    def terminal_cost_fn(x):
        return _state_cost(x)

    def batch_cost_fn(xs, us):
        action_cost = onp.square(us).sum(axis=-1)
        vel_cost = 10 * onp.square(xs[:,9] - 4)
        steady_cost = 200 * onp.square(xs[:,10])

        total_cost = action_cost + vel_cost + steady_cost
        return total_cost

    cost = JaxCost(cost_fn, terminal_cost_fn, batch_cost_fn)
    return cost

class Config:
    xmlpath = 'ilqr/xmls/half_cheetah.xml'
    action_bounds = [-1, 1]
    cost_fn = jax_cost_fn()