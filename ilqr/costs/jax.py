import numpy as onp
import jax
import jax.numpy as jnp
import pdb

import gtimer as gt


def jacobian(f):
    return jax.jacfwd(f)

def hessian(f):
    return jax.jacfwd(jax.jacrev(f))

def to_np(*jax_arrays):
    onp_arrays = [
        onp.array(array)
        for array in jax_arrays
    ]
    if len(onp_arrays) == 1:
        return onp_arrays[0]
    else:
        return onp_arrays


class JaxCost:

    def __init__(self, cost_fn, terminal_cost_fn, batch_cost_fn_np):
        self.cost_fn = cost_fn
        self.terminal_cost_fn = terminal_cost_fn
        self.batch_cost_fn_np = batch_cost_fn_np

        self.vmap_cost_fn = jax.jit(jax.vmap(cost_fn))
        self.vmap_jacobian_fn = jax.jit(jax.vmap(jacobian(cost_fn)))
        self.vmap_hessian_fn = jax.jit(jax.vmap(hessian(cost_fn)))

        self.jacobian_terminal_cost_fn = jax.jit(jacobian(terminal_cost_fn))
        self.hessian_terminal_cost_fn = jax.jit(hessian(terminal_cost_fn))

    def l_np(self, xs, us):
        L = self.batch_cost_fn_np(xs, us)
        return L

    def terminal_l(self, x):
        L = self.terminal_cost_fn(x)
        return L.item()

    def l_derivs(self, xs, us):
        assert len(xs) == len(us) + 1, 'Trajectory length should be one greater than number of control inputs'

        state_dim = xs.shape[1]
        act_dim = us.shape[1]

        joined = jnp.concatenate([xs[:-1], us], axis=-1)
        L = self.vmap_cost_fn(joined)

        ## first-order derivatives
        L_jac = self.vmap_jacobian_fn(joined)
        L_x = L_jac[:,:state_dim]
        L_u = L_jac[:,state_dim:]

        ## second-order derivatives
        L_hessian = self.vmap_hessian_fn(joined)
        L_xx = L_hessian[:, :state_dim, :state_dim]
        L_uu = L_hessian[:, state_dim:, state_dim:]
        L_ux = L_hessian[:, state_dim:, :state_dim]

        return to_np(L, L_x, L_u, L_xx, L_ux, L_uu)

    def terminal_l_derivs(self, x):
        L = self.terminal_cost_fn(x)
        L_jac = self.jacobian_terminal_cost_fn(x)
        L_hessian = self.hessian_terminal_cost_fn(x)
        return to_np(L, L_jac, L_hessian)

