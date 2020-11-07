import numpy as onp
import jax
import jax.numpy as jnp
import pdb

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

def jacobian(f):
    return jax.jacfwd(f)

def hessian(f):
    return jax.jacfwd(jax.jacrev(f))

def jac_hess(f):
    jac = jax.jacrev(f)
    hess = jax.jacfwd(jac)
    return jac, hess

def to_np(*jax_arrays):
    onp_arrays = [
        onp.array(array)
        for array in jax_arrays
    ]
    if len(onp_arrays) == 1:
        return onp_arrays[0]
    else:
        return onp_arrays

vmap_cost_fn = jax.vmap(cost_fn)
vmap_jacobian_fn = jax.vmap(jacobian(cost_fn))
vmap_hessian_fn = jax.vmap(hessian(cost_fn))

jacobian_terminal_cost_fn = jacobian(terminal_cost_fn)
hessian_terminal_cost_fn = hessian(terminal_cost_fn)

class JaxCost:


    def l(self, x, u):
        joined = jnp.concatenate([x, u], axis=-1)
        L = vmap_cost_fn(joined)
        return to_np(L)

    def terminal_l(self, x):
        L = terminal_cost_fn(x)
        return L.item()

    def l_derivs(self, xs, us):
        assert len(xs) == len(us) + 1, 'Trajectory length should be one greater than number of control inputs'

        state_dim = xs.shape[1]
        act_dim = us.shape[1]

        joined = jnp.concatenate([xs[:-1], us], axis=-1)
        L = vmap_cost_fn(joined)

        ## first-order derivatives
        L_jac = vmap_jacobian_fn(joined)
        L_x = L_jac[:,:state_dim]
        L_u = L_jac[:,state_dim:]

        ## second-order derivatives
        L_hessian = vmap_hessian_fn(joined)
        L_xx = L_hessian[:, :state_dim, :state_dim]
        L_uu = L_hessian[:, state_dim:, state_dim:]
        L_ux = L_hessian[:, state_dim:, :state_dim]

        out = jac_hess(cost_fn)(joined)
        pdb.set_trace()

        return to_np(L, L_x, L_u, L_xx, L_ux, L_uu)

    def terminal_l_derivs(self, x):
        L = terminal_cost_fn(x)
        L_jac = jacobian_terminal_cost_fn(x)
        L_hessian = hessian_terminal_cost_fn(x)
        return to_np(L, L_jac, L_hessian)

