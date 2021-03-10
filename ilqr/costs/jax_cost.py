import os
import numpy as np
import jax.numpy as jnp
from jax import jit, grad, vmap, jacfwd
import jax

from .base import Cost

class JaxCost(Cost):

    def __init__(self,
                 cost_func,
                 terminal_cost_func):
        """
        Args:
            l: cost function, inputs = x, u
            l_terminal: terminal cost function, inputs = x
        """
        self.cost_func = jit(cost_func)
        self.vmapped_cost_func = jit(vmap(self.cost_func))
        self.terminal_cost_func = jit(terminal_cost_func)

        self.l_x_func = jit(grad(self.cost_func, argnums=0))
        self.vmapped_l_x_func = jit(vmap(self.l_x_func))
        self.terminal_l_x_func = jit(grad(self.terminal_cost_func))

        self.l_u_func = jit(grad(self.cost_func, argnums=1))
        self.vmapped_l_u_func = jit(vmap(self.l_u_func))

        self.l_xx_func = jit(jacfwd(self.l_x_func, argnums=0))
        self.vmapped_l_xx_func = jit(vmap(self.l_xx_func))
        self.terminal_l_xx_func = jit(jacfwd(self.terminal_l_x_func))

        self.l_ux_func = jit(jacfwd(self.l_u_func, argnums=0))
        self.vmapped_l_ux_func = jit(vmap(self.l_ux_func))

        self.l_uu_func = jit(jacfwd(self.l_u_func, argnums=1))
        self.vmapped_l_uu_func = jit(vmap(self.l_uu_func))

        super(JaxCost, self).__init__()

    def l_derivs(self, xs, us):
        xs = xs[:us.shape[0]]
        L = self.vmapped_cost_func(xs, us)
        L_x = self.vmapped_l_x_func(xs, us)
        L_u = self.vmapped_l_u_func(xs, us)
        L_xx = self.vmapped_l_xx_func(xs, us)
        L_ux = self.vmapped_l_ux_func(xs, us)
        L_uu = self.vmapped_l_uu_func(xs, us)

        return (list(L), list(L_x), list(L_u), list(L_xx), list(L_ux), list(L_uu))

    def l(self, x, u, i, terminal=False):
        """

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            return self.terminal_cost_func(x)
        return self.cost_func(x, u)


    def l_x(self, x, u, i, terminal=False):
        """

        Args:
            x: state [state_size]
            u: control [action_size]
            i: timestep
            terminal: True for terminal cost

        Returns:
            l_x at x, u [state_size]
        """
        if terminal:
            return self.terminal_l_x_func(x)
        return self.l_x_func(x, u)

    def l_u(self, x, u, i, terminal=False):
        """

        Args:
            x: state [state_size]
            u: control [action_size]
            i: timestep
            terminal: True for terminal cost

        Returns:
            l_u at x, u [action_size]
        """
        if terminal:
            return np.zeros_like(u)
        return self.l_u_func(x, u)

    def l_xx(self, x, u, i, terminal=False):
        """ Must be implemented by user.

        Args:
            x: state [state_size]
            u: control [action_size]
            i: timestep
            terminal: True for terminal cost

        Returns:
            l_xx at x, u [state_size, state_size]
        """
        if terminal:
            return self.terminal_l_xx_func(x)
        return self.l_xx_func(x, u)

    def l_ux(self, x, u, i, terminal=False):
        """ Must be implemented by user.

        Args:
            x: state [state_size]
            u: control [action_size]
            i: timestep
            terminal: True for terminal cost

        Returns:
            l_u at x, u [action_size, state_size]
        """
        if terminal:
            return np.zeros((u.shape[0], x.shape[0]))
        return self.l_ux_func(x, u)

    def l_uu(self, x, u, i, terminal=False):
        """ Must be implemented by user.

        Args:
            x: state [state_size]
            u: control [action_size]
            i: timestep
            terminal: True for terminal cost

        Returns:
            l_u at x, u [action_size, action_size]
        """
        if terminal:
            return np.zeros((u.shape[0], u.shape[0]))
        return self.l_uu_func(x, u)
