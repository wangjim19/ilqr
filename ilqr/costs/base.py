# Copyright (C) 2018, Anass Al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""Instantaneous Cost Function."""

import six
import abc
import numpy as np
import theano.tensor as T
import multiprocessing as mp
import os

from .autodiff import as_function, hessian_scalar, jacobian_scalar


@six.add_metaclass(abc.ABCMeta)
class Cost():

    """Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

    @abc.abstractmethod
    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        raise NotImplementedError


class AutoDiffCost(Cost):

    """Auto-differentiated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

    def __init__(self, l, l_terminal, x_inputs, u_inputs, i=None, **kwargs):
        """Constructs an AutoDiffCost.

        Args:
            l: Vector Theano tensor expression for instantaneous cost.
                This needs to be a function of x and u and must return a scalar.
            l_terminal: Vector Theano tensor expression for terminal cost.
                This needs to be a function of x only and must retunr a scalar.
            x_inputs: Theano state input variables [state_size].
            u_inputs: Theano action input variables [action_size].
            i: Theano tensor time step variable.
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
        self._i = T.dscalar("i") if i is None else i
        self._x_inputs = x_inputs
        self._u_inputs = u_inputs

        non_t_inputs = np.hstack([x_inputs, u_inputs]).tolist()
        inputs = np.hstack([x_inputs, u_inputs, self._i]).tolist()
        terminal_inputs = np.hstack([x_inputs, self._i]).tolist()

        x_dim = len(x_inputs)
        u_dim = len(u_inputs)

        self._J = jacobian_scalar(l, non_t_inputs)
        self._Q = hessian_scalar(l, non_t_inputs)

        self._l = as_function(l, inputs, name="l", **kwargs)

        self._l_x = as_function(self._J[:x_dim], inputs, name="l_x", **kwargs)
        self._l_u = as_function(self._J[x_dim:], inputs, name="l_u", **kwargs)

        self._l_xx = as_function(self._Q[:x_dim, :x_dim],
                                 inputs,
                                 name="l_xx",
                                 **kwargs)
        self._l_ux = as_function(self._Q[x_dim:, :x_dim],
                                 inputs,
                                 name="l_ux",
                                 **kwargs)
        self._l_uu = as_function(self._Q[x_dim:, x_dim:],
                                 inputs,
                                 name="l_uu",
                                 **kwargs)

        # Terminal cost only depends on x, so we only need to evaluate the x
        # partial derivatives.
        self._J_terminal = jacobian_scalar(l_terminal, x_inputs)
        self._Q_terminal = hessian_scalar(l_terminal, x_inputs)

        self._l_terminal = as_function(l_terminal,
                                       terminal_inputs,
                                       name="l_term",
                                       **kwargs)
        self._l_x_terminal = as_function(self._J_terminal[:x_dim],
                                         terminal_inputs,
                                         name="l_term_x",
                                         **kwargs)
        self._l_xx_terminal = as_function(self._Q_terminal[:x_dim, :x_dim],
                                          terminal_inputs,
                                          name="l_term_xx",
                                          **kwargs)

        super(AutoDiffCost, self).__init__()

    @property
    def x(self):
        """The state variables."""
        return self._x_inputs

    @property
    def u(self):
        """The control variables."""
        return self._u_inputs

    @property
    def i(self):
        """The time step variable."""
        return self._i

    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            z = np.hstack([x, i])
            return np.asscalar(self._l_terminal(*z))

        z = np.hstack([x, u, i])
        return np.asscalar(self._l(*z))

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            z = np.hstack([x, i])
            return np.array(self._l_x_terminal(*z))

        z = np.hstack([x, u, i])
        return np.array(self._l_x(*z))

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros(self._action_size)

        z = np.hstack([x, u, i])
        return np.array(self._l_u(*z))

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        if terminal:
            z = np.hstack([x, i])
            return np.array(self._l_xx_terminal(*z))

        z = np.hstack([x, u, i])
        return np.array(self._l_xx(*z))

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        z = np.hstack([x, u, i])
        return np.array(self._l_ux(*z))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        z = np.hstack([x, u, i])
        return np.array(self._l_uu(*z))


class BatchAutoDiffCost(Cost):

    """Batch Auto-differentiated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.

    NOTE: This offers faster derivatives than AutoDiffCosts if you can
          describe your cost as a symbolic function.
    """

    def __init__(self, f, state_size, action_size, **kwargs):
        """Constructs an BatchAutoDiffCost.

        Args:
            f: Symbolic function with the following signature:
                Args:
                    x: Batch of state variables.
                    u: Batch of action variables.
                    i: Batch of time step variables.
                    terminal: Whether to compute the terminal cost instead.
                Returns:
                    f: Batch of instantaneous costs.
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
        self._fn = f
        self._state_size = state_size
        self._action_size = action_size

        # Prepare inputs.
        self._x = x = T.dvector("x")
        self._u = u = T.dvector("u")
        self._i = i = T.dscalar("i")
        inputs = [self._x, self._u, self._i]
        inputs_term = [self._x, self._i]

        x_rep_x = T.tile(x, (state_size, 1))
        u_rep_x = T.tile(u, (state_size, 1))
        i_rep_x = T.tile(i, (state_size, 1))

        x_rep_u = T.tile(x, (action_size, 1))
        u_rep_u = T.tile(u, (action_size, 1))
        i_rep_u = T.tile(i, (action_size, 1))

        x_rep_1 = T.tile(x, (1, 1))
        u_rep_1 = T.tile(u, (1, 1))
        i_rep_1 = T.tile(i, (1, 1))
        l_tensor = f(x_rep_1, u_rep_1, i_rep_1, terminal=False)[0]
        J_x, J_u = T.grad(l_tensor, [x, u], disconnected_inputs="ignore")

        # Compute the hessians in batches.
        l_tensor_rep_x = f(x_rep_x, u_rep_x, i_rep_x, terminal=False)
        l_tensor_rep_u = f(x_rep_u, u_rep_u, i_rep_u, terminal=False)
        J_x_rep = T.grad(cost=None,
                         wrt=x_rep_x,
                         known_grads={
                             l_tensor_rep_x: T.ones(state_size),
                         },
                         disconnected_inputs="ignore")
        J_u_rep = T.grad(cost=None,
                         wrt=u_rep_u,
                         known_grads={
                             l_tensor_rep_u: T.ones(action_size),
                         },
                         disconnected_inputs="ignore")
        Q_xx = T.grad(cost=None,
                      wrt=x_rep_x,
                      known_grads={
                          J_x_rep: T.eye(state_size),
                      },
                      disconnected_inputs="ignore")
        Q_ux = T.grad(cost=None,
                      wrt=x_rep_u,
                      known_grads={
                          J_u_rep: T.eye(action_size),
                      },
                      disconnected_inputs="ignore")
        Q_uu = T.grad(cost=None,
                      wrt=u_rep_u,
                      known_grads={
                          J_u_rep: T.eye(action_size),
                      },
                      disconnected_inputs="warn")

        # Terminal cost only depends on x, so we only need to evaluate the x
        # partial derivatives.
        l_tensor_term = f(x_rep_1, None, i, terminal=True)[0]
        J_x_term, _ = T.grad(l_tensor_term,
                             inputs_term,
                             disconnected_inputs="ignore")

        l_tensor_rep_term = f(x_rep_x, None, i_rep_x, terminal=True)
        J_x_rep_term = T.grad(cost=None,
                              wrt=x_rep_x,
                              known_grads={
                                  l_tensor_rep_term:
                                      T.ones_like(l_tensor_rep_term),
                              },
                              disconnected_inputs="ignore")
        Q_xx_term = T.grad(cost=None,
                           wrt=x_rep_x,
                           known_grads={
                               J_x_rep_term: T.eye(state_size),
                           },
                           disconnected_inputs="ignore")

        # Compile all functions.
        self._l = as_function(l_tensor, inputs, name="l", **kwargs)
        self._l_x = as_function(J_x, inputs, name="l_x", **kwargs)
        self._l_u = as_function(J_u, inputs, name="l_u", **kwargs)
        self._l_xx = as_function(Q_xx, inputs, name="l_xx", **kwargs)
        self._l_ux = as_function(Q_ux, inputs, name="l_ux", **kwargs)
        self._l_uu = as_function(Q_uu, inputs, name="l_uu", **kwargs)

        self._l_term = as_function(l_tensor_term,
                                   inputs_term,
                                   name="l_term",
                                   **kwargs)
        self._l_x_term = as_function(J_x_term,
                                     inputs_term,
                                     name="l_x_term",
                                     **kwargs)
        self._l_xx_term = as_function(Q_xx_term,
                                      inputs_term,
                                      name="l_xx_term",
                                      **kwargs)

        super(BatchAutoDiffCost, self).__init__()

    @property
    def x(self):
        """The state variables."""
        return self._x

    @property
    def u(self):
        """The control variables."""
        return self._u

    @property
    def i(self):
        """The time step variable."""
        return self._i

    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            return np.asscalar(self._l_term(x, i))

        return np.asscalar(self._l(x, u, i))

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            return np.array(self._l_x_term(x, i))

        return np.array(self._l_x(x, u, i))

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros(self._action_size)

        return np.array(self._l_u(x, u, i))

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        if terminal:
            return np.array(self._l_xx_term(x, i))

        return np.array(self._l_xx(x, u, i))

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        return np.array(self._l_ux(x, u, i))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        return np.array(self._l_uu(x, u, i))
