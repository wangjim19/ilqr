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
"""Dynamics model."""

import six
import abc
import numpy as np
import torch
from scipy.optimize import approx_fprime
from .autodiff import (hessian_vector,
                       jacobian_vector, jacobian_vector_test)


@six.add_metaclass(abc.ABCMeta)
class Dynamics():

    """Dynamics Model."""

    @property
    @abc.abstractmethod
    def state_size(self):
        """State size."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_size(self):
        """Action size."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        raise NotImplementedError

    @abc.abstractmethod
    def f(self, x, u, i):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.

        Note:
            This is not necessary to implement if you're planning on skipping
            Hessian evaluation as the iLQR implementation does by default.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u and x.

        Note:
            This is not necessary to implement if you're planning on skipping
            Hessian evaluation as the iLQR implementation does by default.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u.

        Note:
            This is not necessary to implement if you're planning on skipping
            Hessian evaluation as the iLQR implementation does by default.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        raise NotImplementedError

    def f_derivs(self, x, u, i):
        """
        Returns (f_x, f_u) if not has hessians, else (f_x, f_u, f_xx, f_ux, f_uu)
        """

        raise NotImplementedError


class AutoDiffDynamics(Dynamics):

    """Auto-differentiated Dynamics Model."""

    def __init__(self, f, state_size, action_size, i=None, hessians=False):
        """Constructs an AutoDiffDynamics model.

        Args:
            f: Python vector function (x, u, i) -> (x) where x and u are pytorch tensors.
                NOTE: all computations must be tracked
            state_size: number of state dimensions
            action_size: number of action dimensions
            i: torch tensor time step variable.
            hessians: Evaluate the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
        """
        self._state_size = state_size
        self._action_size = action_size
        self._has_hessians = hessians
        self._i = i
        self._f = f

        super(AutoDiffDynamics, self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    @property
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return self._has_hessians

    @property
    def i(self):
        """The time step variable."""
        return self._i

    def f(self, x, u, i):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        x = torch.from_numpy(x)
        u = torch.from_numpy(u)
        return self._f(x, u, i).numpy()

    def f_derivs(self, x, u, i):
        """
        Returns (f_x, f_u) if not has hessians, else (f_x, f_u, f_xx, f_ux, f_uu)
        """
        x = torch.from_numpy(x)
        u = torch.from_numpy(u)
        first_grads =  jacobian_vector_test(lambda x, u: self._f(x, u, i), (x, u)).detach().numpy()
        if self._has_hessians:
            raise NotImplementedError
        else:
            return (first_grads[:, :self.state_size], first_grads[:, self.state_size:])

    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """
        x = torch.from_numpy(x)
        u = torch.from_numpy(u)
        return jacobian_vector_test(lambda x, u: self._f(x, u, i), (x, u))[:, :self.state_size].detach().numpy()

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        x = torch.from_numpy(x)
        u = torch.from_numpy(u)
        return jacobian_vector_test(lambda x, u: self._f(x, u, i), (x, u))[:, self.state_size:].detach().numpy()

    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        x = torch.from_numpy(x)
        u = torch.from_numpy(u)
        return hessian_vector(lambda x, u: self._f(x, u, i), (x, u), self._state_size)[0][0].numpy()

    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        x = torch.from_numpy(x)
        u = torch.from_numpy(u)
        return hessian_vector(lambda x, u: self._f(x, u, i), (x, u), self._state_size)[1][0].numpy()

    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        x = torch.from_numpy(x)
        u = torch.from_numpy(u)
        return hessian_vector(lambda x, u: self._f(x, u, i), (x, u), self._state_size)[1][1].numpy()


class FiniteDiffDynamics(Dynamics):

    """Finite difference approximated Dynamics Model."""

    def __init__(self, f, state_size, action_size, x_eps=None, u_eps=None):
        """Constructs an FiniteDiffDynamics model.

        Args:
            f: Function to approximate. Signature: (x, u, i) -> x.
            state_size: State size.
            action_size: Action size.
            x_eps: Increment to the state to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
            u_eps: Increment to the action to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).

        Note:
            The square root of the provided epsilons are used when computing
            the Hessians instead.
        """
        self._f = f
        self._state_size = state_size
        self._action_size = action_size

        self._x_eps = x_eps if x_eps else np.sqrt(np.finfo(float).eps)
        self._u_eps = u_eps if x_eps else np.sqrt(np.finfo(float).eps)

        self._x_eps_hess = np.sqrt(self._x_eps)
        self._u_eps_hess = np.sqrt(self._u_eps)

        super(FiniteDiffDynamics, self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    @property
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return True

    def f_derivs(self, x, u, i):
        """
        Returns (f_x, f_u) if not has hessians, else (f_x, f_u, f_xx, f_ux, f_uu)
        """

        if self._has_hessians:
            return (self.f_x(x,u,i), self.f_u(x,u,i), self.f_xx(x,u,i), self.f_ux(x,u,i), self.f_uu(x,u,i))
        else:
            return (self.f_x(x,u,i), self.f_u(x,u,i))

    def f(self, x, u, i):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        return self._f(x, u, i)

    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """
        J = np.vstack([
            approx_fprime(x, lambda x: self._f(x, u, i)[m], self._x_eps)
            for m in range(self._state_size)
        ])
        return J

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        J = np.vstack([
            approx_fprime(u, lambda u: self._f(x, u, i)[m], self._u_eps)
            for m in range(self._state_size)
        ])
        return J

    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        eps = self._x_eps_hess

        # yapf: disable
        Q = np.array([
            [
                approx_fprime(x, lambda x: self.f_x(x, u, i)[m, n], eps)
                for n in range(self._state_size)
            ]
            for m in range(self._state_size)
        ])
        # yapf: enable
        return Q

    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        eps = self._x_eps_hess

        # yapf: disable
        Q = np.array([
            [
                approx_fprime(x, lambda x: self.f_u(x, u, i)[m, n], eps)
                for n in range(self._action_size)
            ]
            for m in range(self._state_size)
        ])
        # yapf: enable
        return Q

    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        eps = self._u_eps_hess

        # yapf: disable
        Q = np.array([
            [
                approx_fprime(u, lambda u: self.f_u(x, u, i)[m, n], eps)
                for n in range(self._action_size)
            ]
            for m in range(self._state_size)
        ])
        # yapf: enable
        return Q


def constrain(u, min_bounds, max_bounds):
    """Constrains a control vector between given bounds through a squashing
    function.

    Args:
        u: Control vector [action_size].
        min_bounds: Minimum control bounds [action_size].
        max_bounds: Maximum control bounds [action_size].

    Returns:
        Constrained control vector [action_size].
    """
    diff = (max_bounds - min_bounds) / 2.0
    mean = (max_bounds + min_bounds) / 2.0
    return diff * np.tanh(u) + mean


def tensor_constrain(u, min_bounds, max_bounds):
    """Constrains a control vector tensor variable between given bounds through
    a squashing function.

    This is implemented with torch, so as to be auto-differentiable.

    Args:
        u: Control vector tensor variable [action_size].
        min_bounds: Minimum control bounds [action_size].
        max_bounds: Maximum control bounds [action_size].

    Returns:
        Constrained control vector tensor variable [action_size].
    """
    diff = (max_bounds - min_bounds) / 2.0
    mean = (max_bounds + min_bounds) / 2.0
    return diff * torch.tanh(u) + mean
