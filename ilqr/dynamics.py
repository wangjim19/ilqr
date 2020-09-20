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
import torch
from scipy.optimize import approx_fprime
from .autodiff import (hessian_vector, jacobian_vector)


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
    def f(self, x, u):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            Next state [state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_x(self, x, u):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            df/dx [state_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_u(self, x, u):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            df/du [state_size, action_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_xx(self, x, u):
        """Second partial derivative of dynamics model with respect to x.

        Note:
            This is not necessary to implement if you're planning on skipping
            Hessian evaluation as the iLQR implementation does by default.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_ux(self, x, u):
        """Second partial derivative of dynamics model with respect to u and x.

        Note:
            This is not necessary to implement if you're planning on skipping
            Hessian evaluation as the iLQR implementation does by default.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_uu(self, x, u):
        """Second partial derivative of dynamics model with respect to u.

        Note:
            This is not necessary to implement if you're planning on skipping
            Hessian evaluation as the iLQR implementation does by default.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        raise NotImplementedError


class AutoDiffDynamics(Dynamics):

    """Auto-differentiated Dynamics Model.

    Uses torch tensors."""

    def __init__(self, f, state_size, action_size, hessians=False):
        """Constructs an AutoDiffDynamics model.

        Args:
            f: Python function of the form f(x, u) that takes state and input (torch vectors) and produces next state (torch vector).
                NOTE: all computations in function must be tracked
            state_size: number of state dimensions.
            action_size: number of action dimensions.
            hessians: Evaluate the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
        """
        self._f = f
        self._state_size = state_size
        self._action_size = action_size
        self._has_hessians = hessians

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

    def f(self, x, u):
        """Dynamics model.

        Args:
            x: Current state. torch tensor of shape(state_size)
            u: Current control. torch tensor of shape(action_size)

        Returns:
            Next state. torch tensor of shape(state_size)
        """
        return self._f(x, u)

    def f_x(self, x, u):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state. torch tensor of shape(state_size)
            u: Current control. torch tensor of shape(action_size)

        Returns:
            df/dx. torch tensor of shape(state_size, state_size)
        """
        return jacobian_vector(self._f, (x, u))[0]

    def f_u(self, x, u):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state. torch tensor of shape(state_size)
            u: Current control. torch tensor of shape(action_size)

        Returns:
            df/du. torch tensor of shape(state_size, action_size)
        """
        return jacobian_vector(self._f, (x, u))[1]

    def f_xx(self, x, u):
        """Second partial derivative of dynamics model with respect to x.

        Args:
            x: Current state. torch tensor of shape(state_size)
            u: Current control. torch tensor of shape(action_size)

        Returns:
            d^2f/dx^2. torch tensor of shape (state_size, state_size, state_size)
        """
        if not self._has_hessians:
            raise NotImplementedError

        return hessian_vector(self._f, (x, u))[0][0]

    def f_ux(self, x, u):
        """Second partial derivative of dynamics model with respect to u and x.

        Args:
            x: Current state. torch tensor of shape(state_size)
            u: Current control. torch tensor of shape(action_size)

        Returns:
            d^2f/dudx. torch tensor of shape(state_size, action_size, state_size)
        """
        if not self._has_hessians:
            raise NotImplementedError

        return hessian_vector(self._f, (x, u))[1][0]

    def f_uu(self, x, u):
        """Second partial derivative of dynamics model with respect to u.

        Args:
            x: Current state. torch tensor of shape(state_size)
            u: Current control. torch tensor of shape(action_size)

        Returns:
            d^2f/du^2. torch tensor of shape(state_size, action_size, action_size)
        """
        if not self._has_hessians:
            raise NotImplementedError

        return hessian_vector(self._f, (x, u))[1][1]


class FiniteDiffDynamics(Dynamics):

    """Finite difference approximated Dynamics Model.

    Internally uses scipy/numpy to compute finite differences, not pytorch."""

    def __init__(self, f, state_size, action_size, x_eps=None, u_eps=None):
        """Constructs an FiniteDiffDynamics model.

        Args:
            f: Function to approximate. Signature: (x, u) -> x. where x, u, x are torch tensors
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
        self._f = lambda x, u : f(x, u).numpy()
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

    def f(self, x, u):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            Next state [state_size].
        """
        return torch.from_numpy(self._f(x, u))

    def f_x(self, x, u):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            df/dx [state_size, state_size].
        """
        J = np.vstack([
            approx_fprime(x, lambda x: self._f(x, u)[m], self._x_eps)
            for m in range(self._state_size)
        ])
        return torch.from_numpy(J)

    def f_u(self, x, u):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            df/du [state_size, action_size].
        """
        J = np.vstack([
            approx_fprime(u, lambda u: self._f(x, u)[m], self._u_eps)
            for m in range(self._state_size)
        ])
        return torch.from_numpy(J)

    def f_xx(self, x, u):
        """Second partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        eps = self._x_eps_hess

        # yapf: disable
        Q = np.array([
            [
                approx_fprime(x, lambda x: self.f_x(x, u).numpy()[m, n], eps)
                for n in range(self._state_size)
            ]
            for m in range(self._state_size)
        ])
        # yapf: enable
        return torch.from_numpy(Q)

    def f_ux(self, x, u):
        """Second partial derivative of dynamics model with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        eps = self._x_eps_hess

        # yapf: disable
        Q = np.array([
            [
                approx_fprime(x, lambda x: self.f_u(x, u).numpy()[m, n], eps)
                for n in range(self._action_size)
            ]
            for m in range(self._state_size)
        ])
        # yapf: enable
        return torch.from_numpy(Q)

    def f_uu(self, x, u):
        """Second partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        eps = self._u_eps_hess

        # yapf: disable
        Q = np.array([
            [
                approx_fprime(u, lambda u: self.f_u(x, u).numpy()[m, n], eps)
                for n in range(self._action_size)
            ]
            for m in range(self._state_size)
        ])
        # yapf: enable
        return torch.from_numpy(Q)


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

    This is implemented with Pytorch, so as to be auto-differentiable.

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
