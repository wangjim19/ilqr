import numpy as np
import torch
import os

class LearnedDynamics:

    """Learned Dynamics Model."""

    def __init__(self,
                 model,
                 state_size,
                 action_size,
                 use_vmap = False):
        """

        Args:
            model: torch model (input: concatenated state+action, output: state delta).
        """

        self._model = model
        self._model.eval()
        self._state_size = state_size
        self._action_size = action_size
        self._state = np.zeros(state_size)
        self._use_vmap = use_vmap

    @property
    def state_size(self):
        """State size."""
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    def set_state(self, state):
        """Sets state of simulator

        Args:
            state: numpy state vector
        """
        self._state[:] = state

    def get_state(self):
        """Gets state of simulator

        Returns:
            numpy state vector
        """
        return self._state.copy()

    def step(self, action):
        """Simulates one timestep

        Args:
            action: numpy action vector (unconstrained)
        Returns:
            next state vector
        """

        input = np.concatenate((self._state, action))
        input = torch.from_numpy(input).float()
        delta = self._model(input)
        delta = delta.detach().numpy()
        self._state += delta
        return self.get_state()

    def f_derivs(self, xs, us):
        """Computes dynamics derivatives.

        Args:
            xs: [N or N + 1, state_size] numpy array with state vectors
            us: [N, action_size] numpy array with action vectors
        Returns:
            (f_x, f_u), where f_x and f_u are N-length lists of numpy arrays
                of shape [state_size, state_size] and [state_size, action_size].
        """
        F = torch.zeros((us.shape[0], xs.shape[1], xs.shape[1] + us.shape[1]))
        input = np.hstack((xs[:us.shape[0]], us))
        input = torch.from_numpy(input).float().requires_grad_(True)
        deltas = self._model(input)
        if self._use_vmap:
            jacobian = torch.vmap(lambda y: torch.autograd.grad(y, input), out_dims = 1)
            for i in range(us.shape[0]):
                F += jacobian(deltas[i])
                F[i, :, :xs.shape[1]] += torch.eye(xs.shape[1])
        else:
            for i in range(us.shape[0]):
                for j in range(xs.shape[1]):
                    F[:, j, :] += torch.autograd.grad(deltas[i][j], input, retain_graph=True)[0]
                F[i, :, :xs.shape[1]] += torch.eye(xs.shape[1]) # add ds/ds to d(s' - s)/ds to get d(s')/ds
        F_x = F[:, :, :xs.shape[1]].detach().numpy()
        F_u = F[:, :, xs.shape[1]:].detach().numpy()
        return (F_x, F_u)

    def f_x(self, state, action):
        """Evaluate f_x at specified state and action by finite differencing.
        Does not change main sim (uses simpool for simulating).

        Args:
            state: numpy state vector
            action: numpy action vector (unconstrained)
        Returns:
            f_x
        """

        return f_derivs(np.reshape(state, (1, -1)), np.reshape(action, (1, -1)))[0][0]

    def f_u(self, state, action):
        """Evaluate f_u at specified state and action by finite differencing.
        Does not change main sim (uses simpool for simulating).

        Args:
            state: numpy state vector
            action: numpy action vector (unconstrained)
        Returns:
            f_u
        """

        return f_derivs(np.reshape(state, (1, -1)), np.reshape(action, (1, -1)))[1][0]

    def render(self, dim=512):
        ## rendered img is upside-down by default
        raise NotImplementedError
