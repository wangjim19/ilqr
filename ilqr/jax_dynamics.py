import numpy as np
import jax
from jax import jit, vmap, jacfwd
import jax.numpy as jnp
import haiku as hk
import os
import time

class JaxEnsembleDynamics:

    """Learned Dynamics Model."""

    def __init__(self,
                 model_fn,
                 model_parameters,
                 ensemble_size,
                 state_size,
                 action_size,
                 aggregation_mode = 'average'):
        """

        Args:
            model_fn: model forward function,
                inputs: (ensemble_size, batch_size, state_size + action_size) ->
                deltas: (ensemble_size, batch_size, state_size + action_size)
            model_parameters: haiku parameters object
            ensemble_size: number of bootstraps in ensemble
            aggregation_mode: how to aggregate bootstrap predictions
                'average': takes the mean prediction per timestep
                'random': samples a random bootstrap per timestep
        """

        self._model = jit(hk.without_apply_rng(hk.transform(model_fn)).apply)
        self._model(model_parameters, np.zeros((ensemble_size, state_size + action_size)))

        self._params = model_parameters
        self.ensemble_size = ensemble_size
        self._state_size = state_size
        self._action_size = action_size
        self._state = np.zeros(state_size)
        self.aggregation_mode = aggregation_mode

        #setup to compute jacobians
        f = lambda inputs: jnp.add(self._model(self._params, inputs), inputs[:, :self._state_size]) #add inputs because model predicts deltas
        jac_fn = jacfwd(f)
        jac_fn_shaped = lambda inputs: np.sum(jac_fn(inputs), axis=2)
        self._vmapped_jac_fn = jit(vmap(jac_fn_shaped, in_axes = 1)) #output shape: (N, ensemble_size, state_size, state_size + action_size)
        J = self._vmapped_jac_fn(np.zeros((self.ensemble_size, 10, self._state_size + self._action_size))) #jit compile

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
        self._state = state.copy()

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

        inputs = np.tile(np.hstack((self._state, action)), (self.ensemble_size, 1))
        deltas = self._model(self._params, inputs)
        if self.aggregation_mode == 'average':
            self._state += np.mean(deltas, axis=0)
        elif self.aggregation_mode == 'random':
            self._state += deltas[np.random.randint(self.ensemble_size)]
        return self.get_state()

    def f(self, x_batch, u_batch):
        """Predicts next states given states and actions.

        Args:
            xs: [batch_size, state_size]
            us: [batch_size, action_size]
        Returns:
            [batch_size, state_size]
        """
        inputs = np.tile(np.hstack((x_batch, u_batch)), (self.ensemble_size, 1, 1))
        deltas = self._model(self._params, inputs)
        if self.aggregation_mode == 'average':
            predictions = x_batch + np.mean(deltas, axis=0)
        elif self.aggregation_mode == 'random':
            predictions = x_batch + deltas[np.random.randint(self.ensemble_size)]
        return predictions

    def f_derivs(self, xs, us):
        """Computes dynamics derivatives.

        Args:
            xs: [N or N + 1, state_size] numpy array with state vectors
            us: [N, action_size] numpy array with action vectors
        Returns:
            (f_x, f_u), where f_x and f_u are N-length lists of numpy arrays
                of shape [state_size, state_size] and [state_size, action_size].
        """
        inputs = np.tile(np.hstack((xs[:us.shape[0]], us)), (self.ensemble_size, 1, 1))
        J = self._vmapped_jac_fn(inputs) #shape (N, ensemble_size, state_size, state_size + action_size)
        if self.aggregation_mode == 'average':
            J = np.mean(J, axis=1)
        elif self.aggregation_mode == 'random':
            J = J[np.arange(us.shape[0]), np.random.randint(self.ensemble_size, size=us.shape[0])]
        F_x = J[:, :, :self.state_size]
        F_u = J[:, :, self.state_size:]
        return (F_x, F_u)

    def f_x(self, state, action):
        """Evaluate f_x at specified state and action.

        Args:
            state: numpy state vector
            action: numpy action vector (unconstrained)
        Returns:
            f_x
        """

        return f_derivs(np.reshape(state, (1, -1)), np.reshape(action, (1, -1)))[0][0]

    def f_u(self, state, action):
        """Evaluate f_u at specified state and action.

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
