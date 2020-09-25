import mujoco_py
from mujoco_py import MjSim, load_model_from_path
import numpy as np

class MujocoDynamics:

    """MuJoCo Dynamics Model."""

    def __init__(self,
                 model_xml_path,
                 frame_skip = 1,
                 constrain = True,
                 bounds = None,
                 x_eps = 1e-6,
                 u_eps = 1e-6):
        """Constructs an AutoDiffDynamics model.

        Args:
            model_xml_path: Path of XML file of model.
            frame_skip: Number of timesteps simulated for each call to step.
            constrain: Whether to constrain actions before passing into model.
                NOTE: Only set constrain to False if model has no forced constraints.
            bounds: [action space, 2] array specifying min, max for each action.
                Defaults to model-defined constraints if constrain=True and bounds unspecified.
                NOTE: make sure user-defined bounds are not wider than model constraints.
            x_eps: Epsilon used for finite differencing wrt state.
            u_eps: Epsilon used for finite differencing wrt action.

        NOTE:
            state space: [qpos[:] qvel[:]] where qpos and qvel are position and velocity
                of joints in order of definition in model.
            action space: In order of actuator definition in model.
        """

        self._model = load_model_from_path(model_xml_path)
        self._frame_skip = frame_skip
        self.sim = MjSim(self._model, nsubsteps = self._frame_skip)
        self._action_size = self.sim.data.ctrl.shape[0]
        self.constrained = constrain
        self.bounds = None
        if constrain:
            if bounds:
                self.bounds = bounds
            else:
                self.bounds = self.sim.model.actuator_ctrlrange
        self.x_eps = x_eps
        self.u_eps = u_eps
        self._simpool = []

    @property
    def state_size(self):
        """State size."""
        return self.sim.model.nq + self.sim.model.nv

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    @property
    def dt(self):
        """Time elapsed per step"""
        return self.model.opt.timestep * self.frame_skip

    def set_state(self, state):
        """Sets state of simulator

        Args:
            state: numpy state vector
        """
        self.sim.data.qpos[:] = state[:self.sim.model.nq]
        self.sim.data.qvel[:] = state[self.sim.model.nq:]

    def get_state(self):
        """Gets state of simulator

        Returns:
            numpy state vector
        """
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel])

    def step(self, action):
        """Simulates one timestep

        Args:
            action: numpy action vector (unconstrained)
        Returns:
            next state vector
        """
        self.sim.data.ctrl[:] = self.constrain(action)
        self.sim.step()
        return self.get_state()

    def f_x(self, state, action):
        """Evaluate f_x at specified state and action by finite differencing

        Args:
            state: numpy state vector
            action: numpy action vector (unconstrained)
        Returns:
            f_x
        """
        self.set_state(state)
        self.step(action)
        center = self.get_state()
        f_x = np.empty((self.state_size, self.state_size))

        self.sim.data.ctrl[:] = self.constrain(action)
        for i in range(self.sim.model.nq):
            self.sim.data.qpos[:] = state[:self.sim.model.nq]
            self.sim.data.qvel[:] = state[self.sim.model.nq:]

            self.sim.data.qpos[i] += x_eps

            self.sim.step()
            newstate = np.concatenate([self.sim.data.qpos, self.sim.data.qvel])
            deriv = (newstate - center) / x_eps
            f_x[:, i] = deriv

        for i in range(self.sim.model.nv):
            self.sim.data.qpos[:] = state[:self.sim.model.nq]
            self.sim.data.qvel[:] = state[self.sim.model.nq:]

            self.sim.data.qvel[i] += x_eps

            self.sim.step()
            newstate = np.concatenate([self.sim.data.qpos, self.sim.data.qvel])
            deriv = (newstate - center) / x_eps
            f_x[:, self.sim.model.nq + i] = deriv

        return f_x

    def f_u(self, state, action):
        """Evaluate f_u at specified state and action by finite differencing

        Args:
            state: numpy state vector
            action: numpy action vector (unconstrained)
        Returns:
            f_u
        """
        self.set_state(state)
        self.step(action)
        center = self.get_state()
        f_u = np.empty((self.state_size, self.action_size))


        for i in range(self.action_size):
            self.sim.data.qpos[:] = state[:self.sim.model.nq]
            self.sim.data.qvel[:] = state[self.sim.model.nq:]

            action[i] += u_eps
            self.sim.data.ctrl[:] = self.constrain(action)

            self.sim.step()
            newstate = np.concatenate([self.sim.data.qpos, self.sim.data.qvel])
            deriv = (newstate - center) / u_eps
            f_u[:, i] = deriv

            action[i] -= u_eps

    def constrain(self, action):
        """Calculates control vector to be passed into model, constraining if necessary

        Args:
            action: numpy action vector (unconstrained)
        Returns:
            control vector for model (constrained if necessary)
        """
        if not self.constrained:
            return action
        diff = (self.bounds[:, 1] - self.bounds[:, 0]) / 2.0
        mean = (self.bounds[:, 1] + self.bounds[:, 0]) / 2.0
        return diff * np.tanh(action) + mean
