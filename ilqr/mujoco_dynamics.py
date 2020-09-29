import mujoco_py
from mujoco_py import MjSim, load_model_from_path
import numpy as np
from scipy.optimize import approx_fprime
import time

class MujocoDynamics:

    """MuJoCo Dynamics Model."""

    def __init__(self,
                 model_xml_path,
                 frame_skip = 1,
                 constrain = True,
                 bounds = None,
                 x_eps = 1.5e-8,
                 u_eps = 1.5e-8):
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
            if bounds is not None:
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
        return self._model.opt.timestep * self._frame_skip

    def set_state(self, state):
        """Sets state of simulator

        Args:
            state: numpy state vector
        """
        self.sim.data.qpos[:] = state[:self.sim.model.nq]
        self.sim.data.qvel[:] = state[self.sim.model.nq:]
        self.sim.forward()

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

        while len(self._simpool) < 1:
            self._simpool.append(MjSim(self._model, nsubsteps = self._frame_skip))

        self._simpool[0].data.qpos[:] = state[:self._simpool[0].model.nq]
        self._simpool[0].data.qvel[:] = state[self._simpool[0].model.nq:]
        self._simpool[0].data.ctrl[:] = self.constrain(action)
        self._simpool[0].step()
        center = np.concatenate([self._simpool[0].data.qpos, self._simpool[0].data.qvel])
        f_x = np.empty((self.state_size, self.state_size))


        for i in range(self._simpool[0].model.nq):
            self._simpool[0].data.qpos[:] = state[:self._simpool[0].model.nq]
            self._simpool[0].data.qvel[:] = state[self._simpool[0].model.nq:]

            self._simpool[0].data.qpos[i] += self.x_eps

            self._simpool[0].step()
            newstate = np.concatenate([self._simpool[0].data.qpos, self._simpool[0].data.qvel])
            f_x[:, i] = (newstate - center) / self.x_eps

        for i in range(self._simpool[0].model.nv):
            self._simpool[0].data.qpos[:] = state[:self._simpool[0].model.nq]
            self._simpool[0].data.qvel[:] = state[self._simpool[0].model.nq:]

            self._simpool[0].data.qvel[i] += self.x_eps

            self._simpool[0].step()
            newstate = np.concatenate([self._simpool[0].data.qpos, self._simpool[0].data.qvel])
            f_x[:, self._simpool[0].model.nq + i] = (newstate - center) / self.x_eps

        return f_x

    def f_u(self, state, action):
        """Evaluate f_u at specified state and action by finite differencing

        Args:
            state: numpy state vector
            action: numpy action vector (unconstrained)
        Returns:
            f_u
        """

        while len(self._simpool) < 1:
            self._simpool.append(MjSim(self._model, nsubsteps = self._frame_skip))

        self._simpool[0].data.qpos[:] = state[:self._simpool[0].model.nq]
        self._simpool[0].data.qvel[:] = state[self._simpool[0].model.nq:]
        self._simpool[0].data.ctrl[:] = self.constrain(action)
        self._simpool[0].step()
        center = np.concatenate([self._simpool[0].data.qpos, self._simpool[0].data.qvel])
        f_u = np.empty((self.state_size, self.action_size))


        for i in range(self.action_size):
            self._simpool[0].data.qpos[:] = state[:self._simpool[0].model.nq]
            self._simpool[0].data.qvel[:] = state[self._simpool[0].model.nq:]

            action[i] += self.u_eps
            self._simpool[0].data.ctrl[:] = self.constrain(action)

            self._simpool[0].step()
            newstate = np.concatenate([self._simpool[0].data.qpos, self._simpool[0].data.qvel])
            f_u[:, i] = (newstate - center) / self.u_eps

            action[i] -= self.u_eps

        return f_u

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
