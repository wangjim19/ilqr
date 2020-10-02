import mujoco_py
from mujoco_py import MjSim, load_model_from_path
import numpy as np
from scipy.optimize import approx_fprime
import multiprocessing as mp
import os

class MujocoDynamics:

    """MuJoCo Dynamics Model."""

    def __init__(self,
                 model_xml_path,
                 frame_skip = 1,
                 constrain = True,
                 bounds = None,
                 x_eps = 1.5e-8,
                 u_eps = 1.5e-8,
                 use_multiprocessing = False):
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
            use_multiprocessing: Whether to use multiprocessing for computing derivatives

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
        self.multiprocessing = use_multiprocessing
        if self.multiprocessing:
            self._pool = mp.Pool(initializer = MujocoDynamics._worker_init,
                                 initargs = (model_xml_path, frame_skip, constrain, bounds, x_eps, u_eps, False))
        else:
            self._derivsim = MjSim(self._model, nsubsteps = self._frame_skip)

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


    @staticmethod
    def _worker_init(model_xml_path,
                     frame_skip,
                     constrain,
                     bounds,
                     x_eps,
                     u_eps,
                     use_multiprocessing):
        """
        Initializes sims for workers in multiprocessing Pool.
        """
        global mjdynamics
        mjdynamics = MujocoDynamics(model_xml_path, frame_skip, constrain, bounds, x_eps, u_eps, use_multiprocessing)
        print("Finished loading process", os.getpid())

    @staticmethod
    def _worker(state, action):
        return (mjdynamics.f_x(state, action), mjdynamics.f_u(state, action))


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

    def f_derivs(self, xs, us):
        """Computes dynamics derivatives.

        Args:
            xs: [N or N + 1, state_size] numpy array with state vectors
            us: [N, action_size] numpy array with action vectors
        Returns:
            (f_x, f_u), where f_x and f_u are N-length lists of numpy arrays
                of shape [state_size, state_size] and [state_size, action_size].
        """
        if self.multiprocessing:
            results = self._pool.starmap(MujocoDynamics._worker, [(xs[i], us[i]) for i in range(us.shape[0])], chunksize = us.shape[0] // mp.cpu_count())
            return ([result[0] for result in results], [result[1] for result in results])
        else:
            F_x = [self.f_x(xs[i], us[i]) for i in range(us.shape[0])]
            F_u = [self.f_u(xs[i], us[i]) for i in range(us.shape[0])]
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

        self._derivsim.data.qpos[:] = state[:self._derivsim.model.nq]
        self._derivsim.data.qvel[:] = state[self._derivsim.model.nq:]
        self._derivsim.data.ctrl[:] = self.constrain(action)
        self._derivsim.step()
        center = np.concatenate([self._derivsim.data.qpos, self._derivsim.data.qvel])
        f_x = np.empty((self.state_size, self.state_size))


        for i in range(self._derivsim.model.nq):
            self._derivsim.data.qpos[:] = state[:self._derivsim.model.nq]
            self._derivsim.data.qvel[:] = state[self._derivsim.model.nq:]

            self._derivsim.data.qpos[i] += self.x_eps

            self._derivsim.step()
            newstate = np.concatenate([self._derivsim.data.qpos, self._derivsim.data.qvel])
            f_x[:, i] = (newstate - center) / self.x_eps

        for i in range(self._derivsim.model.nv):
            self._derivsim.data.qpos[:] = state[:self._derivsim.model.nq]
            self._derivsim.data.qvel[:] = state[self._derivsim.model.nq:]

            self._derivsim.data.qvel[i] += self.x_eps

            self._derivsim.step()
            newstate = np.concatenate([self._derivsim.data.qpos, self._derivsim.data.qvel])
            f_x[:, self._derivsim.model.nq + i] = (newstate - center) / self.x_eps

        return f_x

    def f_u(self, state, action):
        """Evaluate f_u at specified state and action by finite differencing.
        Does not change main sim (uses simpool for simulating).

        Args:
            state: numpy state vector
            action: numpy action vector (unconstrained)
        Returns:
            f_u
        """

        self._derivsim.data.qpos[:] = state[:self._derivsim.model.nq]
        self._derivsim.data.qvel[:] = state[self._derivsim.model.nq:]
        self._derivsim.data.ctrl[:] = self.constrain(action)
        self._derivsim.step()
        center = np.concatenate([self._derivsim.data.qpos, self._derivsim.data.qvel])
        f_u = np.empty((self.state_size, self.action_size))


        for i in range(self.action_size):
            self._derivsim.data.qpos[:] = state[:self._derivsim.model.nq]
            self._derivsim.data.qvel[:] = state[self._derivsim.model.nq:]

            action[i] += self.u_eps
            self._derivsim.data.ctrl[:] = self.constrain(action)

            self._derivsim.step()
            newstate = np.concatenate([self._derivsim.data.qpos, self._derivsim.data.qvel])
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
