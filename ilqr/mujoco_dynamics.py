import mujoco_py
from mujoco_py import MjSim, load_model_from_path

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
            frame_skip: Number of substeps for each call to step.
            constrain: Whether to constrain actions before passing into model.
                NOTE: Only set constrain to False if model has no forced constraints.
            bounds: [action space, 2] numpy array specifying min, max for each action.
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
        self._state_size = self.sim.data.qpos.shape[0] + self.sim.data.qvel.shape[0]
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
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    @property
    def dt(self):
        """Time elapsed per step"""
        return self.model.opt.timestep * self.frame_skip
