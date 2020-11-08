import time
import os
import numpy as np
import gtimer as gt
from tap import Tap
import pdb

from ilqr.dynamics.mujoco_dynamics import MujocoDynamics
from ilqr.mujoco_controller import (
    iLQR,
    RecedingHorizonController,
)
from ilqr.utils.config import load_config
from ilqr.utils.rollouts import monitored_rollout
from ilqr.utils.visualization import save_video
from ilqr.utils.logging import verbose_iteration_callback


class Parser(Tap):
    config_path: str = 'config.cartpole'
    path_length: int = 100
    horizon: int = 100
    mpc_initial_itrs: int = 500
    mpc_subsequent_itrs: int = 100
    logdir: str = 'logs/cartpole-receding-horizon'

args = Parser().parse_args()

config = load_config(args.config_path)

dynamics = MujocoDynamics(config.xmlpath, frame_skip=2, use_multiprocessing=True)
print(dynamics.dt)

x0 = np.array([0.0, np.random.uniform(-np.pi, np.pi), 0.0, 0.0])

us_init = np.random.uniform(*config.action_bounds, (args.horizon, dynamics.action_size))
ilqr = iLQR(dynamics, config.cost_fn, args.horizon)
mpc = RecedingHorizonController(x0, ilqr)
gt.stamp('initialization')

## run ilqr
mpc_trajectory, controls = mpc.control(us_init,
    args.path_length,
    initial_n_iterations=args.mpc_initial_itrs,
    subsequent_n_iterations=args.mpc_subsequent_itrs,
    on_iteration=verbose_iteration_callback)
gt.stamp('control')

## save rollout video to disk
video_trajectory, video_frames = monitored_rollout(dynamics, x0, controls)
save_video(os.path.join(args.logdir, 'rollout.mp4'), video_frames)
gt.stamp('video logging')

## print time information
print(gt.report())

## trajectory from rolling out resulting control sequence
## should match the trajectory given by the mpc solver
assert (mpc_trajectory == video_trajectory).all()


