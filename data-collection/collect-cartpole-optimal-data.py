import time
import os
import numpy as np
import gtimer as gt
from tap import Tap
import pdb

from ilqr.mujoco_dynamics import MujocoDynamics
from ilqr.mujoco_controller import (
    iLQR,
    RecedingHorizonController,
)
from ilqr.utils.config import load_config
from ilqr.utils.rollouts import monitored_rollout
from ilqr.utils.logging import save_optimal_only


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

x0 = np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2)])


ilqr = iLQR(dynamics, config.cost_fn, args.horizon, multiprocessing = True)
mpc = RecedingHorizonController(x0, ilqr)
gt.stamp('initialization')

rollout_history = {}
## run ilqr
while 'observations' not in rollout_history or rollout_history['observations'].shape[0] < 800000:
    us_init = np.random.uniform(*config.action_bounds, (args.horizon, dynamics.action_size))
    mpc_trajectory, controls = mpc.control(us_init,
        args.path_length,
        initial_n_iterations=args.mpc_initial_itrs,
        subsequent_n_iterations=args.mpc_subsequent_itrs,
        on_iteration=save_optimal_only,
        rollout_history = rollout_history)
    if 'observations' in rollout_history and rollout_history['observations'].shape[0] % 1000 == 0:
        print('collected', rollout_history['observations'].shape[0] % 1000, 'samples')
gt.stamp('control')

## save rollout history to file
print("\n\nobservations shape:", rollout_history["observations"].shape)
print("actions shape:", rollout_history["actions"].shape)
print("next_observations shape:", rollout_history["next_observations"].shape)
print('\n\n')
with open('data-collection/data/cartpole-optimal/observations.txt', 'a') as f:
    np.savetxt(f, rollout_history["observations"])
with open('data-collection/data/cartpole-optimal/actions.txt', 'a') as f:
    np.savetxt(f, rollout_history["actions"])
with open('data-collection/data/cartpole-optimal/next_observations.txt', 'a') as f:
    np.savetxt(f, rollout_history["next_observations"])

with open('data-collection/data/cartpole-optimal/observations.txt', 'r') as f:
    observations = np.loadtxt(f)
with open('data-collection/data/cartpole-optimal/actions.txt', 'r') as f:
    actions = np.loadtxt(f)
with open('data-collection/data/cartpole-optimal/next_observations.txt', 'r') as f:
    next_observations = np.loadtxt(f)
print("\n\nobservations shape:", observations.shape)
print("actions shape:", actions.shape)
print("next_observations shape:", next_observations.shape)
print('\n\n')

## print time information
print(gt.report())
