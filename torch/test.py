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
from ilqr.utils.logging import verbose_iteration_callback


class Parser(Tap):
    config_path: str = 'config.cartpole'
    path_length: int = 100
    horizon: int = 100
    mpc_initial_itrs: int = 500
    mpc_subsequent_itrs: int = 100
    logdir: str = 'logs/cartpole-receding-horizon'
    num_collection_iterations = 10

args = Parser().parse_args()

config = load_config(args.config_path)

dynamics = MujocoDynamics(config.xmlpath, frame_skip=2, use_multiprocessing=True)


state_size = 4
action_size = 1

with open('data-collection/data/cartpole/observations.txt', 'r') as f:
    observations = np.loadtxt(f).reshape(-1, state_size)
with open('data-collection/data/cartpole/actions.txt', 'r') as f:
    actions = np.loadtxt(f).reshape(-1, action_size)
with open('data-collection/data/cartpole/next_observations.txt', 'r') as f:
    next_observations = np.loadtxt(f).reshape(-1, state_size)


for _ in range(20):
    i = np.random.choice(len(observations))
    obs = observations[i]
    ac = actions[i]
    next_obs = next_observations[i]
    print('\n\nDATA:')
    print(obs)
    print(ac)
    print(next_obs)

    print('\nACTUAL:')
    print(obs)
    print(ac)
    dynamics.set_state(obs)
    print(dynamics.step(ac))
