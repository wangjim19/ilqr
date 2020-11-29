import os
import numpy as np
from tap import Tap
import torch

from ilqr.mujoco_dynamics import MujocoDynamics
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


x0 = np.array([np.random.uniform(-1.0, 1.0), np.random.uniform(-np.pi, np.pi), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)])


us_init = np.random.uniform(*config.action_bounds, (args.horizon, dynamics.action_size))
ilqr = iLQR(dynamics, config.cost_fn, args.horizon, multiprocessing = True)
mpc = RecedingHorizonController(x0, ilqr)

## run ilqr
mpc_trajectory, controls = mpc.control(us_init,
    args.path_length,
    initial_n_iterations=args.mpc_initial_itrs,
    subsequent_n_iterations=args.mpc_subsequent_itrs,
    on_iteration=verbose_iteration_callback)

## save rollout video to disk
video_trajectory, video_frames = monitored_rollout(dynamics, x0, controls)
save_video(os.path.join(args.logdir, 'actual_rollout.mp4'), video_frames)


## run evaluation
class Model(nn.Module):
    def __init__(self, state_size, action_size):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_size)
        )

    def forward(self, x):
        return self.layers(x)

state_size = 4
action_size = 1

with open('data-collection/data/cartpole/observations.txt', 'r') as f:
    observations = np.loadtxt(f).reshape(-1, state_size)
with open('data-collection/data/cartpole/actions.txt', 'r') as f:
    actions = np.loadtxt(f).reshape(-1, action_size)
with open('data-collection/data/cartpole/next_observations.txt', 'r') as f:
    next_observations = np.loadtxt(f).reshape(-1, state_size)

observation_mean = np.mean(observations, axis=0)
action_mean = np.mean(actions, axis=0)

observation_std = np.std(observations, axis=0)
action_std = np.std(actions, axis=0)


model = Model(state_size, action_size)
model.load_state_dict(torch.load('model-training/saved-models/cartpole/state-dict.pt'))
model.eval()

dynamics.set_state(x0)

predicted_trajectory = [x0.copy()]
predicted_video_frames = [dynamics.render()]

x = x0
for control in controls:
    normalized_state = (x - observation_mean) / observation_std
    normalized_action = (control - action_mean) / action_std
    normalized_input = torch.from_numpy(np.concatenate(normalized_state, normalized_action)).float()
    delta = model(normalized_input)
    normalized_output = normalized_input + delta
    x = normalized_output * observation_std + observation_mean

    predicted_trajectory.append(x)
    dynamics.set_state(x)
    predicted_video_frames.append(dynamics.render())

save_video(os.path.join(args.logdir, 'predicted_rollout.mp4'), predicted_video_frames)

print("\n\npredicted trajectory:")
print(predicted_trajectory)
print('\n\nactual trajectory:')
print(video_trajectory)
