import time
import os
import numpy as np
import gtimer as gt
from tap import Tap
import pdb
import gym

from ilqr.utils.visualization import save_video

state_size=12
action_size=3

with open('data-collection/data/hopper/observations.txt', 'r') as f:
    observations = np.loadtxt(f).reshape(-1, state_size)
with open('data-collection/data/hopper/actions.txt', 'r') as f:
    actions = np.loadtxt(f).reshape(-1, action_size)
with open('data-collection/data/hopper/next_observations.txt', 'r') as f:
    next_observations = np.loadtxt(f).reshape(-1, state_size)
with open('data-collection/data/hopper/terminals.txt', 'r') as f:
    terminals = np.loadtxt(f).reshape(-1)

trajectories = []
action_sequences = []
current_traj = []
current_actions = []
for i, terminal in enumerate(terminals):
    current_traj.append(observations[i])
    current_actions.append(actions[i])
    if terminal == 1.0:
        current_traj.append(next_observations[i])
        trajectories.append(current_traj)
        action_sequences.append(current_actions)
        current_traj = []
        current_actions = []

for i,traj in enumerate(trajectories):
    print(i, len(traj))
actual_trajectory = trajectories[3130]
controls = action_sequences[3130]
x0 = actual_trajectory[0]


#make actual video
env = gym.make('Hopper-v2')
actual_video_frames = []
for x in actual_trajectory:
    env.set_state(x[:6], x[6:])
    actual_video_frames.append(env.sim.render(512, 512))
save_video(os.path.join('logs/hopper', 'actual_rollout.mp4'), actual_video_frames)

## run evaluation
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, state_size, action_size):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
	    nn.ReLU(),
            nn.Linear(256, state_size)
        )

    def forward(self, x):
        return self.layers(x)


'''with open('data-collection/data/cartpole/observations.txt', 'r') as f:
    observations = np.loadtxt(f).reshape(-1, state_size)
with open('data-collection/data/cartpole/actions.txt', 'r') as f:
    actions = np.loadtxt(f).reshape(-1, action_size)
with open('data-collection/data/cartpole/next_observations.txt', 'r') as f:
    next_observations = np.loadtxt(f).reshape(-1, state_size)

observation_mean = np.mean(observations, axis=0)
action_mean = np.mean(actions, axis=0)

observation_std = np.std(observations, axis=0)
action_std = np.std(actions, axis=0)

print("observation_mean", observation_mean)
print("observation_std", observation_std)
print("action_mean", action_mean)
print("action_std", action_std)'''


model = Model(state_size, action_size)
model.load_state_dict(torch.load('torch/saved-models/hopper/state-dict.pt'))
model.eval()


env.set_state(x0[:6], x0[6:])


predicted_trajectory = [x0.copy()]
predicted_video_frames = [env.sim.render(512, 512)]

x = x0.copy()
for control in controls:
    '''normalized_state = (x - observation_mean) / observation_std
    normalized_action = (control - action_mean) / action_std
    normalized_input = torch.from_numpy(np.concatenate((normalized_state, normalized_action))).float()'''
    input = torch.from_numpy(np.concatenate((x, control))).float()
    delta = model(input).detach().numpy()
    x += delta

    predicted_trajectory.append(x.copy())
    env.set_state(x[:6], x[6:])
    predicted_video_frames.append(env.sim.render(512, 512))

save_video(os.path.join('logs/hopper', 'predicted_rollout.mp4'), predicted_video_frames)

print("\n\npredicted trajectory:")
print(predicted_trajectory)
print('\n\nactual trajectory:')
print(actual_trajectory)
