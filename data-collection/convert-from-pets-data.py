import sys
from scipy.io import loadmat
import numpy as np

if len(sys.argv) != 3:
    print('usage: python convert-from-pets-data.py /path/to/logs.mat /path/to/savedir')
    sys.exit()
path = sys.argv[1]
savedir = sys.argv[2]
if savedir[-1] != '/':
    savedir = savedir + '/'
if path[-8:] != 'logs.mat':
    print('specify a logs.mat file')
    sys.exit()
data = loadmat(path)
print('Loaded data')

state_size = data['observations'].shape[-1]
action_size = data['actions'].shape[-1]
trial_length = data['actions'].shape[-2]

observations = data['observations'][:, :-1, :].reshape((-1, state_size))
next_observations = data['observations'][:, 1:, :].reshape((-1, state_size))
actions = data['actions'].reshape((-1, action_size))
rewards = data['rewards'].reshape((-1, 1))
terminals = np.array([0 if i % trial_length != trial_length - 1 else 1 for i in range(observations.shape[0])])

with open(savedir + 'observations.txt', 'w') as f:
    np.savetxt(f, observations)
with open(savedir + 'actions.txt', 'w') as f:
    np.savetxt(f, actions)
with open(savedir + 'next_observations.txt', 'w') as f:
    np.savetxt(f, next_observations)
with open(savedir + 'rewards.txt', 'w') as f:
    np.savetxt(f, rewards)
with open(savedir + 'terminals.txt', 'w') as f:
    np.savetxt(f, terminals)
print('Saved data')
