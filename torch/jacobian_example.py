import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import time

if torch.cuda.is_available():
    print("Using CUDA")
    device = 'cuda'
else:
    print("Using CPU")
    device = 'cpu'

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


state_size = 4
action_size = 1

batch_size = 40

with open('data-collection/data/cartpole/observations.txt', 'r') as f:
    observations = np.loadtxt(f).reshape(-1, state_size)
with open('data-collection/data/cartpole/actions.txt', 'r') as f:
    actions = np.loadtxt(f).reshape(-1, action_size)
with open('data-collection/data/cartpole/next_observations.txt', 'r') as f:
    next_observations = np.loadtxt(f).reshape(-1, state_size)

deltas = next_observations - observations
data = np.hstack((observations, actions, deltas))
np.random.shuffle(data)
train_size = (data.shape[0] * 4) // 5
train_data = data[:train_size]
test_data = data[train_size:]
print("size of train data:", train_data.shape[0])
print("size of test data:", test_data.shape[0])

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

def get_inputs_labels(data_batch):
    input_batch = data_batch[:, :state_size + action_size].numpy()
    label_batch = data_batch[:, state_size + action_size:].numpy()
    return input_batch, label_batch


print('loading model')
model = Model(state_size, action_size)
model.load_state_dict(torch.load('torch/saved-models/cartpole/state-dict.pt'))
model.eval()

train_iter = iter(train_loader)

t0 = time.time()
x, y = get_inputs_labels(next(train_iter))
F = torch.zeros((batch_size, state_size, state_size + action_size))
input = torch.from_numpy(x).float().requires_grad_(True)
deltas = model(input)
for i in range(batch_size):
    for j in range(state_size):
        F[:, j, :] += torch.autograd.grad(deltas[i][j], input, retain_graph=True)[0]
    F[i, :, :state_size] += torch.eye(state_size) # add ds/ds to d(s' - s)/ds to get d(s')/ds
print('initial time:', time.time() - t0)
print(F[0])

t0 = time.time()
for _ in range(10):
    x, y = get_inputs_labels(next(train_iter))
    F = torch.zeros((batch_size, state_size, state_size + action_size))
    input = torch.from_numpy(x).float().requires_grad_(True)
    deltas = model(input)
    for i in range(batch_size):
        for j in range(state_size):
            F[:, j, :] += torch.autograd.grad(deltas[i][j], input, retain_graph=True)[0]
        F[i, :, :state_size] += torch.eye(state_size) # add ds/ds to d(s' - s)/ds to get d(s')/ds
print("Average time:", (time.time() - t0) / 10)
print(F[0])
