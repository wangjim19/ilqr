import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

if torch.cuda.is_available():
    print("Using CUDA")
    device = 'cuda'
else:
    print("Using CPU")
    device = 'cpu'

#define model
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

#retrieve data
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

observations_normalized = (observations - observation_mean) / observation_std
actions_normalized = (actions - action_mean) / action_std
next_observations_normalized = (next_observations - observation_mean) / observation_std

deltas = next_observations_normalized - observations_normalized

#learn f(normalized observation, normalized action) -> delta(normalized observation)
inputs = torch.from_numpy(np.hstack((observations_normalized, actions_normalized))).float()
labels = torch.from_numpy(deltas).float()
train_data = TensorDataset(inputs, labels)
print("size of dataset:", len(train_data))

#define training parameters
lr = 0.0001
n_epochs = 20
batch_size = 20

model = Model(state_size, action_size).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

#train
def train_step(x, y):
    model.train() #set model to train mode

    optimizer.zero_grad()
    yhat = model(x)
    loss = loss_fn(y, yhat)
    loss.backward()
    optimizer.step()

    return loss.item()


for epoch in range(n_epochs):
    losses = []
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        loss = train_step(x_batch, y_batch)
        losses.append(loss)
    print("epoch", epoch, ": average loss =", sum(losses) / len(losses))
