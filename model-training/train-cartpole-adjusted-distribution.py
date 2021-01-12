import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

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
            nn.Linear(256, 256),
	        nn.ReLU(),
            nn.Linear(256, state_size)
        )

    def forward(self, x):
        return self.layers(x)

#retrieve data
state_size = 4
action_size = 1

optimal_portion = 0.5

with open('data-collection/data/cartpole/observations.txt', 'r') as f:
    observations = np.loadtxt(f).reshape(-1, state_size)
with open('data-collection/data/cartpole/actions.txt', 'r') as f:
    actions = np.loadtxt(f).reshape(-1, action_size)
with open('data-collection/data/cartpole/next_observations.txt', 'r') as f:
    next_observations = np.loadtxt(f).reshape(-1, state_size)

with open('data-collection/data/cartpole-optimal/observations.txt', 'r') as f:
    optimal_observations = np.loadtxt(f).reshape(-1, state_size)
with open('data-collection/data/cartpole-optimal/actions.txt', 'r') as f:
    optimal_actions = np.loadtxt(f).reshape(-1, action_size)
with open('data-collection/data/cartpole-optimal/next_observations.txt', 'r') as f:
    optimal_next_observations = np.loadtxt(f).reshape(-1, state_size)

total_size = min(len(observations), len(optimal_observations))
optimal_amount = int(total_size * optimal_portion)

observations = np.concatenate((observations[:total_size - optimal_amount], optimal_observations[:optimal_amount]))
actions = np.concatenate((actions[:total_size - optimal_amount], optimal_actions[:optimal_amount]))
next_observations = np.concatenate((next_observations[:total_size - optimal_amount], optimal_next_observations[:optimal_amount]))

'''observation_mean = np.mean(observations, axis=0)
action_mean = np.mean(actions, axis=0)

observation_std = np.std(observations, axis=0)
action_std = np.std(actions, axis=0)

observations_normalized = (observations - observation_mean) / observation_std
actions_normalized = (actions - action_mean) / action_std
next_observations_normalized = (next_observations - observation_mean) / observation_std

deltas = next_observations_normalized - observations_normalized'''
deltas = next_observations - observations

#process data
#inputs = torch.from_numpy(np.hstack((observations_normalized, actions_normalized))).float()
inputs = torch.from_numpy(np.hstack((observations, actions))).float()
labels = torch.from_numpy(deltas).float()
dataset = TensorDataset(inputs, labels)
train_size = (len(dataset) * 4) // 5
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])
print("size of train data:", len(train_data))
print("size of test data:", len(test_data))

#define training parameters
lr = 0.0001
n_epochs = 100
batch_size = 40

model = Model(state_size, action_size).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

#train
def train_step(x, y):
    model.train() #set model to train mode

    optimizer.zero_grad()
    yhat = model(x)
    loss = loss_fn(yhat, y)
    loss.backward()
    optimizer.step()

    return loss.item()

test_loss_history = []
train_loss_history = []
for epoch in range(n_epochs):
    losses = []
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        loss = train_step(x_batch, y_batch)
        losses.append(loss)
    print("EPOCH", epoch)
    print("\naverage train loss =", sum(losses) / len(losses))
    train_loss_history.append(sum(losses)/len(losses))
    with torch.no_grad():
        test_losses = []
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            model.eval()

            y_hat = model(x_batch)
            test_loss = loss_fn(y_hat, y_batch).item()
            test_losses.append(test_loss)

    print("test loss =", sum(test_losses) / len(test_losses))
    print('')
    test_loss_history.append(sum(test_losses) / len(test_losses))
    for _ in range(3):
        i = np.random.choice(len(observations))
        obs = observations[i]
        ac = actions[i]
        label = deltas[i]

        print("obs:", obs, "ac:", ac)
        print(label, "LABEL")

        model.eval()
        delta = model(torch.from_numpy(np.concatenate((obs, ac))).float().to(device))
        delta = delta.detach().cpu().numpy()
        print(delta, "PREDICTED")
        print('')
    print('')

torch.save(model.state_dict(), 'model-training/saved-models/cartpole-adjusted-distribution/state-dict.pt')

with open('model-training/saved-models/cartpole-adjusted-distribution/train_losses.txt', 'w') as f:
    np.savetxt(f, np.array(train_loss_history))
with open('model-training/saved-models/cartpole-adjusted-distribution/test_losses.txt', 'w') as f:
    np.savetxt(f, np.array(test_loss_history))
