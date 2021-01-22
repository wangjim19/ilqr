import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader

state_size = 4
action_size = 1

lr = 0.0001
batch_size = 40
n_epochs = 100

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
    return input_batch.astype(jnp.float32), label_batch.astype(jnp.float32)


def mse(yhat, y):
    return jnp.mean(jnp.square(y - yhat))

def loss_fn(inputs, labels):
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(state_size),
    ])
    yhat = mlp(inputs)
    return mse(yhat, labels)

loss_fn_t = hk.without_apply_rng(hk.transform(loss_fn))

rng = jax.random.PRNGKey(0)

x, y = get_inputs_labels(next(iter(train_loader)))
print('initializing params')
params = loss_fn_t.init(rng, x, y)

def sgd(param, update):
    return param - lr * update
print('starting training')
for e in range(n_epochs):
    for data_batch in train_loader:
        x, y = get_inputs_labels(data_batch)
        grads = jax.grad(loss_fn_t.apply)(params, x, y)
        params = jax.tree_multimap(sgd, params, grads)
