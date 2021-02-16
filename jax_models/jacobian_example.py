import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader
import jax.experimental.optimizers as optimizers
from jax import jit, vmap, jacfwd
import pickle
import time

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
    return input_batch.astype(jnp.float32), label_batch.astype(jnp.float32)

def model_fn(inputs):
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(state_size),
    ])
    return mlp(inputs)

model = hk.without_apply_rng(hk.transform(model_fn))


print('loading params')
params = pickle.load(open("jax/saved-models/cartpole/params.pkl", "rb"))

#Jacobian example
f = lambda inputs: jnp.add(model.apply(params, inputs), inputs[:state_size]) #add inputs because model predicts deltas
jac_fn = jacfwd(f)
vmapped_jac_fn = jit(vmap(jac_fn))

train_iter = iter(train_loader)

t0 = time.time()
x, y = get_inputs_labels(next(train_iter))
J = vmapped_jac_fn(x)
print("initial time:", time.time() - t0)
print(J[0])

t0 = time.time()
for _ in range(100):
    x, y = get_inputs_labels(next(train_iter))
    J = vmapped_jac_fn(x)
print("Average time:", (time.time() - t0) / 100)
print(J[0])
