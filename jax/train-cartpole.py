import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader
import jax.experimental.optimizers as optimizers

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

def model_fn(inputs):
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(state_size),
    ])
    return mlp(inputs)

model = hk.without_apply_rng(hk.transform(model_fn))

def loss_fn(params, inputs, labels):
    predictions = model.apply(params, inputs)
    loss = jnp.mean(jnp.square(labels - predictions))
    return loss

@jax.jit
def update(step, opt_state, inputs, labels):
    loss, grads = jax.value_and_grad(loss_fn)(get_params(opt_state), inputs, labels)
    opt_state = opt_update(step, grads, opt_state)
    return loss, opt_state



rng = jax.random.PRNGKey(0)

x, y = get_inputs_labels(next(iter(train_loader)))
print(x.shape, y.shape)
print('initializing params')
params = model.init(rng, x)

print('initializing optimizer')
opt_init, opt_update, get_params = optimizers.adam(lr)
opt_state = opt_init(params)


print('starting training')

test_loss_history = []
train_loss_history = []
i=0

for e in range(n_epochs):
    losses = []
    for data_batch in train_loader:
        x, y = get_inputs_labels(data_batch)
        loss, opt_state = update(i, opt_state, x, y)
        i += 1
        losses.append(loss)
    print("EPOCH", epoch)
    print("\naverage train loss =", sum(losses) / len(losses))
    train_loss_history.append(sum(losses)/len(losses))

    #EVALUATION
    test_losses = []
    for data_batch in test_loader:
        x, y = get_inputs_labels(data_batch)
        loss = loss_fn(get_params(opt_state), x, y)
        test_losses.append(loss)
    print("test loss =", sum(test_losses) / len(test_losses))
    print('')
    test_loss_history.append(sum(test_losses) / len(test_losses))
