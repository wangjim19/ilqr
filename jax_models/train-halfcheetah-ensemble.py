import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader
import jax.experimental.optimizers as optimizers
import pickle
import time
from jax_models.layers.ensemble_linear import EnsembleLinear

state_size = 18
action_size = 6

lr = 0.0001
batch_size = 32
n_epochs = 1000

ensemble_size = 5

with open('data-collection/data/halfcheetah-pets/observations.txt', 'r') as f:
    observations = np.loadtxt(f).reshape(-1, state_size)
with open('data-collection/data/halfcheetah-pets/actions.txt', 'r') as f:
    actions = np.loadtxt(f).reshape(-1, action_size)
with open('data-collection/data/halfcheetah-pets/next_observations.txt', 'r') as f:
    next_observations = np.loadtxt(f).reshape(-1, state_size)

deltas = next_observations - observations
data = np.hstack((observations, actions, deltas))
train_size = (data.shape[0] * 9) // 10
np.random.shuffle(data)

total_train_data = data[:train_size]
test_data = data[train_size:]
train_data = []

ensemble_train_size = (total_train_data.shape[0] * 3) // 4

for e in range(ensemble_size):
    np.random.shuffle(total_train_data)
    train_data.append(total_train_data[:ensemble_train_size].copy())

print("size of total train data:", total_train_data.shape[0])
print("size of test data:", test_data.shape[0])
print("size of train data per ensemble", train_data[0].shape[0])

train_loaders = []
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
for e in range(ensemble_size):
    train_loaders.append(DataLoader(dataset=train_data[e], batch_size=batch_size, shuffle=True))

def get_inputs_labels(data_batch):
    input_batch = data_batch[:, :state_size + action_size].numpy()
    label_batch = data_batch[:, state_size + action_size:].numpy()
    return input_batch.astype(jnp.float32), label_batch.astype(jnp.float32)

#input shape = (ensemble_size, batch_size, state_size+action_size)
#output shape = (ensemble_size, batch_size, state_size)
def model_fn(inputs):
    mlp = hk.Sequential([
        EnsembleLinear(ensemble_size, 200), jax.nn.swish,
        EnsembleLinear(ensemble_size, 200), jax.nn.swish,
        EnsembleLinear(ensemble_size, 200), jax.nn.swish,
        EnsembleLinear(ensemble_size, 200), jax.nn.swish,
        EnsembleLinear(ensemble_size, state_size),
    ])
    return mlp(inputs)

model = hk.without_apply_rng(hk.transform(model_fn))

@jax.jit
def loss_fn(params, inputs, labels):
    predictions = model.apply(params, inputs)
    losses = jnp.mean(jnp.square(labels - predictions), axis=(1,2))
    return losses

def aggregated_loss(params, inputs, labels):
    return jnp.sum(loss_fn(params, inputs, labels))

@jax.jit
def update(step, opt_state, inputs, labels):
    loss, grads = jax.value_and_grad(aggregated_loss)(get_params(opt_state), inputs, labels)
    opt_state = opt_update(step, grads, opt_state)
    return loss, opt_state



rng = jax.random.PRNGKey(np.random.randint(0, 256))

print('initializing params')
xs = []
ys = []
for t in train_loaders:
    x, y = get_inputs_labels(next(iter(t)))
    xs.append(x)
    ys.append(y)
xs = np.array(xs)
ys = np.array(ys)
params = model.init(rng, xs)

print('initializing optimizers')
opt_init, opt_update, get_params = optimizers.adam(lr)
opt_state = opt_init(params)


print('starting training\n')

test_loss_history = []
train_loss_history = []
i=0

t0 = time.time()
for e in range(n_epochs):
    losses = []
    for data_batches in zip(*train_loaders):
        xs, ys = [], []
        for data_batch in data_batches:
            x, y = get_inputs_labels(data_batch)
            xs.append(x)
            ys.append(y)
        loss, opt_state = update(i, opt_state, np.array(xs), np.array(ys))
        i += 1
        losses.append(loss)
    losses = np.array(losses)
    print("EPOCH", e)
    print("average aggregated train loss =", np.mean(losses) / ensemble_size)
    train_loss_history.append(np.mean(losses) / ensemble_size)

    #EVALUATION
    test_losseses = []
    for data_batch in test_loader:
        x, y = get_inputs_labels(data_batch)
        xs = np.array([x for i in range(ensemble_size)])
        ys = np.array([y for i in range(ensemble_size)])
        losses = loss_fn(get_params(opt_state), xs, ys)
        test_losseses.append(losses)
    test_losseses = np.array(test_losseses)
    print("test loss per bootstrap =", np.mean(test_losseses, axis=0))
    print('')
    test_loss_history.append(np.mean(test_losseses, axis=0))
print("Time:", time.time() - t0)

pickle.dump(get_params(opt_state), open("jax/saved-models/halfcheetah-ensemble/params.pkl", "wb"))

with open('jax/saved-models/halfcheetah-ensemble/train_losses.txt', 'w') as f:
    np.savetxt(f, np.array(train_loss_history))
with open('jax/saved-models/halfcheetah-ensemble/test_losses.txt', 'w') as f:
    np.savetxt(f, np.array(test_loss_history))
