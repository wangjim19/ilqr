import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader
import jax.experimental.optimizers as optimizers
import pickle
import time
import math
from jax_models.layers.ensemble_linear import EnsembleLinear

state_size = 18
action_size = 6

ensemble_size = 5

rollout = pickle.load(open("pets/sampled_rollout.pkl", "rb"))

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

print('initializing params')
params = pickle.load(open("jax/saved-models/halfcheetah-ensemble/params.pkl", "rb"))

print('starting evaluation')
actual_traj = rollout["actual_trajectory"]
actions = rollout["action_sequence"]
x = np.array([actual_traj[0] for i in range(ensemble_size)])

predicted_trajs = np.empty((ensemble_size, actual_traj.shape[0], actual_traj.shape[1]))
predicted_trajs[:, 0, :] = x

predict = jax.jit(lambda inputs: model.apply(params, inputs))
for i, a in enumerate(actions):
    a = np.array([a for _ in range(ensemble_size)])
    inputs = np.hstack((x, a))
    x += predict(inputs)
    if math.isnan(x[0][0]):
        print(inputs)
    predicted_trajs[:, i + 1, :] = x

rollout_dict = {
    "actual_trajectory": actual_traj,
    "action_sequence": actions,
    "predicted_trajectories": predicted_trajs,
}

pickle.dump(rollout_dict, open("jax/saved-models/halfcheetah-ensemble/sampled_rollout.pkl", "wb"))
