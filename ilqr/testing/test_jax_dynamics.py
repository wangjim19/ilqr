import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader
import jax.experimental.optimizers as optimizers
import pickle
import time
import math

from ilqr.jax_dynamics import JaxEnsembleDynamics

state_size = 18
action_size = 6

ensemble_size = 5

rollout = pickle.load(open("pets/sampled_rollout.pkl", "rb"))
xs = rollout["actual_trajectory"]
us = rollout["action_sequence"]

#input shape = (ensemble_size, batch_size, state_size+action_size)
#output shape = (ensemble_size, batch_size, state_size)
def model_fn(inputs):
    mlps = []
    for i in range(ensemble_size):
        mlps.append(hk.Sequential([
            hk.Linear(200), jax.nn.swish,
            hk.Linear(200), jax.nn.swish,
            hk.Linear(200), jax.nn.swish,
            hk.Linear(200), jax.nn.swish,
            hk.Linear(state_size),
        ]))
    return jnp.stack([mlps[i](inputs[i]) for i in range(ensemble_size)])

params = pickle.load(open("jax/saved-models/halfcheetah-ensemble/params.pkl", "rb"))

print('Testing average aggregation mode')
dynamics = JaxEnsembleDynamics(model_fn, params, ensemble_size, state_size, action_size)
dynamics.set_state(xs[0])

print('x0:', xs[0])

print('Testing step:')
t0 = time.time()
print(dynamics.step(us[0]))
print('initial time:', time.time() - t0)
t0 = time.time()
for i in range(10):
    dynamics.step(us[i + 1])
print('average time:', (time.time() - t0) / 10)

print('Testing f_derivs:')
t0 = time.time()
F_x, F_u = dynamics.f_derivs(xs[:51], us[:50])
print('F_x.shape:', F_x.shape)
print('F_u.shape:', F_u.shape)
print('F_x[0]:', F_x[0])
print('F_u[0]:', F_u[0])
print("time:", time.time() - t0)
