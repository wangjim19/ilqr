from dmbrl.modeling.models import BNN
from dotmap import DotMap
import numpy as np
import tensorflow as tf
from dmbrl.config.halfcheetah import HalfCheetahConfigModule
import pickle

state_size = 18
action_size = 6
samples_per_bootstrap = 4


with open('data-collection/data/halfcheetah-pets/observations.txt', 'r') as f:
    observations = np.loadtxt(f).reshape(-1, state_size)
with open('data-collection/data/halfcheetah-pets/actions.txt', 'r') as f:
    actions = np.loadtxt(f).reshape(-1, action_size)
with open('data-collection/data/halfcheetah-pets/next_observations.txt', 'r') as f:
    next_observations = np.loadtxt(f).reshape(-1, state_size)
with open('data-collection/data/halfcheetah-pets/terminals.txt', 'r') as f:
    terminals = np.loadtxt(f).reshape(-1)

trajectories = []
action_sequences = []
current_traj = []
current_actions = []
for i, terminal in enumerate(terminals):
    current_traj.append(observations[i])
    current_actions.append(actions[i])
    if terminal == 1.0:
        current_traj.append(next_observations[i])
        trajectories.append(current_traj)
        action_sequences.append(current_actions)
        current_traj = []
        current_actions = []

actual_traj = np.array(trajectories[-1])
actions = np.array(action_sequences[-1])


params = {"name": "model", "model_dir": "../handful-of-trials/log/2021-01-22--23:58:50", "load_model": True}
bnn = BNN(params)

print('\n\nModel:')
for l in bnn.layers:
    print(l)
bnn.finalize(tf.train.AdamOptimizer, {"learning_rate": 1e-3})

predicted_trajs = np.empty((bnn.num_nets, samples_per_bootstrap, actual_traj.shape[0], actual_traj.shape[1]))
predicted_mean_traj = np.empty((bnn.num_nets, actual_traj.shape[0], actual_traj.shape[1]))

preprocess = HalfCheetahConfigModule.obs_preproc
postprocess = HalfCheetahConfigModule.obs_postproc

#initialize x0
for i in range(bnn.num_nets):
    for j in range(samples_per_bootstrap):
        predicted_trajs[i, j, 0, :] = actual_traj[0]
    predicted_mean_traj[i, 0, :] = actual_traj[0]

#sample rollout
for i, action in enumerate(actions):
    action_duped = np.stack([action for i in range(samples_per_bootstrap)])
    for b in range(bnn.num_nets):
        obs = predicted_trajs[b, :, i, :]
        processed_obs = preprocess(obs)
        inputs = np.hstack((processed_obs, action_duped))
        means, vars = bnn.predict(inputs, factored=True)
        means, vars = means[b], vars[b]
        predictions = np.random.normal(means, np.sqrt(vars))
        next_obs = postprocess(obs, predictions)
        predicted_trajs[b, :, i + 1, :] = next_obs

        mean_obs = predicted_mean_traj[b, i, :].reshape(1, -1)
        processed_mean = preprocess(mean_obs)
        inputs = np.hstack((processed_mean, action.reshape(1, -1)))
        means, vars = bnn.predict(inputs, factored=True)
        means, vars = means[b], vars[b]
        next_mean_obs = postprocess(mean_obs, means)
        predicted_mean_traj[b, i + 1, :] = next_mean_obs

rollout_dict = {
    "actual_trajectory": actual_traj,
    "action_sequence": actions,
    "predicted_trajectories": predicted_trajs,
    "predicted_mean_trajectories": predicted_mean_traj
}

pickle.dump(rollout_dict, open("pets/sampled_rollout.pkl", "wb"))
