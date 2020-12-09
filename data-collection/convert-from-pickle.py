import pickle
import numpy as np

with open('data-collection/data/cartpole-SAC/inverted_pendulum.pkl', 'rb') as f:
    data = pickle.load(f)
with open('data-collection/data/cartpole-SAC/observations.txt', 'w') as f:
    np.savetxt(f, data["_observations"])
with open('data-collection/data/cartpole-SAC/actions.txt', 'w') as f:
    np.savetxt(f, data["_actions"])
with open('data-collection/data/cartpole-SAC/next_observations.txt', 'w') as f:
    np.savetxt(f, data["_next_obs"])
with open('data-collection/data/cartpole-SAC/rewards.txt', 'w') as f:
    np.savetxt(f, data["_rewards"])
with open('data-collection/data/cartpole-SAC/terminals.txt', 'w') as f:
    np.savetxt(f, data["_terminals"])
