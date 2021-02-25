import time
import os
import numpy as np
import gtimer as gt
from tap import Tap
import pdb
import pickle

from ilqr.jax_dynamics import JaxEnsembleDynamics
from ilqr.learned_controller import (
    iLQR,
    RecedingHorizonController,
)
from ilqr.utils.config import load_config
from ilqr.utils.rollouts import monitored_rollout
from ilqr.utils.visualization import save_video
from ilqr.utils.logging import verbose_iteration_callback, cost_only_callback

from envs.half_cheetah import HalfCheetahEnv


class Parser(Tap):
    config_path: str = 'config.half_cheetah_learned'
    path_length: int = 100
    horizon: int = 50
    mpc_initial_itrs: int = 500
    mpc_subsequent_itrs: int = 200
    logdir: str = 'logs/half-cheetah-learned'

args = Parser().parse_args()

config = load_config(args.config_path)
dynamics = JaxEnsembleDynamics(config.model_fn, config.params, config.ensemble_size, config.state_size, config.action_size)

## hard-code starting state for reproducibility
x0 = pickle.load(open("pets/sampled_rollout.pkl", "rb"))["actual_trajectory"][0]
print('x0:', x0)

np.random.seed(125)
us_init = np.random.uniform(-1,1, (args.horizon, dynamics.action_size))
ilqr = iLQR(dynamics, config.cost_fn, args.horizon)
mpc = RecedingHorizonController(ilqr, HalfCheetahEnv(), x0)
gt.stamp('initialization')

## run ilqr
time0 = time.time()
mpc_trajectory, controls, video_frames, predicted_trajs, planned_actions = mpc.control(us_init,
    args.path_length,
    step_size=2,
    initial_n_iterations=args.mpc_initial_itrs,
    subsequent_n_iterations=args.mpc_subsequent_itrs,
    on_iteration=cost_only_callback)
print("total time:", time.time() - time0)
gt.stamp('control')

pickle.dump(mpc_trajectory, open(args.logdir + '/mpc_trajectory.pkl', 'wb'))
pickle.dump(controls, open(args.logdir + '/controls.pkl', 'wb'))
pickle.dump(predicted_trajs, open(args.logdir + '/predicted_trajs.pkl', 'wb'))
pickle.dump(planned_actions, open(args.logdir + '/planned_actions.pkl', 'wb'))

## save rollout video to disk
save_video(os.path.join(args.logdir, 'rollout.mp4'), video_frames)
gt.stamp('video logging')

## print time information
print(gt.report())

## trajectory from rolling out resulting control sequence
## should match the trajectory given by the mpc solver
#assert (mpc_trajectory == video_trajectory).all()
#pdb.set_trace()
