import time
import os
import numpy as np
from tap import Tap
import pdb

from ilqr.cost import FiniteDiffCost
from ilqr.mujoco_dynamics import MujocoDynamics
from ilqr.mujoco_controller import (
	iLQR,
	RecedingHorizonController,
)
from ilqr.utils.rollouts import monitored_rollout
from ilqr.utils.visualization import save_video


class Parser(Tap):
    path_length: int = 100
    horizon: int = 100
    mpc_initial_itrs: int = 500
    mpc_subsequent_itrs: int = 100
    xmlpath: str = 'ilqr/xmls/inverted_pendulum.xml'
    logpath: str = 'logs/cartpole-receding-horizon'

args = Parser().parse_args()

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = xs[-1]
    print("iteration", iteration_count, info, J_opt, final_state)

dynamics = MujocoDynamics(args.xmlpath, frame_skip = 2, use_multiprocessing = True)
print(dynamics.dt)

cost = FiniteDiffCost(lambda x, u, i: 2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2 + u[0] ** 2,
                      lambda x, i: (2 * (x[0] ** 2) + 10 * (x[1] ** 2) + x[2] ** 2 + x[3] ** 2),
                      4, 1, use_multiprocessing = True)

x0 = np.array([0.0, np.random.uniform(-np.pi, np.pi), 0.0, 0.0])

us_init = np.random.uniform(-1, 1, (args.horizon, dynamics.action_size))
ilqr = iLQR(dynamics, cost, args.horizon)
mpc = RecedingHorizonController(x0, ilqr)

t0 = time.time()
mpc_trajectory, controls = mpc.control(us_init,
	args.path_length,
	initial_n_iterations=args.mpc_initial_itrs,
	subsequent_n_iterations=args.mpc_subsequent_itrs,
	on_iteration=on_iteration)
    
print('time', time.time() - t0)

video_trajectory, video_frames = monitored_rollout(dynamics, x0, controls)

save_video(os.path.join(args.logpath, 'rollout.mp4'), video_frames)

## trajectory from rolling out resulting control sequence
## should match the trajectory given by the mpc solver
assert (mpc_trajectory == video_trajectory).all()
pdb.set_trace()

