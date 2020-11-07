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
from ilqr.utils.visualization import save_video


class Parser(Tap):
    n_iters: int = 100
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
controls = mpc.control(us_init, initial_n_iterations = args.mpc_initial_itrs, subsequent_n_iterations = args.mpc_subsequent_itrs, on_iteration = on_iteration)
us = []
for i in range(args.n_iters):
    print('ITERATION', i, '\n')
    us.append(next(controls)[1])
    
print('time', time.time() - t0)

dynamics.set_state(x0)
print(dynamics.get_state())
video = []
for i, u in enumerate(us):
    print (i, u)
    print(dynamics.step(u))
    print('')
    video.append(dynamics.sim.render(512, 512)[::-1])

save_video(os.path.join(args.logpath, 'rollout.mp4'), video)

