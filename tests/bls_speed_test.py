import unittest
import numpy as np
from ilqr.mujoco_dynamics import MujocoDynamics
import os
import cProfile
import time
from tap import Tap
import pdb

from ilqr.mujoco_dynamics import MujocoDynamics
from ilqr.mujoco_controller import (
    iLQR,
    RecedingHorizonController,
)
from ilqr.utils.config import load_config
from ilqr.utils.logging import verbose_iteration_callback

class Parser(Tap):
    config_path: str = 'config.half_cheetah'
    path_length: int = 10
    horizon: int = 100
    mpc_initial_itrs: int = 500
    mpc_subsequent_itrs: int = 100
    logdir: str = 'logs/half-cheetah'

args = Parser().parse_args()

config = load_config(args.config_path)

dynamics = MujocoDynamics(config.xmlpath, frame_skip=1, use_multiprocessing=True)

x0 = dynamics.get_state()

us_init = np.random.uniform(-1, 1, (args.horizon, dynamics.action_size))
ilqr_no_mp = iLQR(dynamics, config.cost_fn, args.horizon)
ilqr_mp = iLQR(dynamics, config.cost_fn, args.horizon, multiprocessing = True)
mpc_no_mp = RecedingHorizonController(x0, ilqr_no_mp)
mpc_mp = RecedingHorizonController(x0, ilqr_mp)

def run_with_mp():
    mpc_trajectory, controls = mpc_mp.control(us_init,
        args.path_length,
        initial_n_iterations=args.mpc_initial_itrs,
        subsequent_n_iterations=args.mpc_subsequent_itrs,
        on_iteration=verbose_iteration_callback)
def run_without_mp():
    mpc_trajectory, controls = mpc_no_mp.control(us_init,
        args.path_length,
        initial_n_iterations=args.mpc_initial_itrs,
        subsequent_n_iterations=args.mpc_subsequent_itrs,
        on_iteration=verbose_iteration_callback)


class BLSSpeedTest(unittest.TestCase):
    def test_speed_mp(self):
        print('WITH MULTIPROCESSING FOR BLS\n\n')
        cProfile.run('run_with_mp()')
        print('\n\n\n')
    def test_speed_no_mp(self):
        print('WITHOUT MULTIPROCESSING FOR BLS\n\n')
        cProfile.run('run_without_mp()')
        print('\n\n\n')



if __name__ == '__main__':
    unittest.main()
