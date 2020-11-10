import unittest
import numpy as np
from ilqr.mujoco_dynamics import MujocoDynamics
from mujoco_py_deriv import MjDerivative
import os
import cProfile

xml_path = os.path.join(os.getcwd(), 'ilqr', 'xmls', 'inverted_pendulum.xml')

dynamics = MujocoDynamics(xml_path, x_eps = 1e-6, u_eps = 1e-6)

sim = dynamics.sim

model = sim.model

dmain = sim.data

deriv_obj_pos = MjDerivative(model, dmain, ["qacc"], ["qpos"], nwarmup = 1, niter = 1)
deriv_obj_vel = MjDerivative(model, dmain, ["qacc"], ["qvel"], nwarmup = 1, niter = 1)


def mjderiv_f_x(state, action):
    dynamics.set_state(state)
    dmain.ctrl[:] = action
    #check if data remains same after deriv compute
    deriv_pos = deriv_obj_pos.compute()
    deriv_vel = deriv_obj_vel.compute()
    dqacc_dqpos = deriv_pos[0][0]
    dqacc_dqvel = deriv_vel[0][0]

    f_x = np.eye(dynamics.state_size)
    f_x[:dynamics.state_size//2, :dynamics.state_size//2] += 0.5 * (dynamics.dt ** 2) * dqacc_dqpos
    f_x[:dynamics.state_size//2, dynamics.state_size//2:] += np.eye(dynamics.state_size // 2) * dynamics.dt
    f_x[:dynamics.state_size//2, dynamics.state_size//2:] += 0.5 * (dynamics.dt ** 2) * dqacc_dqvel

    f_x[dynamics.state_size//2:, :dynamics.state_size//2] += dynamics.dt * dqacc_dqpos
    f_x[dynamics.state_size//2:, dynamics.state_size//2:] += dynamics.dt * dqacc_dqvel

    return f_x
def run_finitediff():
    for i in range(100):
        dynamics.f_x(np.zeros(dynamics.state_size), np.zeros(dynamics.action_size))
def run_mjderiv():
    for i in range(100):
        mjderiv_f_x(np.zeros(dynamics.state_size), np.zeros(dynamics.action_size))
class DerivTest(unittest.TestCase):
    def test_speed(self):
        
        print('FINITE DIFF RESULTS\n\n')
        cProfile.run('run_finitediff()')
        print('\n\n\nMJ-PY-DERIV RESULTS\n\n')
        cProfile.run('run_mjderiv()')
        print('\n\n\n')

    def test_accuracy(self):
        state = np.array([0.0, np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)])
        action = np.array([0.0])
        print('TESTING STATE:', state)
        finite_diff_result = dynamics.f_x(state, action)
        mjderiv_result = mjderiv_f_x(state, action)
        print('\n\nFINITE DIFF RESULTS\n')
        print(finite_diff_result)
        print('\n\nMJ-PY-DERIV RESULTS\n')
        print(mjderiv_result)
        print('\n\nDIFFERENCE\n')
        print(mjderiv_result - finite_diff_result)
        assert np.allclose(finite_diff_result, mjderiv_result), "they aren't very close"


if __name__ == '__main__':
    unittest.main()
