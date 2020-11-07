import os
import numpy as np
import multiprocessing as mp

from .base import Cost

class ExactCost(Cost):

    def __init__(self,
                 l,
                 l_terminal,
                 state_size,
                 action_size,
                 x_eps=None,
                 u_eps=None,
                 use_multiprocessing = False):
        
        self._l = l
        self._l_terminal = l_terminal
        self._state_size = state_size
        self._action_size = action_size

        self._x_eps = x_eps if x_eps else np.sqrt(np.finfo(float).eps)
        self._u_eps = u_eps if x_eps else np.sqrt(np.finfo(float).eps)

        self._x_eps_hess = np.sqrt(self._x_eps)
        self._u_eps_hess = np.sqrt(self._u_eps)

        self.multiprocessing = use_multiprocessing
        if self.multiprocessing:
            self._pool = mp.Pool(initializer = ExactCost._worker_init,
                                 initargs = (l, l_terminal, state_size, action_size, x_eps, u_eps, False))

        super(ExactCost, self).__init__()

    @staticmethod
    def _worker_init(l,
                     l_terminal,
                     state_size,
                     action_size,
                     x_eps,
                     u_eps,
                     use_multiprocessing):
        """
        Initializes sims for workers in multiprocessing Pool.
        """
        global cost
        cost = ExactCost(l, l_terminal, state_size, action_size, x_eps, u_eps, use_multiprocessing)
        print("Finished loading process", os.getpid())

    @staticmethod
    def _worker(x, u, i):
        return (cost.l(x, u, i), cost.l_x(x, u, i), cost.l_u(x, u, i), cost.l_xx(x, u, i), cost.l_ux(x, u, i), cost.l_uu(x, u, i))

    def l_derivs(self, xs, us):
        if self.multiprocessing:
            results = self._pool.starmap(ExactCost._worker, [(xs[i], us[i], i) for i in range(us.shape[0])], chunksize = us.shape[0] // mp.cpu_count())
            return ([result[0] for result in results],
                    [result[1] for result in results],
                    [result[2] for result in results],
                    [result[3] for result in results],
                    [result[4] for result in results],
                    [result[5] for result in results])

        L = [self.l(xs[i], us[i], i) for i in range(us.shape[0])]
        L_x = [self.l_x(xs[i], us[i], i) for i in range(us.shape[0])]
        L_u = [self.l_u(xs[i], us[i], i) for i in range(us.shape[0])]
        L_xx = [self.l_xx(xs[i], us[i], i) for i in range(us.shape[0])]
        L_ux = [self.l_ux(xs[i], us[i], i) for i in range(us.shape[0])]
        L_uu = [self.l_uu(xs[i], us[i], i) for i in range(us.shape[0])]
        return (L, L_x, L_u, L_xx, L_ux, L_uu)

    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            return self._l_terminal(x, i)

        return self._l(x, u, i)

    
    def l_x(self, x, u, i, terminal=False):
        if terminal:
            return np.array([4, 20, 2, 2]) * x
        return np.array([4, 20, 2, 2]) * x
    def l_u(self, x, u, i, terminal = False):
        if terminal:
            return np.zeros(1)
        return np.array([2]) * u
    def l_xx(self, x, u, i, terminal=False):
        deriv = np.zeros((4, 4))
        deriv[0][0] = 4
        deriv[1][1] = 20
        deriv[2][2] = 2
        deriv[3][3] = 2
        return deriv
    def l_ux(self, x, u, i, terminal=False):
        return np.zeros((1, 4))
    def l_uu(self, x, u, i, terminal=False):
        if terminal:
            return np.zeros((1, 1))
        return np.array([[2]])
