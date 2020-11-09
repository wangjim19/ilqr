import os
import numpy as np
import multiprocessing as mp

from .base import Cost

class ExactCost(Cost):

    def __init__(self,
                 l,
                 l_x,
                 l_u,
                 l_xx,
                 l_ux,
                 l_uu,
                 use_multiprocessing = False):
        """
        Args:
            see unimplemented methods below.
        """
        self.l = l
        self.l_x = l_x
        self.l_u = l_u
        self.l_xx = l_xx
        self.l_ux = l_ux
        self.l_uu = l_uu


        self.multiprocessing = use_multiprocessing
        if self.multiprocessing:
            self._pool = mp.Pool(initializer = ExactCost._worker_init,
                                 initargs = (l, l_x, l_u, l_xx, l_ux, l_uu, False))

        super(ExactCost, self).__init__()

    @staticmethod
    def _worker_init(l,
                     l_x,
                     l_u,
                     l_xx,
                     l_ux,
                     l_uu,
                     use_multiprocessing):
        """
        Initializes sims for workers in multiprocessing Pool.
        """
        global cost
        cost = ExactCost(l, l_x, l_u, l_xx, l_ux, l_uu, use_multiprocessing)
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
        """ Must be implemented by user.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        raise NotImplementedError


    def l_x(self, x, u, i, terminal=False):
        """ Must be implemented by user.

        Args:
            x: state [state_size]
            u: control [action_size]
            i: timestep
            terminal: True for terminal cost

        Returns:
            l_x at x, u [state_size]
        """
        raise NotImplementedError

    def l_u(self, x, u, i, terminal=False):
        """ Must be implemented by user.

        Args:
            x: state [state_size]
            u: control [action_size]
            i: timestep
            terminal: True for terminal cost

        Returns:
            l_u at x, u [action_size]
        """
        raise NotImplementedError

    def l_xx(self, x, u, i, terminal=False):
        """ Must be implemented by user.

        Args:
            x: state [state_size]
            u: control [action_size]
            i: timestep
            terminal: True for terminal cost

        Returns:
            l_xx at x, u [state_size, state_size]
        """
        raise NotImplementedError

    def l_ux(self, x, u, i, terminal=False):
        """ Must be implemented by user.

        Args:
            x: state [state_size]
            u: control [action_size]
            i: timestep
            terminal: True for terminal cost

        Returns:
            l_u at x, u [action_size, state_size]
        """
        raise NotImplementedError

    def l_uu(self, x, u, i, terminal=False):
        """ Must be implemented by user.

        Args:
            x: state [state_size]
            u: control [action_size]
            i: timestep
            terminal: True for terminal cost

        Returns:
            l_u at x, u [action_size, action_size]
        """
        raise NotImplementedError
