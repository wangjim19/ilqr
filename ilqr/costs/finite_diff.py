import os
import multiprocessing as mp
import numpy as np
from scipy.optimize import approx_fprime

from ilqr.autodiff import as_function, hessian_scalar, jacobian_scalar
from .base import Cost

class FiniteDiffCost(Cost):

    """Finite difference approximated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

    def __init__(self,
                 l,
                 l_terminal,
                 state_size,
                 action_size,
                 x_eps=None,
                 u_eps=None,
                 use_multiprocessing = False):
        """Constructs an FiniteDiffCost.

        Args:
            l: Instantaneous cost function to approximate.
                Signature: (x, u, i) -> scalar.
            l_terminal: Terminal cost function to approximate.
                Signature: (x, i) -> scalar.
            state_size: State size.
            action_size: Action size.
            x_eps: Increment to the state to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
            u_eps: Increment to the action to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).

        Note:
            The square root of the provided epsilons are used when computing
            the Hessians instead.
        """
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
            self._pool = mp.Pool(initializer = FiniteDiffCost._worker_init,
                                 initargs = (l, l_terminal, state_size, action_size, x_eps, u_eps, False))

        super(FiniteDiffCost, self).__init__()

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
        cost = FiniteDiffCost(l, l_terminal, state_size, action_size, x_eps, u_eps, use_multiprocessing)
        print("Finished loading process", os.getpid())

    @staticmethod
    def _worker(x, u, i):
        return (cost.l(x, u, i), cost.l_x(x, u, i), cost.l_u(x, u, i), cost.l_xx(x, u, i), cost.l_ux(x, u, i), cost.l_uu(x, u, i))

    def l_derivs(self, xs, us):
        if self.multiprocessing:
            results = self._pool.starmap(FiniteDiffCost._worker, [(xs[i], us[i], i) for i in range(us.shape[0])], chunksize = us.shape[0] // mp.cpu_count())
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
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            return approx_fprime(x, lambda x: self._l_terminal(x, i),
                                 self._x_eps)

        return approx_fprime(x, lambda x: self._l(x, u, i), self._x_eps)

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros(self._action_size)

        return approx_fprime(u, lambda u: self._l(x, u, i), self._u_eps)

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        eps = self._x_eps_hess

        Q = []
        center = self.l_x(x, u, i, terminal)
        for j in range(self._state_size):
            x[j] += eps
            deriv = (self.l_x(x, u, i, terminal) - center) / eps
            x[j] -= eps
            Q.append(deriv)
        Q = np.column_stack(Q)
        return Q

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        eps = self._x_eps_hess

        Q = []
        center = self.l_u(x, u, i, terminal)
        for j in range(self._state_size):
            x[j] += eps
            deriv = (self.l_u(x, u, i, terminal) - center) / eps
            x[j] -= eps
            Q.append(deriv)
        Q = np.column_stack(Q)
        return Q

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        eps = self._u_eps_hess

        Q = []
        center = self.l_u(x, u, i, terminal)
        for j in range(self._action_size):
            u[j] += eps
            deriv = (self.l_u(x, u, i, terminal) - center) / eps
            u[j] -= eps
            Q.append(deriv)
        Q = np.column_stack(Q)
        return Q