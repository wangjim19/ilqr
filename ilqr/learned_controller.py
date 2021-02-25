# Copyright (C) 2018, Anass Al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""Controllers."""

import six
import abc
import warnings
import numpy as np
import gtimer as gt
import pdb
import os
from ilqr.mujoco_dynamics import MujocoDynamics

@six.add_metaclass(abc.ABCMeta)
class BaseController():

    """Base trajectory optimizer controller."""

    @abc.abstractmethod
    def fit(self, x0, us_init, *args, **kwargs):
        """Computes the optimal controls.

        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [N, action_size].
            *args, **kwargs: Additional positional and key-word arguments.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        raise NotImplementedError


class iLQR(BaseController):

    """Finite Horizon Iterative Linear Quadratic Regulator."""

    def __init__(self, dynamics, cost, N, max_reg=1e10):
        """Constructs an iLQR solver.

        Args:
            dynamics: Plant dynamics.
            cost: Cost function.
            N: Horizon length.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
        """
        self.dynamics = dynamics
        self.cost = cost
        self.N = N

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._k = np.zeros((N, dynamics.action_size))
        self._K = np.zeros((N, dynamics.action_size, dynamics.state_size))


        super(iLQR, self).__init__()

    def fit(self, x0, us_init, n_iterations=100, tol=1e-6, on_iteration=None, rollout_history = None):
        """Computes the optimal controls.

        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration with the
                following signature:
                (iteration_count, x, J_opt, accepted, converged) -> None
                where:
                    iteration_count: Current iteration count.
                    xs: Current state path.
                    us: Current action path.
                    J_opt: Optimal cost-to-go.
                    accepted: Whether this iteration yielded an accepted result.
                    converged: Whether this iteration converged successfully.
                Default: None.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        gt.stamp('fit/pre', unique=False)

        # Reset regularization term.
        self._mu = 1.0
        self._delta = self._delta_0

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 1.1**(-np.arange(10)**2)

        us = us_init.copy()
        k = self._k
        K = self._K

        N = us.shape[0]
        xs = np.empty((N + 1, self.dynamics.state_size))
        xs[0] = x0
        self.dynamics.set_state(x0)
        for i in range(N):
            xs[i+1] = self.dynamics.step(us[i])
        gt.stamp('fit/rollout', unique=False)

        changed = True
        converged = False
        for iteration in range(n_iterations):
            accepted = False

            # Compute derivatives only if it needs to be recomputed.
            if changed:
                (F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu) = self._compute_derivs(xs, us)
                J_opt = L.sum()
                changed = False
                gt.stamp('fit/derivs', unique=False)

            try:
                # Backward pass.
                k, K = self._backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu)
                gt.stamp('fit/backward', unique=False)

                # Backtracking line search.
                xs_new_all, us_new_all = self._control(xs, us, k, K, alphas)

                for i, alpha in enumerate(alphas):
                    xs_new, us_new = xs_new_all[i], us_new_all[i]
                    J_new = self._trajectory_cost(xs_new, us_new)

                    if J_new < J_opt:
                        if np.abs((J_opt - J_new) / J_opt) < tol:
                            converged = True

                        J_opt = J_new
                        xs = xs_new
                        us = us_new
                        changed = True

                        # Decrease regularization term.
                        self._delta = min(1.0, self._delta) / self._delta_0
                        self._mu *= self._delta
                        if self._mu <= self._mu_min:
                            self._mu = 0.0

                        # Accept this.
                        accepted = True
                        break
                gt.stamp('fit/bls', unique=False)
            except np.linalg.LinAlgError as e:
                # Quu was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))

            if not accepted:
                # Increase regularization term.
                self._delta = max(1.0, self._delta) * self._delta_0
                self._mu = max(self._mu_min, self._mu * self._delta)
                if self._mu_max and self._mu >= self._mu_max:
                    warnings.warn("exceeded max regularization term")
                    break

            if on_iteration:
                on_iteration(iteration, xs, us, J_opt, accepted, converged, rollout_history = rollout_history)

            if converged:
                break

        # Store fit parameters.
        self._k = k
        self._K = K
        self._nominal_xs = xs
        self._nominal_us = us

        return xs, us

    def _control(self, xs, us, k, K, alphas):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path [N+1, state_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            alphas: Line search coefficients.

        Returns:
            Tuple of
                xs: state paths [len(alphas), N+1, state_size].
                us: control paths [len(alphas), N, action_size].
        """
        xs_new = np.zeros(alphas.shape + xs.shape)
        us_new = np.zeros(alphas.shape + us.shape)
        xs_new[:, 0, :] = np.tile(xs[0], alphas.shape + (1,))
        self.dynamics.set_state(xs[0])

        for i in range(self.N):
            # Eq (12).
            us_new[:, i, :] = us[i] + np.outer(alphas, k[i]) + (xs_new[:, i, :] - xs[i]).dot(K[i].T)

            # Eq (8c).
            xs_new[:, i + 1, :] = self.dynamics.f(xs_new[:, i, :], us_new[:, i, :])
        return xs_new, us_new

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N+1, state_size].
            us: Control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us,
                                                     range(self.N)))
        return sum(J) + self.cost.l(xs[-1], None, self.N, terminal=True)

    def _compute_derivs(self, xs, us):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            x0: Initial state [state_size].
            us: Control path [N, action_size].

        Returns:
            Tuple of:
                xs: State path [N+1, state_size].
                F_x: Jacobian of state path w.r.t. x
                    [N, state_size, state_size].
                F_u: Jacobian of state path w.r.t. u
                    [N, state_size, action_size].
                L: Cost path [N+1].
                L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
                L_u: Jacobian of cost path w.r.t. u [N, action_size].
                L_xx: Hessian of cost path w.r.t. x, x
                    [N+1, state_size, state_size].
                L_ux: Hessian of cost path w.r.t. u, x
                    [N, action_size, state_size].
                L_uu: Hessian of cost path w.r.t. u, u
                    [N, action_size, action_size].
                F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                    [N, state_size, state_size, state_size].
                F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                    [N, state_size, action_size, state_size].
                F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                    [N, state_size, action_size, action_size].
        """
        state_size = self.dynamics.state_size
        action_size = self.dynamics.action_size
        N = us.shape[0]

        F_x, F_u = self.dynamics.f_derivs(xs, us)
        gt.stamp('derivs/dyn', unique=False)

        F_x = np.stack(F_x)
        F_u = np.stack(F_u)

        L, L_x, L_u, L_xx, L_ux, L_uu = self.cost.l_derivs(xs, us)
        gt.stamp('derivs/cost', unique=False)

        x = xs[-1]
        L.append(self.cost.l(x, None, N, terminal=True))
        L_x.append(self.cost.l_x(x, None, N, terminal=True))
        L_xx.append(self.cost.l_xx(x, None, N, terminal=True))

        L = np.stack(L)
        L_x = np.stack(L_x)
        L_u = np.stack(L_u)
        L_xx = np.stack(L_xx)
        L_ux = np.stack(L_ux)
        L_uu = np.stack(L_uu)
        gt.stamp('derivs/misc', unique=False)


        return F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu

    def _backward_pass(self,
                       F_x,
                       F_u,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu):
        """Computes the feedforward and feedback gains k and K.

        Args:
            F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
            F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
            L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
            L_u: Jacobian of cost path w.r.t. u [N, action_size].
            L_xx: Hessian of cost path w.r.t. x, x
                [N+1, state_size, state_size].
            L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
            L_uu: Hessian of cost path w.r.t. u, u
                [N, action_size, action_size].
            F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                [N, state_size, state_size, state_size].
            F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                [N, state_size, action_size, state_size].
            F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                [N, state_size, action_size, action_size].

        Returns:
            Tuple of
                k: feedforward gains [N, action_size].
                K: feedback gains [N, action_size, state_size].
        """
        V_x = L_x[-1]
        V_xx = L_xx[-1]

        k = np.empty_like(self._k)
        K = np.empty_like(self._K)

        for i in range(self.N - 1, -1, -1):
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                 L_u[i], L_xx[i], L_ux[i],
                                                 L_uu[i], V_x, V_xx)

            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return np.array(k), np.array(K)

    def _Q(self,
           f_x,
           f_u,
           l_x,
           l_u,
           l_xx,
           l_ux,
           l_uu,
           V_x,
           V_xx):
        """Computes second order expansion.

        Args:
            F_x: Jacobian of state w.r.t. x [state_size, state_size].
            F_u: Jacobian of state w.r.t. u [state_size, action_size].
            L_x: Jacobian of cost w.r.t. x [state_size].
            L_u: Jacobian of cost w.r.t. u [action_size].
            L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
            L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
            L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
            V_x: Jacobian of the value function at the next time step
                [state_size].
            V_xx: Hessian of the value function at the next time step w.r.t.
                x, x [state_size, state_size].
            F_xx: Hessian of state w.r.t. x, x if Hessians are used
                [state_size, state_size, state_size].
            F_ux: Hessian of state w.r.t. u, x if Hessians are used
                [state_size, action_size, state_size].
            F_uu: Hessian of state w.r.t. u, u if Hessians are used
                [state_size, action_size, action_size].

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu


class RecedingHorizonController(object):

    """Receding horizon controller for Model Predictive Control."""

    def __init__(self, controller, env, x0 = None):
        """Constructs a RecedingHorizonController.

        Args:
            x0: Initial state [state_size].
            controller: Controller to fit with.
        """
        self._controller = controller
        self._random = np.random.RandomState()
        self.env = env
        if x0 is None:
            self._x = self.env.reset()
        else:
            self.env.set_state(
                np.concatenate([[0], x0[1:self.env.model.nq]]),
                x0[self.env.model.nq:]
            )
            self._x = self.env._get_obs()

    def seed(self, seed):
        self._random.seed(seed)

    def control(self,
                us_init,
                path_length,
                step_size=1,
                initial_n_iterations=100,
                subsequent_n_iterations=1,
                *args,
                **kwargs):
        """Yields the optimal controls to run at every step as a receding
        horizon problem.

        Note: The first iteration will be slow, but the successive ones will be
        significantly faster.

        Note: This will automatically move the current controller's state to
        what the dynamics model believes will be the next state after applying
        the entire control path computed. Should you want to correct this state
        between iterations, simply use the `set_state()` method.

        Note: If your cost or dynamics are time dependent, then you might need
        to shift their internal state accordingly.

        Args:
            us_init: Initial control path [N, action_size].
            step_size: Number of steps between each controller fit. Default: 1.
                i.e. re-fit at every time step. You might need to increase this
                depending on how powerful your machine is in order to run this
                in real-time.
            initial_n_iterations: Initial max number of iterations to fit.
                Default: 100.
            subsequent_n_iterations: Subsequent max number of iterations to
                fit. Default: 1.
            *args, **kwargs: Additional positional and key-word arguments to
                pass to `controller.fit()`.

        Yields:
            Tuple of
                xs: optimal state path [step_size+1, state_size].
                us: optimal control path [step_size, action_size].
        """
        action_size = self._controller.dynamics.action_size
        n_iterations = initial_n_iterations

        trajectory = [self._x.copy()]
        controls = []
        video_frames = [self.render()]
        predicted_trajs = []
        planned_actions = []

        for i in range(path_length):
        # for i in gt.timed_for(range(path_length)):
            xs, us = self._controller.fit(self._x,
                                          us_init,
                                          n_iterations=n_iterations,
                                          *args,
                                          **kwargs)
            predicted_trajs.append(xs.copy())
            planned_actions.append(us.copy())
            for i in range(step_size):
                ob, reward, done, info = self.env.step(us[i])
                self._x = ob
            video_frames.append(self.render())
            # yield xs[:step_size + 1], us[:step_size], us

            # Set up next action path seed by simply moving along the current
            # optimal path and appending random unoptimal values at the end.
            us_start = us[step_size:]
            us_end = us[-step_size:]
            us_init = np.vstack([us_start, us_end])
            n_iterations = subsequent_n_iterations

            trajectory.append(self._x.copy())
            controls.append(us[:step_size])

        return np.stack(trajectory, axis=0), np.concatenate(controls, axis=0), video_frames, np.stack(predicted_trajs), np.stack(planned_actions)

    def render(self, dim=512):
        ## rendered img is upside-down by default
        img = self.env.sim.render(dim, dim)
        return img[::-1]