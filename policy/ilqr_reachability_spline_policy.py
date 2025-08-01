# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Tuple, Optional, Dict
import time
import numpy as np
import jax
from jax import numpy as jnp
from jax import Array
from functools import partial

from .ilqr_spline_policy import ILQRSpline


class ILQRReachabilitySpline(ILQRSpline):

  def get_action(
      self, obs: np.ndarray, controls: Optional[np.ndarray] = None,
      agents_action: Optional[Dict] = None, **kwargs
  ) -> np.ndarray:
    status = 0

    # `controls` include control input at timestep N-1, which is a dummy
    # control of zeros.
    if controls is None:
      controls = jnp.zeros((self.dim_u, self.plan_horizon))
    else:
      assert controls.shape[1] == self.plan_horizon
      controls = jnp.array(controls)

    # Rolls out the nominal trajectory and gets the initial cost.
    # * This is differnet from the naive ILQR as it relies on the information
    # * from the pyspline.
    states, controls = self.rollout_nominal(
        jnp.array(kwargs.get('state')), controls
    )
    closest_pt, slope, theta = self.track.get_closest_pts(
        np.asarray(states[:2, :])
    )
    closest_pt = jnp.array(closest_pt)
    slope = jnp.array(slope)
    theta = jnp.array(theta)
    # J = self.cost.get_traj_cost(states, controls, closest_pt, slope, theta)
    state_costs = self.cost.constraint.get_cost(
        states, controls, closest_pt, slope, theta
    )
    ctrl_costs = self.cost.ctrl_cost.get_cost(states, controls)
    critical, fut_cost = self.get_critical_points(state_costs)
    J = (fut_cost + jnp.sum(ctrl_costs)).astype(float)

    converged = False
    time0 = time.time()
    for i in range(self.max_iter):
      # We need cost derivatives from 0 to N-1, but we only need dynamics
      # jacobian from 0 to N-2.
      c_x, c_u, c_xx, c_uu, c_ux = self.cost.get_derivatives(
          states, controls, closest_pt, slope, theta
      )
      fx, fu = self.dyn.get_jacobian(states[:, :-1], controls[:, :-1])
      K_closed_loop, k_open_loop = self.backward_pass(
          c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu,
          critical=critical
      )
      updated = False
      for alpha in self.alphas:
        (
            X_new, U_new, J_new, closest_pt_new, slope_new, theta_new,
            critical_new, state_costs_new
        ) = (
            self.forward_pass(
                states, controls, K_closed_loop, k_open_loop, alpha
            )
        )
        # critical_imp = jnp.max(
        #     state_costs_new[critical] - state_costs[critical]
        # )
        critical_imp = (
            jnp.max(state_costs_new[critical_new])
            - jnp.max(state_costs[critical])
        )
        if critical_imp < 0.:
          # if J_new <= J:  # Improved!
          if np.abs((J-J_new) / J) < self.tol:  # Small improvement.
            converged = True

          # Updates nominal trajectory and best cost.
          J = J_new
          states = X_new
          controls = U_new
          closest_pt = closest_pt_new
          slope = slope_new
          theta = theta_new
          critical = critical_new
          state_costs = state_costs_new
          updated = True
          break

      # Terminates early if there is no update within alphas.
      if not updated:
        status = 2
        break

      # Terminates early if the objective improvement is negligible.
      if converged:
        status = 1
        break
    t_process = time.time() - time0

    states = np.asarray(states)
    controls = np.asarray(controls)
    K_closed_loop = np.asarray(K_closed_loop)
    k_open_loop = np.asarray(k_open_loop)
    solver_info = dict(
        states=states, controls=controls, K_closed_loop=K_closed_loop,
        k_open_loop=k_open_loop, t_process=t_process, status=status, J=J
    )
    return controls[:, 0], solver_info

  @partial(jax.jit, static_argnames='self')
  def get_critical_points(
      self, state_costs: Array
  ) -> Tuple[Array, Array]:

    @jax.jit
    def true_func(args):
      idx, critical, cur_cost, fut_cost = args
      critical = critical.at[idx].set(True)
      return critical, cur_cost

    @jax.jit
    def false_func(args):
      idx, critical, cur_cost, fut_cost = args
      return critical, fut_cost

    @jax.jit
    def critical_pt(i, _carry):
      idx = self.plan_horizon - 1 - i
      critical, fut_cost = _carry
      critical, fut_cost = jax.lax.cond(
          state_costs[idx] > fut_cost, true_func, false_func,
          (idx, critical, state_costs[idx], fut_cost)
      )
      return critical, fut_cost

    critical = jnp.zeros(shape=(self.plan_horizon,), dtype=bool)
    critical = critical.at[self.plan_horizon - 1].set(True)
    critical, fut_cost = jax.lax.fori_loop(
        1, self.plan_horizon - 1, critical_pt, (critical, state_costs[-1])
    )  # backward until timestep 1
    return critical, fut_cost

  def forward_pass(
      self, nominal_states: Array, nominal_controls: Array,
      K_closed_loop: Array, k_open_loop: Array, alpha: float
  ) -> Tuple[Array, Array, float, Array, Array, Array, Array]:
    X, U = self.rollout(
        nominal_states, nominal_controls, K_closed_loop, k_open_loop, alpha
    )
    # * This is differnet from the naive ILQR as it relies on the information
    # * from the pyspline.
    closest_pt, slope, theta = self.track.get_closest_pts(np.asarray(X[:2, :]))
    closest_pt = jnp.array(closest_pt)
    slope = jnp.array(slope)
    theta = jnp.array(theta)

    # J = self.cost.get_traj_cost(X, U, closest_pt, slope, theta)
    # ! hacky
    state_costs = self.cost.constraint.get_cost(X, U, closest_pt, slope, theta)
    ctrl_costs = self.cost.ctrl_cost.get_cost(X, U)

    critical, fut_cost = self.get_critical_points(state_costs)
    J = (fut_cost + jnp.sum(ctrl_costs)).astype(float)
    return X, U, J, closest_pt, slope, theta, critical, state_costs

  @partial(jax.jit, static_argnames='self')
  def backward_pass(
      self, c_x: Array, c_u: Array, c_xx: Array,
      c_uu: Array, c_ux: Array, fx: Array, fu: Array,
      critical: Array
  ) -> Tuple[Array, Array]:
    """
    Jitted backward pass looped computation.

    Args:
        c_x (Array): (dim_x, N)
        c_u (Array): (dim_u, N)
        c_xx (Array): (dim_x, dim_x, N)
        c_uu (Array): (dim_u, dim_u, N)
        c_ux (Array): (dim_u, dim_x, N)
        fx (Array): (dim_x, dim_x, N-1)
        fu (Array): (dim_x, dim_u, N-1)

    Returns:
        Ks (Array): gain matrices (dim_u, dim_x, N - 1)
        ks (Array): gain vectors (dim_u, N - 1)
    """

    @jax.jit
    def true_func(args):
      idx, V_x, V_xx, ks, Ks = args

      # ! Q_x, Q_xx are not used if this time step is critical.
      # Q_x = c_x[:, idx] + fx[:, :, idx].T @ V_x
      # Q_xx = c_xx[:, :, idx] + fx[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_ux = c_ux[:, :, idx] + fu[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_u = c_u[:, idx] + fu[:, :, idx].T @ V_x
      Q_uu = c_uu[:, :, idx] + fu[:, :, idx].T @ V_xx @ fu[:, :, idx]

      Q_uu_inv = jnp.linalg.inv(Q_uu + reg_mat)
      Ks = Ks.at[:, :, idx].set(-Q_uu_inv @ Q_ux)
      ks = ks.at[:, idx].set(-Q_uu_inv @ Q_u)
      return c_x[:, idx], c_xx[:, :, idx], ks, Ks

    @jax.jit
    def false_func(args):
      idx, V_x, V_xx, ks, Ks = args

      Q_x = fx[:, :, idx].T @ V_x
      Q_xx = fx[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_ux = c_ux[:, :, idx] + fu[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_u = c_u[:, idx] + fu[:, :, idx].T @ V_x
      Q_uu = c_uu[:, :, idx] + fu[:, :, idx].T @ V_xx @ fu[:, :, idx]

      Q_uu_inv = jnp.linalg.inv(Q_uu + reg_mat)
      Ks = Ks.at[:, :, idx].set(-Q_uu_inv @ Q_ux)
      ks = ks.at[:, idx].set(-Q_uu_inv @ Q_u)

      V_x = Q_x + Q_ux.T @ ks[:, idx]
      V_xx = Q_xx + Q_ux.T @ Ks[:, :, idx]
      return V_x, V_xx, ks, Ks

    @jax.jit
    def backward_pass_looper(i, _carry):
      V_x, V_xx, ks, Ks, critical = _carry
      idx = self.plan_horizon - 2 - i

      V_x, V_xx, ks, Ks = jax.lax.cond(
          critical[idx], true_func, false_func, (idx, V_x, V_xx, ks, Ks)
      )
      return V_x, V_xx, ks, Ks, critical

    # Initializes.
    Ks = jnp.zeros((self.dim_u, self.dim_x, self.plan_horizon - 1))
    ks = jnp.zeros((self.dim_u, self.plan_horizon - 1))
    V_x = c_x[:, -1]
    V_xx = c_xx[:, :, -1]
    reg_mat = self.eps * jnp.eye(self.dim_u)

    _, _, ks, Ks, _ = jax.lax.fori_loop(
        0, self.plan_horizon - 1, backward_pass_looper,
        (V_x, V_xx, ks, Ks, critical)
    )
    return Ks, ks
