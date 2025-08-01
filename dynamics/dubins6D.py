# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for Dubins vehicle pursuit-evasion dynamics.

This file implements a class for Dubins vehicle dynamics in a pursuit-evasion game.
The state is represented by [x_e, y_e, theta_e, x_p, y_p, theta_p], where 
(x_e, y_e, theta_e) is the evader's position and heading, and 
(x_p, y_p, theta_p) is the pursuer's position and heading.
The control is [omega_e] (evader's turn rate), and the disturbance is [omega_p] (pursuer's turn rate).
"""

from typing import Tuple, Any, Dict, Optional
import numpy as np
from functools import partial
from jax import Array  # modern JAX array type
import jax
from jax import numpy as jnp

from .base_dstb_dynamics import BaseDstbDynamics


class Dubins6D(BaseDstbDynamics):

  def __init__(self, cfg: Any, action_space: Dict[str, np.ndarray]) -> None:
    """
    Implements the Dubins pursuit-evasion dynamics.

    Args:
        cfg (Any): an object specifies configuration.
        action_space (Dict[str, np.ndarray]): action space with 'ctrl' and 'dstb' keys.
    """
    super().__init__(cfg, action_space)
    self.dim_x = 6  # [x_e, y_e, theta_e, x_p, y_p, theta_p]

    # Load parameters
    self.evader_velocity: float = getattr(cfg, 'evader_velocity', 0.6)
    self.pursuer_velocity: float = getattr(cfg, 'pursuer_velocity', 0.6)
    self.omega_e_max: float = getattr(cfg, 'omega_e_max', 2.0)
    self.omega_p_max: float = getattr(cfg, 'omega_p_max', 2.0)
    self.goalR: float = getattr(cfg, 'goalR', 0.25)  # collision radius
    self.state_max: float = getattr(cfg, 'state_max', 2.0)
    self.dim_u_dstb: int = 1  # dimension of disturbance
    self.dim_u_ctrl: int = 1  # dimension of control

  def integrate_forward(
      self, state: np.ndarray, control: np.ndarray,
      noise: Optional[np.ndarray] = None, noise_type: Optional[str] = 'unif',
      adversary: Optional[np.ndarray] = None, **kwargs
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Override the base method to handle disturbance correctly for Dubins6D.
    """
    if adversary is not None:
      assert adversary.shape[0] == self.dim_u_dstb, ("Adversary dim. is incorrect!")
      disturbance = adversary
    elif noise is not None:
      assert noise.shape[0] == self.dim_u_dstb, ("Noise dim. is incorrect!")
      # For Dubins6D, disturbance is 1D (pursuer control)
      if noise_type == 'unif':
        rv = (np.random.rand(self.dim_u_dstb) - 0.5) * 2  # Maps to [-1, 1]
      else:
        rv = np.random.normal(size=(self.dim_u_dstb))
      disturbance = noise * rv
    else:
      disturbance = np.zeros(self.dim_u_dstb)
    
    state_nxt, ctrl_clip, dstb_clip = self.integrate_forward_jax(
        jnp.array(state), jnp.array(control), jnp.array(disturbance)
    )
    return np.array(state_nxt), np.array(ctrl_clip), np.array(dstb_clip)


  @partial(jax.jit, static_argnames='self')
  def integrate_forward_jax(
      self, state: Array, control: Array, disturbance: Array
  ) -> Tuple[Array, Array, Array]:
    """Clips the control and disturbance and computes one-step time evolution
    of the system.

    Args:
        state (Array): [x_e, y_e, theta_e, x_p, y_p, theta_p].
        control (Array): [omega_e] (evader's turn rate).
        disturbance (Array): [omega_p] (pursuer's turn rate).

    Returns:
        Array: next state.
        Array: clipped control.
        Array: clipped disturbance.
    """
    # Clips the controller and disturbance values
    ctrl_clip = jnp.clip(control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])
    dstb_clip = jnp.clip(
        disturbance, self.dstb_space[:, 0], self.dstb_space[:, 1]
    )

    # Compute next state using RK4 integration
    state_nxt = self._integrate_forward(state, ctrl_clip, dstb_clip)
    
    # Wrap angles to [-pi, pi]
    state_nxt = state_nxt.at[2].set(
        jnp.mod(state_nxt[2] + jnp.pi, 2 * jnp.pi) - jnp.pi
    )
    state_nxt = state_nxt.at[5].set(
        jnp.mod(state_nxt[5] + jnp.pi, 2 * jnp.pi) - jnp.pi
    )
    
    return state_nxt, ctrl_clip, dstb_clip

  @partial(jax.jit, static_argnames='self')
  def disc_deriv(
      self, state: Array, control: Array, disturbance: Array
  ) -> Array:
    """Computes the continuous-time derivatives of the Dubins dynamics.
    
    Args:
        state (Array): [x_e, y_e, theta_e, x_p, y_p, theta_p].
        control (Array): [omega_e] (evader's turn rate).
        disturbance (Array): [omega_p] (pursuer's turn rate).
        
    Returns:
        Array: derivatives [dx_e/dt, dy_e/dt, dtheta_e/dt, dx_p/dt, dy_p/dt, dtheta_p/dt].
    """
    # Extract state components
    xe, ye, theta_e = state[0], state[1], state[2]
    xp, yp, theta_p = state[3], state[4], state[5]
    
    # Extract control and disturbance
    omega_e = control[0]  # evader's turn rate
    omega_p = disturbance[0]  # pursuer's turn rate
    
    # Dubins dynamics
    deriv = jnp.zeros((self.dim_x,))
    deriv = deriv.at[0].set(self.evader_velocity * jnp.cos(theta_e))  # dx_e/dt
    deriv = deriv.at[1].set(self.evader_velocity * jnp.sin(theta_e))  # dy_e/dt
    deriv = deriv.at[2].set(omega_e)  # dtheta_e/dt
    deriv = deriv.at[3].set(self.pursuer_velocity * jnp.cos(theta_p))  # dx_p/dt
    deriv = deriv.at[4].set(self.pursuer_velocity * jnp.sin(theta_p))  # dy_p/dt
    deriv = deriv.at[5].set(omega_p)  # dtheta_p/dt
    
    return deriv

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward(
      self, state: Array, control: Array, disturbance: Array
  ) -> Array:
    """
    Computes one-step time evolution of the system using RK4 integration.
    
    Args:
        state (Array): [x_e, y_e, theta_e, x_p, y_p, theta_p].
        control (Array): [omega_e] (evader's turn rate).
        disturbance (Array): [omega_p] (pursuer's turn rate).

    Returns:
        Array: next state.
    """
    return self._integrate_forward_dt(state, control, disturbance, self.dt)

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward_dt(
      self, state: Array, control: Array, disturbance: Array,
      dt: float
  ) -> Array:
    """RK4 integration for Dubins dynamics.
    
    Args:
        state (Array): current state.
        control (Array): control input.
        disturbance (Array): disturbance input.
        dt (float): time step.
        
    Returns:
        Array: next state.
    """
    k1 = self.disc_deriv(state, control, disturbance)
    k2 = self.disc_deriv(state + k1*dt/2, control, disturbance)
    k3 = self.disc_deriv(state + k2*dt/2, control, disturbance)
    k4 = self.disc_deriv(state + k3*dt, control, disturbance)
    return state + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

  def get_collision_distance(self, state: np.ndarray) -> float:
    """Compute distance between evader and pursuer.
    
    Args:
        state (np.ndarray): [x_e, y_e, theta_e, x_p, y_p, theta_p].
        
    Returns:
        float: distance between evader and pursuer.
    """
    xe, ye = state[0], state[1]
    xp, yp = state[3], state[4]
    return np.sqrt((xe - xp)**2 + (ye - yp)**2)

  def is_collision(self, state: np.ndarray) -> bool:
    """Check if evader and pursuer are in collision.
    
    Args:
        state (np.ndarray): [x_e, y_e, theta_e, x_p, y_p, theta_p].
        
    Returns:
        bool: True if collision, False otherwise.
    """
    return self.get_collision_distance(state) <= self.goalR

  def is_within_bounds(self, state: np.ndarray) -> bool:
    """Check if state is within the defined bounds.
    
    Args:
        state (np.ndarray): [x_e, y_e, theta_e, x_p, y_p, theta_p].
        
    Returns:
        bool: True if within bounds, False otherwise.
    """
    xe, ye = state[0], state[1]
    xp, yp = state[3], state[4]
    
    # Check if both evader and pursuer are within bounds
    evader_in_bounds = (-self.state_max <= xe <= self.state_max and 
                       -self.state_max <= ye <= self.state_max)
    pursuer_in_bounds = (-self.state_max <= xp <= self.state_max and 
                        -self.state_max <= yp <= self.state_max)
    
    return evader_in_bounds and pursuer_in_bounds
