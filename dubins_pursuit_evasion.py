# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for Dubins pursuit-evasion environment.

This file implements a zero-sum environment for Dubins vehicle pursuit-evasion games.
The evader (controller) tries to avoid collision while staying within bounds,
while the pursuer (disturbance) tries to cause collision.
"""

from typing import Dict, Tuple, Optional, Any, Union
import numpy as np
import torch
from gym import spaces
import matplotlib.pyplot as plt

from .base_zs_env import BaseZeroSumEnv
from .agent import Agent
from .utils import ActionZS


class DubinsPursuitEvasionEnv(BaseZeroSumEnv):
    """
    Implements a zero-sum environment for Dubins vehicle pursuit-evasion games.
    """

    def __init__(self, cfg_env: Any, cfg_agent: Any, cfg_cost: Any) -> None:
        super().__init__(cfg_env, cfg_agent)
        
        # Initialize your dynamics
        if cfg_agent.dyn == "Dubins6D":
            self.agent.dyn = self.agent.dyn  # Already set by parent class
        
        # Set up cost function parameters
        self.cost_params = cfg_cost
        
        # Load parameters from config
        self.goalR = getattr(cfg_cost, 'goalR', 0.25)  # collision radius
        self.state_max = getattr(cfg_cost, 'state_max', 2.0)
        self.set_mode = getattr(cfg_cost, 'set_mode', 'avoid')  # 'avoid', 'reach', or 'reach_avoid'
        
        # Cost weights
        self.q1_collision = getattr(cfg_cost, 'q1_collision', 10.0)
        self.q2_bounds = getattr(cfg_cost, 'q2_bounds', 1.0)
        self.w_control = getattr(cfg_cost, 'w_control', 0.01)
        self.w_disturbance = getattr(cfg_cost, 'w_disturbance', 0.01)
        
        # Set up observation type and reset parameters
        self.obs_type = getattr(cfg_env, 'obs_type', 'perfect')
        self.failure_thr = getattr(cfg_env, 'failure_thr', 0.0)
        self.reset_thr = getattr(cfg_env, 'reset_thr', 0.0)
        self.reset_rej_sampling = getattr(cfg_env, 'reset_rej_sampling', False)
        self.end_criterion = getattr(cfg_env, 'end_criterion', 'failure')
        self.g_x_fail = getattr(cfg_env, 'g_x_fail', 0.1)
        self.timeout = getattr(cfg_env, 'timeout', 300)
        
        # Set up visualization bounds
        self.visual_bounds = np.array([[-self.state_max, self.state_max], [-self.state_max, self.state_max]])
        x_eps = (2 * self.state_max) * 0.005
        y_eps = (2 * self.state_max) * 0.005
        self.visual_extent = np.array([
            self.visual_bounds[0, 0] - x_eps, self.visual_bounds[0, 1] + x_eps,
            self.visual_bounds[1, 0] - y_eps, self.visual_bounds[1, 1] + y_eps
        ])
        
        # Initialize observation and reset spaces
        self.build_obs_rst_space(cfg_env, cfg_agent, cfg_cost)
        self.seed(cfg_env.seed)
        
        # Add track attribute for visualization compatibility
        self.track = self  # Self-reference for track plotting methods
        
        self.reset()

    def get_constraints(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray) -> Dict:
        """Define safety constraints.
        
        Args:
            state: Current state [x_e, y_e, theta_e, x_p, y_p, theta_p]
            action: Dictionary with 'ctrl' and 'dstb' actions
            state_nxt: Next state
            
        Returns:
            Dictionary of constraint values
        """
        constraints = {}
        
        # Extract state components
        xe, ye = state[0], state[1]
        xp, yp = state[3], state[4]
        
        # Collision constraint (positive when safe, negative when colliding)
        collision_dist = np.sqrt((xe - xp)**2 + (ye - yp)**2)
        constraints['collision'] = np.array([[collision_dist - self.goalR]])
        
        # Bounds constraints (positive when within bounds)
        evader_bounds_x = min(xe + self.state_max, self.state_max - xe)
        evader_bounds_y = min(ye + self.state_max, self.state_max - ye)
        pursuer_bounds_x = min(xp + self.state_max, self.state_max - xp)
        pursuer_bounds_y = min(yp + self.state_max, self.state_max - yp)
        
        constraints['evader_bounds'] = np.array([[min(evader_bounds_x, evader_bounds_y)]])
        constraints['pursuer_bounds'] = np.array([[min(pursuer_bounds_x, pursuer_bounds_y)]])
        
        return constraints

    def get_constraints_all(
        self, states: np.ndarray, actions: Union[np.ndarray, dict]
    ) -> Dict[str, np.ndarray]:
        """
        Gets the values of all constraint functions for multiple states.
        
        Args:
            states: Array of states [batch_size, state_dim]
            actions: Array of actions or dict of actions
            
        Returns:
            Dict: each (key, value) pair is the name and values of a constraint
                evaluated at the states and actions input.
        """
        # For simplicity, we'll evaluate constraints for each state individually
        # In a more efficient implementation, you could vectorize this
        batch_size = states.shape[1] if len(states.shape) > 1 else 1
        
        if batch_size == 1:
            # Single state case
            state = states.flatten()
            if isinstance(actions, dict):
                action = actions
            else:
                action = {'ctrl': actions[:1], 'dstb': actions[1:2]}
            
            # Use the existing get_constraints method
            constraints = self.get_constraints(state, action, state)
            
            # Return as is (already in correct format)
            return constraints
        else:
            # Multiple states case - evaluate each one
            constraint_dict = {}
            for i in range(batch_size):
                state = states[:, i]
                if isinstance(actions, dict):
                    action = {k: v[i] if hasattr(v, '__len__') else v for k, v in actions.items()}
                else:
                    action = {'ctrl': actions[i:i+1], 'dstb': actions[i+1:i+2]}
                
                constraints = self.get_constraints(state, action, state)
                
                # Initialize arrays if first iteration
                if i == 0:
                    constraint_dict = {k: np.zeros((1, batch_size)) for k in constraints.keys()}
                
                # Store values
                for k, v in constraints.items():
                    constraint_dict[k][0, i] = v[0, 0]  # Extract scalar value from array
            
            return constraint_dict

    def get_cost(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray, 
                constraints: Optional[Dict] = None) -> float:
        """Define cost function.
        
        Args:
            state: Current state
            action: Dictionary with 'ctrl' and 'dstb' actions
            state_nxt: Next state
            constraints: Constraint values from get_constraints
            
        Returns:
            Cost value (controller wants to minimize, disturbance wants to maximize)
        """
        cost = 0.0
        
        # Extract actions
        ctrl = action['ctrl']
        dstb = action['dstb']
        
        # Control cost (controller wants to minimize control effort)
        cost += self.w_control * np.sum(ctrl**2)
        
        # Disturbance cost (disturbance wants to minimize disturbance effort)
        cost -= self.w_disturbance * np.sum(dstb**2)
        
        # Collision penalty (large penalty for collision)
        if constraints is not None and 'collision' in constraints:
            collision_val = constraints['collision'][0, 0] if isinstance(constraints['collision'], np.ndarray) else constraints['collision']
            if collision_val < 0:  # Collision occurred
                cost += self.q1_collision * abs(collision_val)
        
        # Bounds penalty (penalty for going out of bounds)
        if constraints is not None:
            evader_bounds_val = constraints['evader_bounds'][0, 0] if isinstance(constraints['evader_bounds'], np.ndarray) else constraints['evader_bounds']
            pursuer_bounds_val = constraints['pursuer_bounds'][0, 0] if isinstance(constraints['pursuer_bounds'], np.ndarray) else constraints['pursuer_bounds']
            
            if evader_bounds_val < 0:
                cost += self.q2_bounds * abs(evader_bounds_val)
            if pursuer_bounds_val < 0:
                cost += self.q2_bounds * abs(pursuer_bounds_val)
        
        return cost

    def get_target_margin(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray) -> Dict:
        """Define target margins for reach-avoid problems.
        
        Args:
            state: Current state
            action: Dictionary with 'ctrl' and 'dstb' actions
            state_nxt: Next state
            
        Returns:
            Dictionary of target margins
        """
        xe, ye = state[0], state[1]
        xp, yp = state[3], state[4]
        
        # Distance to collision
        collision_dist = np.sqrt((xe - xp)**2 + (ye - yp)**2)
        
        # Distance to bounds (how close to boundary)
        evader_dist_to_bounds = min(
            self.state_max - abs(xe),
            self.state_max - abs(ye)
        )
        pursuer_dist_to_bounds = min(
            self.state_max - abs(xp),
            self.state_max - abs(yp)
        )
        
        return {
            'collision_distance': collision_dist,
            'evader_bounds_distance': evader_dist_to_bounds,
            'pursuer_bounds_distance': pursuer_dist_to_bounds
        }

    def get_done_and_info(self, state: np.ndarray, constraints: Dict, targets: Dict,
                         final_only: bool = True, end_criterion: Optional[str] = None) -> Tuple[bool, Dict]:
        """Define episode termination conditions.
        
        Args:
            state: Current state
            constraints: Constraint values
            targets: Target margin values
            final_only: Whether to only check final state
            end_criterion: End criterion type
            
        Returns:
            Tuple of (done, info)
        """
        if end_criterion is None:
            end_criterion = self.end_criterion

        done = False
        done_type = "not_raised"
        
        # Check timeout
        if self.cnt >= self.timeout:
            done = True
            done_type = "timeout"
        
        # Get constraint values (handle both scalar and array formats)
        if isinstance(constraints['collision'], np.ndarray):
            g_x = min(constraints['collision'][0, 0], constraints['evader_bounds'][0, 0], constraints['pursuer_bounds'][0, 0])
        else:
            g_x = min(constraints['collision'], constraints['evader_bounds'], constraints['pursuer_bounds'])
        
        # Get target values (l_x) - for Dubins, we use collision distance as target
        if targets is not None:
            l_x = targets['collision_distance'] - self.goalR  # Positive when safe
        else:
            l_x = np.inf
        
        # Binary cost (1 if safe, 0 if unsafe)
        binary_cost = 1.0 if g_x > self.failure_thr else 0.0
        
        # Check for failure
        if end_criterion == 'failure':
            if g_x > self.failure_thr:
                done = True
                done_type = "failure"
                g_x = self.g_x_fail
        elif end_criterion == 'timeout':
            pass  # Already handled above
        
        # Build info dictionary
        info = {
            'g_x': float(g_x),
            'l_x': float(l_x),
            'binary_cost': float(binary_cost),
            'done_type': done_type,
            'termination_reason': done_type
        }
        
        # Add additional info if available
        if targets is not None:
            info['collision_distance'] = targets['collision_distance']
            info['evader_bounds_distance'] = targets['evader_bounds_distance']
            info['pursuer_bounds_distance'] = targets['pursuer_bounds_distance']
        
        return done, info

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        """Define observation space.
        
        Args:
            state: Current state [x_e, y_e, theta_e, x_p, y_p, theta_p]
            
        Returns:
            Observation (can be full state or processed)
        """
        # For now, return the full state
        # You could also return relative coordinates or other processed observations
        return state.copy()

    def reset(self, state: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Reset the environment.
        
        Args:
            state: Optional initial state
            **kwargs: Additional arguments
            
        Returns:
            Initial observation
        """
        super().reset()
        
        if state is None:
            # Use reset space if available, otherwise generate random state
            if hasattr(self, 'reset_sample_sapce'):
                self.state = self.reset_sample_sapce.sample()
            else:
                # Generate random initial state
                self.state = np.array([
                    np.random.uniform(-self.state_max, self.state_max),  # x_e
                    np.random.uniform(-self.state_max, self.state_max),  # y_e
                    np.random.uniform(-np.pi, np.pi),                    # theta_e
                    np.random.uniform(-self.state_max, self.state_max),  # x_p
                    np.random.uniform(-self.state_max, self.state_max),  # y_p
                    np.random.uniform(-np.pi, np.pi)                     # theta_p
                ])
            
            # Ensure initial separation is greater than collision radius
            xe, ye = self.state[0], self.state[1]
            xp, yp = self.state[3], self.state[4]
            initial_dist = np.sqrt((xe - xp)**2 + (ye - yp)**2)
            
            if initial_dist < self.goalR * 4:  # Ensure safe initial separation (increased from 2 to 4)
                # Move pursuer away from evader
                angle = np.arctan2(yp - ye, xp - xe)
                self.state[3] = xe + self.goalR * 4 * np.cos(angle)
                self.state[4] = ye + self.goalR * 4 * np.sin(angle)
        else:
            self.state = state.copy()
        
        self.cnt = 0
        return self.get_obs(self.state)

    def build_obs_rst_space(self, cfg_env, cfg_agent, cfg_cost):
        """Build observation and reset spaces."""
        # Reset Sample Space
        reset_space = np.array(cfg_env.reset_space, dtype=np.float32)
        self.reset_sample_sapce = spaces.Box(
            low=reset_space[:, 0], high=reset_space[:, 1]
        )

        # Observation space
        if self.obs_type == "perfect":
            low = np.zeros((self.state_dim,))
            low[0] = -self.state_max  # x_e
            low[1] = -self.state_max  # y_e
            low[2] = 0.0  # v_e (evader velocity)
            low[3] = -np.pi  # theta_e
            low[4] = -self.state_max  # x_p
            low[5] = -self.state_max  # y_p
            high = np.zeros((self.state_dim,))
            high[0] = self.state_max  # x_e
            high[1] = self.state_max  # y_e
            high[2] = cfg_agent.evader_velocity  # v_e
            high[3] = np.pi  # theta_e
            high[4] = self.state_max  # x_p
            high[5] = self.state_max  # y_p
        else:
            raise ValueError(f"Observation type {self.obs_type} is not supported!")
        
        self.observation_space = spaces.Box(
            low=np.float32(low), high=np.float32(high)
        )
        self.obs_dim = self.observation_space.low.shape[0]

    def seed(self, seed: int = 0):
        """Set random seed."""
        super().seed(seed)
        if hasattr(self, 'reset_sample_sapce'):
            self.reset_sample_sapce.seed(seed)

    def get_samples(self, nx: int, ny: int):
        """Get state samples for value function plotting."""
        xs = np.linspace(self.visual_bounds[0, 0], self.visual_bounds[0, 1], nx)
        ys = np.linspace(self.visual_bounds[1, 0], self.visual_bounds[1, 1], ny)
        return xs, ys

    def render(self):
        """Render the environment (placeholder)."""
        print(f"State: {self.state}")
        print(f"Evader: ({self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f})")
        print(f"Pursuer: ({self.state[3]:.2f}, {self.state[4]:.2f}, {self.state[5]:.2f})")
        print(f"Distance: {np.sqrt((self.state[0]-self.state[3])**2 + (self.state[1]-self.state[4])**2):.2f}")

    def render_obs(self, ax=None, c='r'):
        """Render obstacles (placeholder for Dubins environment)."""
        # Dubins environment doesn't have obstacles, so this is a no-op
        pass

    def plot_track(self, ax, c='k'):
        """Plot track boundaries (placeholder for Dubins environment)."""
        # For Dubins, plot the state space boundaries
        x_min, x_max = self.visual_bounds[0]
        y_min, y_max = self.visual_bounds[1]
        
        # Plot boundary rectangle
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                           linewidth=2, edgecolor=c, facecolor='none')
        ax.add_patch(rect)
        
        # Plot collision radius around current positions
        if hasattr(self, 'state') and self.state is not None:
            xe, ye = self.state[0], self.state[1]
            xp, yp = self.state[3], self.state[4]
            
            # Evader circle
            evader_circle = plt.Circle((xe, ye), self.goalR, color='blue', alpha=0.3)
            ax.add_patch(evader_circle)
            
            # Pursuer circle
            pursuer_circle = plt.Circle((xp, yp), self.goalR, color='red', alpha=0.3)
            ax.add_patch(pursuer_circle)

    def report(self):
        """Report environment information."""
        print("=== Dubins Pursuit-Evasion Environment ===")
        print(f"State dimension: {self.state_dim}")
        print(f"Control dimension: {self.action_dim_ctrl}")
        print(f"Disturbance dimension: {self.action_dim_dstb}")
        print(f"Collision radius: {self.goalR}")
        print(f"State bounds: [-{self.state_max}, {self.state_max}]")
        print(f"Set mode: {self.set_mode}")
        print(f"Timeout: {self.timeout}")
        print("==========================================") 