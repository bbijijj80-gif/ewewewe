"""Cart-pole balancing task implemented from scratch (no external RL libs, CPU-only).

The network gets [cart_pos, cart_vel, pole_angle, pole_angular_vel] and outputs
a single value; its sign picks "push left" / "push right". Reward is the
number of steps the pole stays upright before falling or the cart running
off the track.
"""
from __future__ import annotations

import math

import numpy as np

GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = CART_MASS + POLE_MASS
POLE_HALF_LENGTH = 0.5
FORCE_MAG = 10.0
TAU = 0.02  # seconds per simulation step

ANGLE_LIMIT = 12 * 2 * math.pi / 360
POS_LIMIT = 2.4
MAX_STEPS = 500


def _step(state: np.ndarray, action: int) -> np.ndarray:
    x, x_dot, theta, theta_dot = state
    force = FORCE_MAG if action == 1 else -FORCE_MAG
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    temp = (force + POLE_MASS * POLE_HALF_LENGTH * theta_dot ** 2 * sin_t) / TOTAL_MASS
    theta_acc = (GRAVITY * sin_t - cos_t * temp) / (
        POLE_HALF_LENGTH * (4.0 / 3.0 - POLE_MASS * cos_t ** 2 / TOTAL_MASS)
    )
    x_acc = temp - POLE_MASS * POLE_HALF_LENGTH * theta_acc * cos_t / TOTAL_MASS

    x = x + TAU * x_dot
    x_dot = x_dot + TAU * x_acc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * theta_acc
    return np.array([x, x_dot, theta, theta_dot])


def run_episode(network, seed: int, max_steps: int = MAX_STEPS) -> float:
    """Run one episode and return the reward (steps survived)."""
    rng = np.random.default_rng(seed)
    state = rng.uniform(-0.05, 0.05, size=4)
    reward = 0.0
    for _ in range(max_steps):
        out = network.forward(state)
        action = 1 if out[0] > 0 else 0
        state = _step(state, action)
        x, _, theta, _ = state
        if abs(x) > POS_LIMIT or abs(theta) > ANGLE_LIMIT:
            break
        reward += 1.0
    return reward


def evaluate(network, seeds: list[int]) -> float:
    """Average reward across a fixed set of seeds (fair, reproducible comparison)."""
    return float(np.mean([run_episode(network, s) for s in seeds]))
