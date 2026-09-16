"""Snake game implemented from scratch (no external game libs), CPU-only.

The network sees 11 hand-crafted features (danger ahead/left/right, current
direction, food direction) and outputs 3 values; argmax picks:
turn-left / go-straight / turn-right (relative to current heading).
Fitness rewards eating food far more than just surviving, so the agent is
pushed to actually hunt food instead of looping in circles.
"""
from __future__ import annotations

import numpy as np

GRID_SIZE = 12
# clockwise order: UP, RIGHT, DOWN, LEFT
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
N_INPUTS = 11
N_OUTPUTS = 3  # turn left / straight / turn right


class SnakeGame:
    def __init__(self, grid_size: int = GRID_SIZE, seed: int = 0):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        mid = grid_size // 2
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        self.dir_idx = 1  # moving RIGHT
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.game_over = False
        self.food = self._place_food()

    def _place_food(self):
        occupied = set(self.snake)
        free = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]
        if not free:
            return self.snake[0]
        idx = self.rng.integers(0, len(free))
        return free[idx]

    def get_state(self) -> np.ndarray:
        head = self.snake[0]

        def blocked(idx):
            dx, dy = DIRECTIONS[idx % 4]
            nx, ny = head[0] + dx, head[1] + dy
            if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                return 1.0
            if (nx, ny) in self.snake[:-1]:
                return 1.0
            return 0.0

        danger_straight = blocked(self.dir_idx)
        danger_right = blocked(self.dir_idx + 1)
        danger_left = blocked(self.dir_idx - 1)

        moving_up = float(self.dir_idx == 0)
        moving_right = float(self.dir_idx == 1)
        moving_down = float(self.dir_idx == 2)
        moving_left = float(self.dir_idx == 3)

        food_left = float(self.food[0] < head[0])
        food_right = float(self.food[0] > head[0])
        food_up = float(self.food[1] < head[1])
        food_down = float(self.food[1] > head[1])

        return np.array([
            danger_straight, danger_right, danger_left,
            moving_up, moving_right, moving_down, moving_left,
            food_left, food_right, food_up, food_down,
        ])

    def step(self, action: int) -> None:
        if action == 0:
            self.dir_idx = (self.dir_idx - 1) % 4
        elif action == 2:
            self.dir_idx = (self.dir_idx + 1) % 4
        # action == 1 -> keep going straight

        dx, dy = DIRECTIONS[self.dir_idx]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)

        self.steps += 1
        self.steps_since_food += 1

        out_of_bounds = not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size)
        hits_self = new_head in self.snake[:-1]
        if out_of_bounds or hits_self:
            self.game_over = True
            return

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            self.food = self._place_food()
        else:
            self.snake.pop()

        # prevent an agent from looping forever without making progress
        if self.steps_since_food > self.grid_size * self.grid_size * 3:
            self.game_over = True


def run_episode(network, seed: int, grid_size: int = GRID_SIZE, max_steps: int = 2000) -> float:
    game = SnakeGame(grid_size, seed)
    while not game.game_over and game.steps < max_steps:
        out = network.forward(game.get_state())
        action = int(np.argmax(out))
        game.step(action)
    # eating food matters far more than merely surviving
    return game.score * 1000.0 + min(game.steps, 500)


def evaluate(network, seeds: list[int]) -> float:
    return float(np.mean([run_episode(network, s) for s in seeds]))
