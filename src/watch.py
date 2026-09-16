"""Real-time visualization: watch the current best snake AI play.

Run this in a separate window WHILE evolve.py is training (or after it
finishes) — it reloads the checkpoint before every new game, so as soon as
evolve.py saves an improved version, this window starts showing the new,
better behaviour.

    python watch.py --checkpoint checkpoints/best.npz
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pygame

from network import Network
from snake_env import SnakeGame, GRID_SIZE, DIRECTIONS

CELL = 32
MARGIN = 2
BG_COLOR = (20, 20, 24)
GRID_COLOR = (35, 35, 42)
SNAKE_HEAD_COLOR = (90, 220, 120)
SNAKE_BODY_COLOR = (60, 170, 90)
FOOD_COLOR = (230, 80, 80)
TEXT_COLOR = (230, 230, 230)


def load_best(checkpoint_path: str, layer_sizes, rng):
    if os.path.exists(checkpoint_path):
        try:
            return Network.load(checkpoint_path)
        except Exception:
            pass
    return Network(layer_sizes, rng=rng)


def draw(screen, font, game: SnakeGame, generation_info: str):
    screen.fill(BG_COLOR)
    size = game.grid_size

    for x in range(size):
        for y in range(size):
            rect = (x * CELL, y * CELL, CELL - MARGIN, CELL - MARGIN)
            pygame.draw.rect(screen, GRID_COLOR, rect)

    fx, fy = game.food
    pygame.draw.rect(screen, FOOD_COLOR, (fx * CELL, fy * CELL, CELL - MARGIN, CELL - MARGIN))

    for i, (x, y) in enumerate(game.snake):
        color = SNAKE_HEAD_COLOR if i == 0 else SNAKE_BODY_COLOR
        pygame.draw.rect(screen, color, (x * CELL, y * CELL, CELL - MARGIN, CELL - MARGIN))

    hud = f"score: {game.score}   steps: {game.steps}   {generation_info}"
    text = font.render(hud, True, TEXT_COLOR)
    screen.blit(text, (8, size * CELL + 8))

    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="Watch the current best snake AI play, live")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.npz")
    parser.add_argument("--fps", type=int, default=12, help="game speed (steps per second)")
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    args = parser.parse_args()

    layer_sizes = [11, 16, 3]
    rng = np.random.default_rng()

    pygame.init()
    width, height = args.grid_size * CELL, args.grid_size * CELL + 32
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Self-improving AI plays Snake")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    episode = 0
    last_mtime = None
    network = load_best(args.checkpoint, layer_sizes, rng)

    running = True
    while running:
        # reload the checkpoint at the start of each game so improvements
        # made by evolve.py (running separately) show up immediately
        if os.path.exists(args.checkpoint):
            mtime = os.path.getmtime(args.checkpoint)
            if mtime != last_mtime:
                network = load_best(args.checkpoint, layer_sizes, rng)
                last_mtime = mtime

        episode += 1
        game = SnakeGame(grid_size=args.grid_size, seed=int(time.time() * 1000) % 1_000_000)

        while not game.game_over and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            out = network.forward(game.get_state())
            action = int(np.argmax(out))
            game.step(action)

            draw(screen, font, game, f"episode {episode}  |  checkpoint: {args.checkpoint}")
            clock.tick(args.fps)

        # brief pause between games so the final position is visible
        for _ in range(int(args.fps)):
            if not running:
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
