"""Self-improvement loop, CPU/RAM only (numpy), no GPU.

Each generation:
  1. Mutate the current best network -> candidate.
  2. Evaluate BOTH best and candidate on the same fresh seeds (fair test).
  3. If candidate is better -> it REPLACES best (checkpoint saved).
     Otherwise -> candidate is discarded, best is unchanged (rollback).

Progress and the best checkpoint are persisted to disk after every
generation, so the run can be interrupted and resumed at any time.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from network import Network
from snake_env import evaluate, N_INPUTS, N_OUTPUTS

LAYER_SIZES = [N_INPUTS, 16, N_OUTPUTS]  # snake state -> turn-left/straight/turn-right

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "best.npz")
DEFAULT_LOG = os.path.join(SCRIPT_DIR, "checkpoints", "log.csv")


def load_or_init_best(checkpoint_path: str, rng: np.random.Generator) -> Network:
    if os.path.exists(checkpoint_path):
        print(f"[resume] loading existing best from {checkpoint_path}")
        return Network.load(checkpoint_path)
    print("[init] no checkpoint found, starting from a fresh random network")
    return Network(LAYER_SIZES, rng=rng)


def main():
    parser = argparse.ArgumentParser(description="Self-improving CPU-only agent (evolutionary hill-climbing)")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=8, help="seeds per evaluation (fitness = mean reward)")
    parser.add_argument("--sigma", type=float, default=0.3, help="mutation strength (gaussian noise stddev)")
    parser.add_argument("--sigma-decay", type=float, default=0.999, help="multiply sigma by this each generation")
    parser.add_argument("--min-sigma", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--log", type=str, default=DEFAULT_LOG)
    args = parser.parse_args()
    args.checkpoint = os.path.abspath(args.checkpoint)
    args.log = os.path.abspath(args.log)

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)

    best = load_or_init_best(args.checkpoint, rng)
    eval_seeds = [int(s) for s in rng.integers(0, 1_000_000, size=args.episodes)]
    best_fitness = evaluate(best, eval_seeds)

    log_exists = os.path.exists(args.log)
    log_file = open(args.log, "a", newline="")
    writer = csv.writer(log_file)
    if not log_exists:
        writer.writerow(["generation", "best_fitness", "candidate_fitness", "accepted", "sigma", "timestamp"])

    sigma = args.sigma
    print(f"starting fitness: {best_fitness:.1f} (score*1000 + steps survived, up to 500)")

    try:
        for gen in range(1, args.generations + 1):
            # Fresh seeds each generation so the network can't overfit to one scenario,
            # but best and candidate are always compared on the SAME seeds (fair test).
            eval_seeds = [int(s) for s in rng.integers(0, 1_000_000, size=args.episodes)]

            candidate = best.mutate(sigma, rng)

            best_fitness = evaluate(best, eval_seeds)
            candidate_fitness = evaluate(candidate, eval_seeds)

            accepted = candidate_fitness > best_fitness
            if accepted:
                best = candidate  # replace: the improved version becomes the new baseline
                best_fitness = candidate_fitness
                best.save(args.checkpoint)
            # else: rollback -> `best` is simply left as-is, candidate is discarded

            sigma = max(args.min_sigma, sigma * args.sigma_decay)

            writer.writerow([gen, best_fitness, candidate_fitness, accepted, sigma, time.time()])
            log_file.flush()

            if gen % 10 == 0 or accepted:
                status = "IMPROVED" if accepted else "rejected"
                print(f"gen {gen:5d}  best={best_fitness:6.1f}  candidate={candidate_fitness:6.1f}  "
                      f"sigma={sigma:.4f}  [{status}]")

    finally:
        log_file.close()

    print(f"final best fitness: {best_fitness:.1f}, checkpoint saved at {args.checkpoint}")


if __name__ == "__main__":
    main()
