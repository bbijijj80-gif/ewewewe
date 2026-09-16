# Self-improving agent (CPU-only)

A small AI that improves itself through trial-and-error, no GPU required —
only `numpy` running on CPU/RAM.

## How it works

Each generation:

1. **Mutate**: the current best network's weights get a small random
   perturbation, producing a *candidate*.
2. **Test**: both the current best and the candidate are evaluated on the
   same set of task episodes (balancing a cart-pole), so the comparison is
   fair.
3. **Keep or rollback**:
   - If the candidate scores higher → it **replaces** the current best
     (saved to `checkpoints/best.npz`).
   - If not → the candidate is discarded and the previous best is kept
     unchanged (**rollback**).

This repeats for many generations; the network keeps only what makes it
provably better, so its score never regresses.

## Usage

```bash
pip install -r requirements.txt
cd src
python evolve.py --generations 500
```

Useful flags:

- `--episodes N` — how many seeds each candidate is tested on per generation
  (more = more reliable "better or not" judgement, slower).
- `--sigma X` — initial mutation strength.
- `--sigma-decay X` — shrinks mutation strength over time (fine-tunes once
  close to a good solution).
- `--checkpoint path.npz` — where the current best network is stored; delete
  it to start over, or just re-run to resume from it.

Progress (including every accepted/rejected mutation) is appended to
`checkpoints/log.csv`.

## Task

The network controls a simulated cart-pole (classic control problem,
implemented from scratch — no external RL environment library). Its input
is `[cart position, cart velocity, pole angle, pole angular velocity]` and
its output picks push-left vs push-right. Fitness is the number of
simulation steps (up to 500) the pole stays balanced.

Swap `src/environment.py` for a different task/fitness function to make the
same self-improvement loop optimize something else.
