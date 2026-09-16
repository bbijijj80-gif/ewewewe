# Self-improving Snake AI (CPU-only)

A small AI that learns to play Snake by improving itself through
trial-and-error — no GPU required, only `numpy` (and `pygame` for the
live view) running on CPU/RAM.

## How it works

Each generation:

1. **Mutate**: the current best network's weights get a small random
   perturbation, producing a *candidate*.
2. **Test**: both the current best and the candidate play the same set of
   Snake games (same seeds), so the comparison is fair.
3. **Keep or rollback**:
   - If the candidate scores higher → it **replaces** the current best
     (saved to `checkpoints/best.npz`).
   - If not → the candidate is discarded and the previous best is kept
     unchanged (**rollback**).

This repeats for many generations; the network keeps only what makes it
provably better, so its score never regresses.

## Usage

Train (no window, runs in the terminal):

```bash
pip install -r requirements.txt
cd src
python evolve.py --generations 2000
```

Watch it play, live, in its own window — run this at the same time (in
another terminal) or after training. It reloads the checkpoint before every
new game, so as soon as `evolve.py` saves an improved version, the window
starts showing the smarter behaviour:

```bash
cd src
python watch.py
```

Useful `evolve.py` flags:

- `--episodes N` — how many seeds each candidate is tested on per generation
  (more = more reliable "better or not" judgement, slower).
- `--sigma X` — initial mutation strength.
- `--sigma-decay X` — shrinks mutation strength over time (fine-tunes once
  close to a good solution).
- `--checkpoint path.npz` — where the current best network is stored (by
  default `src/checkpoints/best.npz`, regardless of which folder you run
  the script from); delete it to start over, or just re-run to resume
  from it.

Useful `watch.py` flags:

- `--fps N` — game speed (steps per second).
- `--checkpoint path.npz` — which checkpoint to watch. By default it points
  at the same `src/checkpoints/best.npz` that `evolve.py` writes to by
  default, so the two normally need no flags at all. If you pass
  `--checkpoint` to `evolve.py`, pass the *same* path to `watch.py`,
  otherwise it will play with an untrained random network — the window's
  HUD always shows `[TRAINED]` or `[UNTRAINED]` so this is obvious at a
  glance, and the terminal prints exactly which path it loaded.

Progress (including every accepted/rejected mutation) is appended to
`checkpoints/log.csv`.

## Task

The network plays Snake on a 12x12 grid (implemented from scratch — no
external game library). Its input is 11 features: danger straight/left/right
ahead, current direction, and food direction relative to the head. Its
output picks turn-left / go-straight / turn-right. Fitness heavily rewards
eating food (score * 1000) plus a small bonus for steps survived, so the
agent is pushed to actually hunt food rather than just avoid dying.

Swap `src/snake_env.py` for a different task/fitness function to make the
same self-improvement loop optimize something else.
