"""Small feedforward neural network, CPU-only (numpy), no GPU dependencies."""
from __future__ import annotations

import numpy as np


class Network:
    """A tiny feedforward net whose weights are a flat vector.

    Kept flat so mutation, saving and loading are trivial (one array).
    """

    def __init__(self, layer_sizes: list[int], weights: np.ndarray | None = None, rng: np.random.Generator | None = None):
        self.layer_sizes = layer_sizes
        self._shapes = [
            (layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        ]
        self._biases_shapes = [(1, layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        self._sizes = [w[0] * w[1] for w in self._shapes] + [b[1] for b in self._biases_shapes]
        self._total = sum(self._sizes)

        if weights is not None:
            if weights.shape[0] != self._total:
                raise ValueError(f"expected {self._total} weights, got {weights.shape[0]}")
            self.weights = weights.astype(np.float64).copy()
        else:
            rng = rng or np.random.default_rng()
            self.weights = rng.normal(0, 0.5, size=self._total)

    def _unpack(self):
        idx = 0
        mats, biases = [], []
        for shape in self._shapes:
            n = shape[0] * shape[1]
            mats.append(self.weights[idx:idx + n].reshape(shape))
            idx += n
        for shape in self._biases_shapes:
            n = shape[1]
            biases.append(self.weights[idx:idx + n].reshape(shape))
            idx += n
        return mats, biases

    def forward(self, x: np.ndarray) -> np.ndarray:
        mats, biases = self._unpack()
        a = x.reshape(1, -1)
        for i, (w, b) in enumerate(zip(mats, biases)):
            a = a @ w + b
            if i < len(mats) - 1:
                a = np.tanh(a)
        return np.tanh(a).flatten()

    def clone(self) -> "Network":
        return Network(self.layer_sizes, weights=self.weights.copy())

    def mutate(self, sigma: float, rng: np.random.Generator) -> "Network":
        """Return a NEW mutated network; self is left untouched (needed for rollback)."""
        noise = rng.normal(0, sigma, size=self.weights.shape)
        return Network(self.layer_sizes, weights=self.weights + noise)

    def save(self, path: str) -> None:
        np.savez(path, weights=self.weights, layer_sizes=np.array(self.layer_sizes))

    @classmethod
    def load(cls, path: str) -> "Network":
        data = np.load(path)
        return cls(list(data["layer_sizes"]), weights=data["weights"])
