"""MetricGAN+-style PESQ discriminator (Fu et al. 2021) - TRAINING ONLY (scripts/train_mgan.py); main.py never imports it.

D(x, ref) -> predicted normalised PESQ-WB in [0, 1] of waveform x against the clean reference ref.
Input: two-channel stack of power-law compressed STFT magnitudes |STFT|^0.3 (n_fft 512, hop 128, periodic hann:
the generator's own STFT) of x and ref, shape (B, 2, 257, T). Body: 4 x [Conv2d 5x5 stride 2 + spectral norm +
LeakyReLU(0.3)], global average pooling, MLP (spectral-norm Linear C -> 50 -> 10, LeakyReLU) -> Linear 10 -> 1 -> sigmoid.
No batch norm: a D batch mixes clean/enhanced/noisy/replay rows, so batch statistics would depend on its composition.
Target (pesq_target): (PESQ-WB - 1.0) / 3.5 clipped to [0, 1].
ReplayBuffer: reservoir sample (fixed capacity, float16 on CPU) of past (enhanced, clean, target) triples; MetricGAN+
mixes such history into each D batch so D does not forget earlier generator outputs.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def pesq_target(pesq: np.ndarray | float) -> np.ndarray:
    """Normalised PESQ target (PESQ - 1) / 3.5 in [0, 1]; NaN (failed PESQ) stays NaN."""
    return np.clip((np.asarray(pesq, dtype=np.float64) - 1.0) / 3.5, 0.0, 1.0)


class MetricDiscriminator(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 128, compress: float = 0.3,
                 channels=(16, 32, 32, 64), hidden=(50, 10), slope: float = 0.3):
        super().__init__()
        self.n_fft, self.hop, self.compress = int(n_fft), int(hop), float(compress)
        self.register_buffer("window", torch.hann_window(self.n_fft, periodic=True), persistent=False)
        layers, cin = [], 2
        for cout in channels:
            layers += [spectral_norm(nn.Conv2d(cin, cout, 5, stride=2, padding=2)), nn.LeakyReLU(slope)]
            cin = cout
        self.body = nn.Sequential(*layers)
        mlp = []
        for h in hidden:
            mlp += [spectral_norm(nn.Linear(cin, h)), nn.LeakyReLU(slope)]
            cin = h
        mlp += [nn.Linear(cin, 1)]
        self.head = nn.Sequential(*mlp)

    def spec(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, N) wave -> (B, F, T) compressed magnitude; sqrt(re^2 + im^2 + eps) keeps gradients finite at zeros."""
        s = torch.stft(wave, self.n_fft, self.hop, window=self.window, center=True, pad_mode="reflect",
                       return_complex=True)
        return (s.real ** 2 + s.imag ** 2 + 1e-7).sqrt() ** self.compress

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """x, ref: (B, N) waves -> (B,) predicted normalised PESQ."""
        z = torch.stack([self.spec(x), self.spec(ref)], dim=1)
        z = self.body(z).mean(dim=(2, 3))
        return torch.sigmoid(self.head(z)).squeeze(-1)


class ReplayBuffer:
    """Reservoir sample of past (enhanced, clean, target) rows; float16 on CPU; state() / load() for last.pt."""

    def __init__(self, capacity: int, length: int, seed: int = 0):
        self.capacity, self.length = int(capacity), int(length)
        self.x = torch.zeros(self.capacity, self.length, dtype=torch.float16)
        self.ref = torch.zeros_like(self.x)
        self.y = torch.zeros(self.capacity, dtype=torch.float32)
        self.size, self.seen = 0, 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, x: torch.Tensor, ref: torch.Tensor, y: torch.Tensor) -> None:
        x, ref, y = x.detach().cpu(), ref.detach().cpu(), y.detach().cpu().float()
        for i in range(x.shape[0]):
            self.seen += 1
            if self.size < self.capacity:
                j = self.size
                self.size += 1
            else:
                j = int(self.rng.integers(self.seen))
                if j >= self.capacity:
                    continue
            self.x[j], self.ref[j], self.y[j] = x[i].half(), ref[i].half(), y[i]

    def sample(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        n = min(int(n), self.size)
        if n <= 0:
            return None
        idx = torch.from_numpy(self.rng.choice(self.size, size=n, replace=False))
        return (self.x[idx].float().to(device), self.ref[idx].float().to(device), self.y[idx].to(device))

    def state(self) -> dict:
        return {"x": self.x[:self.size].clone(), "ref": self.ref[:self.size].clone(), "y": self.y[:self.size].clone(),
                "seen": self.seen, "capacity": self.capacity, "length": self.length, "rng": self.rng.bit_generator.state}

    def load(self, st: dict) -> None:
        if int(st["length"]) != self.length:
            raise ValueError(f"replay length {st['length']} != {self.length}")
        n = min(int(st["x"].shape[0]), self.capacity)
        self.x[:n], self.ref[:n], self.y[:n] = st["x"][:n], st["ref"][:n], st["y"][:n]
        self.size, self.seen = n, int(st["seen"])
        self.rng.bit_generator.state = st["rng"]
