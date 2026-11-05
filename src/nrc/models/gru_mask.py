"""BiGRU magnitude-mask enhancer (noisy phase), trained from scratch.

noisy wave -> STFT (512/128 periodic hann) -> |X| -> log1p -> per-bin normalization
(train-set stats as buffers + learnable affine) -> Linear(F->H) -> BiGRU(L layers, H per direction)
-> Linear(2H->F) -> sigmoid mask -> mask * X -> ISTFT (exact input length).
"""
from __future__ import annotations

import torch
from torch import nn

MAX_PARAMS = 2_000_000


class GRUMask(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 128, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.n_fft, self.hop = int(n_fft), int(hop)
        self.n_bins = self.n_fft // 2 + 1
        self.config = {"n_fft": self.n_fft, "hop": self.hop, "hidden": int(hidden), "layers": int(layers)}
        self.register_buffer("window", torch.hann_window(self.n_fft, periodic=True), persistent=False)
        # Per-bin statistics of log1p|X| over the train noisy set (set by set_norm_stats).
        self.register_buffer("feat_mean", torch.zeros(self.n_bins))
        self.register_buffer("feat_std", torch.ones(self.n_bins))
        self.norm_gain = nn.Parameter(torch.ones(self.n_bins))
        self.norm_bias = nn.Parameter(torch.zeros(self.n_bins))
        self.inp = nn.Linear(self.n_bins, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, self.n_bins)
        n = count_params(self)
        if n >= MAX_PARAMS:
            raise ValueError(f"GRUMask has {n} params (limit {MAX_PARAMS})")

    # --- spectral helpers -------------------------------------------------------------------
    def stft(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, N) -> complex (B, F, T)."""
        return torch.stft(wave, self.n_fft, self.hop, window=self.window, center=True,
                          pad_mode="reflect", return_complex=True)

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(spec, self.n_fft, self.hop, window=self.window, center=True, length=length)

    @torch.no_grad()
    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp_min(1e-3))

    def features(self, mag: torch.Tensor) -> torch.Tensor:
        """|X| (B, F, T) -> normalized log1p features (B, T, F)."""
        f = torch.log1p(mag).transpose(1, 2)
        return (f - self.feat_mean) / self.feat_std * self.norm_gain + self.norm_bias

    def mask(self, mag: torch.Tensor) -> torch.Tensor:
        """|X| (B, F, T) -> mask in (0, 1), (B, F, T)."""
        h = torch.relu(self.inp(self.features(mag)))
        h, _ = self.gru(h)
        return torch.sigmoid(self.out(h)).transpose(1, 2)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, N) noisy -> (B, N) enhanced. Inputs shorter than n_fft are zero-padded internally."""
        n = wave.shape[-1]
        x = wave if n > self.n_fft else nn.functional.pad(wave, (0, self.n_fft + 1 - n))
        spec = self.stft(x)
        enh = self.istft(self.mask(spec.abs()) * spec, length=x.shape[-1])
        return enh[..., :n]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_weights(path, model: GRUMask, **meta) -> None:
    """Inference-only checkpoint: state dict + model config (+ metadata such as epoch/pesq)."""
    torch.save({"format": "gru_mask_v1", "model_cfg": model.config,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                **meta}, path)


def load_weights(path, device: str = "cpu") -> GRUMask:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GRUMask(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()
