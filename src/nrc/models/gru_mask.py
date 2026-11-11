"""BiGRU magnitude-mask enhancer (noisy phase), trained from scratch.

noisy wave -> STFT (512/128 periodic hann) -> |X| -> log1p -> per-bin normalization
(train-set stats as buffers + learnable affine) -> front end -> BiGRU(L layers, H per direction)
-> Linear(2H->F) -> sigmoid mask -> mask * X -> ISTFT (exact input length).

front_end "none" (gru_v1): ReLU(Linear(F->H)).
front_end "conv2d": (B,1,T,F) -> 2 x [Conv2d 3x3, stride (1,2) (frequency only), padding (1,1)
(non-causal), GroupNorm/BatchNorm, PReLU] -> flatten (C x F'') -> ReLU(Linear(-> H)).
Conv modules exist only for "conv2d", so "none" keeps gru_v1's state_dict keys and init order.
"""
from __future__ import annotations

import torch
from torch import nn

MAX_PARAMS = 5_000_000
FRONT_ENDS = ("none", "conv2d")
CONV_NORMS = ("group", "batch")
_DEFAULTS = {"front_end": "none", "conv_channels": 16, "conv_norm": "group"}


class GRUMask(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 128, hidden: int = 128, layers: int = 2,
                 front_end: str = "none", conv_channels: int = 16, conv_norm: str = "group"):
        super().__init__()
        if front_end not in FRONT_ENDS:
            raise ValueError(f"front_end must be one of {FRONT_ENDS}, got {front_end!r}")
        if conv_norm not in CONV_NORMS:
            raise ValueError(f"conv_norm must be one of {CONV_NORMS}, got {conv_norm!r}")
        self.n_fft, self.hop = int(n_fft), int(hop)
        self.n_bins = self.n_fft // 2 + 1
        self.front_end = front_end
        self.config = {"n_fft": self.n_fft, "hop": self.hop, "hidden": int(hidden), "layers": int(layers)}
        extra = {"front_end": front_end, "conv_channels": int(conv_channels), "conv_norm": conv_norm}
        # Only non-default options are stored, so gru_v1 checkpoints keep their exact model_cfg.
        self.config.update({k: v for k, v in extra.items() if v != _DEFAULTS[k]})
        self.register_buffer("window", torch.hann_window(self.n_fft, periodic=True), persistent=False)
        # Per-bin statistics of log1p|X| over the train noisy set (set by set_norm_stats).
        self.register_buffer("feat_mean", torch.zeros(self.n_bins))
        self.register_buffer("feat_std", torch.ones(self.n_bins))
        self.norm_gain = nn.Parameter(torch.ones(self.n_bins))
        self.norm_bias = nn.Parameter(torch.zeros(self.n_bins))
        if front_end == "conv2d":
            ch = int(conv_channels)
            norm = (lambda c: nn.GroupNorm(min(4, c), c)) if conv_norm == "group" else nn.BatchNorm2d
            self.conv = nn.Sequential(
                nn.Conv2d(1, ch, 3, stride=(1, 2), padding=(1, 1)), norm(ch), nn.PReLU(ch),
                nn.Conv2d(ch, ch, 3, stride=(1, 2), padding=(1, 1)), norm(ch), nn.PReLU(ch))
            f_red = self.n_bins
            for _ in range(2):
                f_red = (f_red + 2 - 3) // 2 + 1
            self.inp = nn.Linear(ch * f_red, hidden)
        else:
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
        f = self.features(mag)
        if self.front_end == "conv2d":
            b, t, _ = f.shape
            f = self.conv(f.unsqueeze(1))                   # (B, C, T, F'')
            f = f.permute(0, 2, 1, 3).reshape(b, t, -1)     # (B, T, C*F'')
        h = torch.relu(self.inp(f))
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
