"""BiGRU magnitude-mask enhancer (noisy phase), trained from scratch.

noisy wave -> STFT (512/128 periodic hann) -> |X| -> log1p -> per-bin normalization
(train-set stats as buffers + learnable affine) -> front end -> BiGRU(L layers, H per direction)
-> Linear(2H->F) -> sigmoid mask -> mask * X -> ISTFT (exact input length).

front_end "none" (gru_v1): ReLU(Linear(F->H)).
front_end "conv2d": (B,1,T,F) -> 2 x [Conv2d 3x3, stride (1,2) (frequency only), padding (1,1)
(non-causal), GroupNorm/BatchNorm, PReLU] -> flatten (C x F'') -> ReLU(Linear(-> H)).
Conv modules exist only for "conv2d", so "none" keeps gru_v1's state_dict keys and init order.

Round-2 options (exp/r2-v6; defaults keep gru_v1..v5b keys, configs and outputs bit-identical):
mask_type "crm" (gru_v6a): complex ratio mask (Williamson, Wang & Wang 2016). The sigmoid head is the real part M_r;
  an imaginary head M_i = mask_imag_scale * tanh(Linear(2H->F)) with zero-initialised weight and bias (M_i = 0 at
  init, so a warm start from a "mag" checkpoint reproduces it). Enhanced = (M_r + j M_i) * X.
noise_aware True (gru_v6b): per-frame features from the dsp-v8 estimators (nrc.features: log stationary noise PSD +
  log decision-directed a priori SNR, 2F per frame), normalised by train-set buffers aux_mean/aux_std, added to the
  input projection through Linear(2F->H) with zero-initialised weight and bias: h = relu(inp(f) + inp_aux(a)).
  Features are computed in numpy from the noisy waveform (inside forward when not supplied).
"""
from __future__ import annotations

import torch
from torch import nn

MAX_PARAMS = 5_000_000
FRONT_ENDS = ("none", "conv2d")
CONV_NORMS = ("group", "batch")
MASK_TYPES = ("mag", "crm")
_DEFAULTS = {"front_end": "none", "conv_channels": 16, "conv_norm": "group", "mask_type": "mag",
             "mask_imag_scale": 1.0, "noise_aware": False}
NEW_HEAD_KEYS = ("out_imag.weight", "out_imag.bias", "inp_aux.weight", "inp_aux.bias", "aux_mean", "aux_std")


class GRUMask(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 128, hidden: int = 128, layers: int = 2,
                 front_end: str = "none", conv_channels: int = 16, conv_norm: str = "group",
                 mask_type: str = "mag", mask_imag_scale: float = 1.0, noise_aware: bool = False):
        super().__init__()
        if front_end not in FRONT_ENDS:
            raise ValueError(f"front_end must be one of {FRONT_ENDS}, got {front_end!r}")
        if conv_norm not in CONV_NORMS:
            raise ValueError(f"conv_norm must be one of {CONV_NORMS}, got {conv_norm!r}")
        if mask_type not in MASK_TYPES:
            raise ValueError(f"mask_type must be one of {MASK_TYPES}, got {mask_type!r}")
        self.n_fft, self.hop = int(n_fft), int(hop)
        self.n_bins = self.n_fft // 2 + 1
        self.front_end = front_end
        self.config = {"n_fft": self.n_fft, "hop": self.hop, "hidden": int(hidden), "layers": int(layers)}
        self.mask_type, self.mask_imag_scale = mask_type, float(mask_imag_scale)
        self.noise_aware = bool(noise_aware)
        extra = {"front_end": front_end, "conv_channels": int(conv_channels), "conv_norm": conv_norm,
                 "mask_type": mask_type, "mask_imag_scale": float(mask_imag_scale),
                 "noise_aware": bool(noise_aware)}
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
        # New modules are created after all v5b modules, so the v5b init order / RNG stream is unchanged.
        if mask_type == "crm":
            self.out_imag = nn.Linear(2 * hidden, self.n_bins)
            nn.init.zeros_(self.out_imag.weight)
            nn.init.zeros_(self.out_imag.bias)
        if self.noise_aware:
            self.register_buffer("aux_mean", torch.zeros(2 * self.n_bins))
            self.register_buffer("aux_std", torch.ones(2 * self.n_bins))
            self.inp_aux = nn.Linear(2 * self.n_bins, hidden)
            nn.init.zeros_(self.inp_aux.weight)
            nn.init.zeros_(self.inp_aux.bias)
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

    @torch.no_grad()
    def set_aux_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.aux_mean.copy_(mean)
        self.aux_std.copy_(std.clamp_min(1e-3))

    def aux_features(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, N) waveform -> raw noise-aware features (B, T, 2F) on wave's device (numpy, nrc.features)."""
        from ..features import batch_features
        a = batch_features(wave.detach().cpu().numpy(), self.n_fft, self.hop)
        return torch.from_numpy(a).to(wave.device)

    def features(self, mag: torch.Tensor) -> torch.Tensor:
        """|X| (B, F, T) -> normalized log1p features (B, T, F)."""
        f = torch.log1p(mag).transpose(1, 2)
        return (f - self.feat_mean) / self.feat_std * self.norm_gain + self.norm_bias

    def masks(self, mag: torch.Tensor, aux: torch.Tensor | None = None):
        """|X| (B, F, T) -> (M_r in (0, 1), M_i or None), each (B, F, T). aux: raw features (B, T, 2F)."""
        f = self.features(mag)
        if self.front_end == "conv2d":
            b, t, _ = f.shape
            f = self.conv(f.unsqueeze(1))                   # (B, C, T, F'')
            f = f.permute(0, 2, 1, 3).reshape(b, t, -1)     # (B, T, C*F'')
        if self.noise_aware:
            if aux is None:
                raise ValueError("noise_aware model needs aux features (use forward(wave) or aux_features)")
            h = torch.relu(self.inp(f) + self.inp_aux((aux - self.aux_mean) / self.aux_std))
        else:
            h = torch.relu(self.inp(f))
        h, _ = self.gru(h)
        m_r = torch.sigmoid(self.out(h)).transpose(1, 2)
        if self.mask_type == "crm":
            return m_r, (self.mask_imag_scale * torch.tanh(self.out_imag(h))).transpose(1, 2)
        return m_r, None

    def mask(self, mag: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        """|X| (B, F, T) -> real mask M_r in (0, 1), (B, F, T)."""
        return self.masks(mag, aux)[0]

    def apply_mask(self, spec: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        """Complex noisy STFT (B, F, T) -> complex enhanced STFT."""
        m_r, m_i = self.masks(spec.abs(), aux)
        if m_i is None:
            return m_r * spec
        yr, yi = spec.real, spec.imag
        return torch.complex(m_r * yr - m_i * yi, m_r * yi + m_i * yr)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, N) noisy -> (B, N) enhanced. Inputs shorter than n_fft are zero-padded internally."""
        n = wave.shape[-1]
        x = wave if n > self.n_fft else nn.functional.pad(wave, (0, self.n_fft + 1 - n))
        spec = self.stft(x)
        aux = self.aux_features(x) if self.noise_aware else None
        enh = self.istft(self.apply_mask(spec, aux), length=x.shape[-1])
        return enh[..., :n]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_partial(model: GRUMask, ckpt: dict) -> list[str]:
    """Warm start from a checkpoint of the same base architecture: strict except the round-2 heads/buffers that
    the checkpoint lacks (they keep their zero / identity init). Returns the keys left at init."""
    base = {k: v for k, v in model.config.items() if k not in ("mask_type", "mask_imag_scale", "noise_aware")}
    src = {k: v for k, v in ckpt["model_cfg"].items() if k not in ("mask_type", "mask_imag_scale", "noise_aware")}
    if base != src:
        raise ValueError(f"init checkpoint base model_cfg {src} != config {base}")
    res = model.load_state_dict(ckpt["model_state"], strict=False)
    bad = [k for k in res.missing_keys if k not in NEW_HEAD_KEYS]
    if bad or res.unexpected_keys:
        raise ValueError(f"init mismatch: missing {bad}, unexpected {res.unexpected_keys}")
    return list(res.missing_keys)


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
