"""Zero-shot inference with a public pretrained enhancer (exp/r2-gen; reference only, never part of main.py).

Runs inside the model's own conda env (ext_dfn / ext_mpsenet / ext_cmgan; not in requirements*.txt) and is
self-contained (no nrc import). For every file of a split it writes outputs/ext_<model>/<split>/<noisy name>
(16-bit PCM, 16 kHz, same length: shorter/longer model outputs are zero-padded/cropped and counted) and
outputs/ext_<model>/<split>_timing.csv (model CPU seconds per file, torch threads). Existing outputs are
skipped (resumable). A file whose output would clip in 16-bit is scaled as a whole to peak 0.99 (PESQ is
level-invariant), counted as rescaled_files in <split>_meta.json. Score with eval.py --config configs/gen/ext_<model>.yaml --pred-dir ...

  dfn3     DeepFilterNet3 (pip deepfilternet 0.5.6, default model): 16 kHz -> 48 kHz resample_poly(3, 1),
           df.enhance.enhance, -> 16 kHz resample_poly(1, 3).
  mpsenet  MP-SENet official best_ckpt/g_best_vb (VoiceBank+DEMAND), steps of the repo's inference.py
           (RMS normalisation, compressed mag/phase STFT 400/100/400, inverse, de-normalisation).
  cmgan    CMGAN official src/best_ckpt/ckpt (VoiceBank+DEMAND), steps of src/evaluation.py
           enhance_one_track on CPU (torch.stft return_complex + view_as_real: same numbers as the old API).
Run: python scripts/gen/ext_infer.py --model <m> --split <split> [--repo data_ext/repos/<repo>] [--threads 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[2]
REPOS = {"mpsenet": REPO / "data_ext" / "repos" / "MP-SENet", "cmgan": REPO / "data_ext" / "repos" / "CMGAN" / "src"}


def split_files(split: str) -> tuple[Path, pd.DataFrame]:
    if split in ("train", "validation"):
        root = REPO / "data" / "noise_reduction_challenge"
        return root, pd.read_csv(root / split / "samples.csv", encoding="utf-8-sig")
    root = REPO / "data" / ("val_remix" if split == "val_remix" else f"gen/{split}")
    return root, pd.read_csv(root / "samples.csv", encoding="utf-8-sig")


class AttrDict(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.__dict__ = self


def load_dfn3():
    from df.enhance import enhance, init_df
    model, df_state, _ = init_df()  # default = DeepFilterNet3 checkpoint
    model.eval()

    def run(x16: np.ndarray) -> np.ndarray:
        x48 = torch.from_numpy(resample_poly(x16, 3, 1).astype(np.float32)).unsqueeze(0)
        y48 = enhance(model, df_state, x48).squeeze(0).numpy().astype(np.float64)
        return resample_poly(y48, 1, 3)
    return model, run, {"checkpoint": "DeepFilterNet3 (deepfilternet 0.5.6 default)", "sr_model": 48000}


def load_mpsenet(repo: Path):
    sys.path.insert(0, str(repo))
    from models.model import MPNet  # noqa: E402  (repo code)
    ckpt = repo / "best_ckpt" / "g_best_vb"
    h = AttrDict(json.loads((repo / "best_ckpt" / "config.json").read_text()))
    model = MPNet(h)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["generator"])
    model.eval()
    win = torch.hann_window(h.win_size)

    def stft(y):
        s = torch.view_as_real(torch.stft(y, h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window=win,
                                          center=True, pad_mode="reflect", normalized=False, return_complex=True))
        mag = torch.sqrt(s.pow(2).sum(-1) + 1e-9)
        pha = torch.atan2(s[..., 1] + 1e-10, s[..., 0] + 1e-5)
        return torch.pow(mag, h.compress_factor), pha

    def istft(mag, pha):
        mag = torch.pow(mag, 1.0 / h.compress_factor)
        com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
        return torch.istft(com, h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window=win, center=True)

    def run(x16: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x16.astype(np.float32))
        norm = torch.sqrt(len(x) / torch.sum(x ** 2.0))
        amp, pha = stft((x * norm).unsqueeze(0))
        amp_g, pha_g, _ = model(amp, pha)
        return (istft(amp_g, pha_g) / norm).squeeze().numpy().astype(np.float64)
    return model, run, {"checkpoint": str(ckpt.relative_to(REPO).as_posix()), "sr_model": 16000}


def load_cmgan(repo: Path):
    sys.path.insert(0, str(repo))
    from models import generator  # noqa: E402  (repo code)
    from utils import power_compress, power_uncompress  # noqa: E402
    ckpt = repo / "best_ckpt" / "ckpt"
    n_fft, hop, cut_len = 400, 100, 16000 * 16
    model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    win = torch.hamming_window(n_fft)

    def run(x16: np.ndarray) -> np.ndarray:
        noisy = torch.from_numpy(x16.astype(np.float32)).unsqueeze(0)
        c = torch.sqrt(noisy.size(-1) / torch.sum(noisy ** 2.0, dim=-1))
        noisy = torch.transpose(torch.transpose(noisy, 0, 1) * c, 0, 1)
        length = noisy.size(-1)
        frame_num = int(np.ceil(length / 100))
        padded_len = frame_num * 100
        noisy = torch.cat([noisy, noisy[:, :padded_len - length]], dim=-1)
        if padded_len > cut_len:
            bs = int(np.ceil(padded_len / cut_len))
            while 100 % bs != 0:
                bs += 1
            noisy = torch.reshape(noisy, (bs, -1))
        spec = torch.view_as_real(torch.stft(noisy, n_fft, hop, window=win, onesided=True, return_complex=True))
        spec = power_compress(spec).permute(0, 1, 3, 2)
        er, ei = model(spec)
        er, ei = er.permute(0, 1, 3, 2), ei.permute(0, 1, 3, 2)
        unc = power_uncompress(er, ei).squeeze(1)
        y = torch.istft(torch.view_as_complex(unc.contiguous()), n_fft, hop, window=win, onesided=True)
        return (torch.flatten(y / c)[:length]).numpy().astype(np.float64)
    return model, run, {"checkpoint": str(ckpt.relative_to(REPO).as_posix()), "sr_model": 16000}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=("dfn3", "mpsenet", "cmgan"))
    ap.add_argument("--split", required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if args.model == "dfn3":
        model, run, info = load_dfn3()
    else:
        repo = Path(args.repo) if args.repo else REPOS[args.model]
        model, run, info = (load_mpsenet if args.model == "mpsenet" else load_cmgan)(repo.resolve())
    n_params = sum(p.numel() for p in model.parameters())
    root, man = split_files(args.split)
    out_root = REPO / "outputs" / f"ext_{args.model}"
    out = out_root / args.split
    out.mkdir(parents=True, exist_ok=True)
    tpath = out_root / f"{args.split}_timing.csv"
    timing = pd.read_csv(tpath).to_dict("records") if tpath.is_file() else []
    # A row written before peak rescaling existed (no "rescaled" field) that clipped was hard-clipped: redo it.
    seen = {r["file"] for r in timing if not (pd.isna(r.get("rescaled", np.nan)) and r.get("clipped", 0) > 0)}
    for _, r in man.iterrows():
        name = Path(r.noisy_path).name
        dst = out / name
        if dst.is_file() and name in seen:
            continue
        x, sr = sf.read(str(root / r.noisy_path), dtype="float64")
        assert sr == 16000 and x.ndim == 1
        t0 = time.process_time(); w0 = time.perf_counter()
        with torch.inference_mode():
            y = run(x)
        cpu, wall = time.process_time() - t0, time.perf_counter() - w0
        len_diff = int(y.size - x.size)
        y = y[:x.size] if y.size >= x.size else np.pad(y, (0, x.size - y.size))
        if not np.isfinite(y).all():
            y = np.nan_to_num(y)
        scaled = np.round(y * 32768.0)
        clipped = int(np.count_nonzero((scaled > 32767) | (scaled < -32768)))  # would-clip samples before rescale
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        rescaled = int(clipped > 0)
        if rescaled:  # PESQ is level-invariant: scale the whole file to peak 0.99 instead of hard-clipping
            scaled = np.round(y * (0.99 / peak) * 32768.0)
        sf.write(str(dst), np.clip(scaled, -32768, 32767).astype(np.int16), 16000, subtype="PCM_16")
        timing = [t for t in timing if t["file"] != name] + [
            {"file": name, "seconds_audio": x.size / 16000, "cpu_s": cpu, "wall_s": wall, "len_diff": len_diff,
             "clipped": clipped, "rescaled": rescaled, "peak": peak}]
        pd.DataFrame(timing).to_csv(tpath, index=False)
    t = pd.DataFrame(timing)
    meta = {"model": args.model, "params": n_params, "threads": torch.get_num_threads(), "torch": torch.__version__,
            **info, "files": len(t), "cpu_s_per_file_mean": float(t.cpu_s.mean()),
            "wall_s_per_file_mean": float(t.wall_s.mean()), "len_mismatch_files": int((t.len_diff != 0).sum()),
            "clipped_samples": int(t.clipped.sum()),
            "rescaled_files": int(t["rescaled"].fillna(0).sum()) if "rescaled" in t else 0}
    (out_root / f"{args.split}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
