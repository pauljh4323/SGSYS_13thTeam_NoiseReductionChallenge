"""main.py length check on short/long inputs (exp/r2-gen): a 1 s and a 10 s noisy file (validation clean,
concatenated for 10 s, plus a gen_test external noise at 5 dB) go through
`python main.py --input logs/length_check/input --output logs/length_check/output` (submission config,
unchanged); the outputs must keep sample rate, channel count, sample count and filename.
Run: python scripts/gen/length_check.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "src")]
from noiselib import REPO, active_noise, loop_excerpt, read_noise  # noqa: E402
from nrc.io import load_manifest, read_row, resolve_data_root, to_pcm16  # noqa: E402


def main() -> int:
    base = REPO / "logs" / "length_check"
    inp, outp = base / "input", base / "output"
    for d in (inp, outp):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.wav"):
            f.unlink()
    root = resolve_data_root(None)
    df = load_manifest(root, "validation").sort_values("segment_id").head(4)
    speech = np.concatenate([read_row(root, r, "clean") for _, r in df.iterrows()])
    noise, _ = active_noise(read_noise((REPO / "splits" / "gen" / "musan_gen_test.txt")
                                       .read_text(encoding="utf-8").split()[0]))
    for name, sec in (("len_1s.wav", 1), ("len_10s.wav", 10)):
        c = speech[:sec * 16000]
        n = loop_excerpt(noise, c.size, 0)
        y = c + n * np.sqrt(np.mean(c ** 2) / (np.mean(n ** 2) * 10 ** 0.5))
        y *= min(1.0, 0.99 / np.max(np.abs(y)))
        sf.write(str(inp / name), to_pcm16(y)[0], 16000, subtype="PCM_16")
    r = subprocess.run([sys.executable, "main.py", "--input", str(inp), "--output", str(outp)], cwd=REPO,
                       capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    ok = r.returncode == 0
    for f in sorted(inp.glob("*.wav")):
        a, o = sf.info(str(f)), outp / f.name
        b = sf.info(str(o)) if o.is_file() else None
        same = b is not None and (a.samplerate, a.channels, a.frames) == (b.samplerate, b.channels, b.frames)
        print(f"{f.name}: in {a.samplerate} Hz {a.channels} ch {a.frames} | out "
              f"{(b.samplerate, b.channels, b.frames) if b else 'MISSING'} -> {'OK' if same else 'FAIL'}")
        ok &= same
    extra = sorted({p.name for p in outp.iterdir()} - {p.name for p in inp.iterdir()})
    ok &= not extra
    print("LENGTH CHECK", "PASSED" if ok else f"FAILED (extra: {extra})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
