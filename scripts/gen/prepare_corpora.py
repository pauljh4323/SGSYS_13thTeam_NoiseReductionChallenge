"""Extract the external noise corpora in data_ext/ into 16 kHz mono wavs + an inventory (exp/r2-gen).

  data_ext/dl_demand/<ENV>_16k.zip   -> data_ext/demand/<ENV>/ch01.wav       (channel 1 only)
  data_ext/dl_esc50/ESC-50-master.zip -> data_ext/esc50/audio/*.wav (44.1 -> 16 kHz, resample_poly 160/441)
                                         data_ext/esc50/esc50.csv
  data_ext/dl_musan/musan.tar.gz     -> data_ext/musan/noise/{free-sound,sound-bible}/*.wav (16 kHz as shipped)
Archives are untrusted: members are checked to stay inside the target (zip) / tarfile filter="data" (tar),
and only the needed members are extracted. Writes data_ext/inventory.json (archive SHA-256, sizes,
file counts, hours). Run: python -I scripts/gen/prepare_corpora.py [--corpus demand esc50 musan]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "data_ext"
DEMAND_ENVS = ("DKITCHEN", "DLIVING", "DWASHING", "NFIELD", "NPARK", "NRIVER", "OHALLWAY", "OMEETING",
               "OOFFICE", "PCAFETER", "PRESTO", "PSTATION", "SPSQUARE", "STRAFFIC", "TBUS", "TCAR", "TMETRO")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return p


def hours(files) -> float:
    return sum(sf.info(str(f)).frames / sf.info(str(f)).samplerate for f in files) / 3600


def write16(path: Path, x: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 32767 / 32768), sr, subtype="PCM_16")


def demand(inv: dict) -> None:
    out, archives = EXT / "demand", {}
    for env in DEMAND_ENVS:
        z = EXT / "dl_demand" / f"{env}_16k.zip"
        archives[z.name] = {"sha256": sha256(z), "bytes": z.stat().st_size}
        with zipfile.ZipFile(z) as zf:
            members = [m for m in zf.namelist() if safe_name(m).name == "ch01.wav"]
            assert len(members) == 1, (env, members)
            x, sr = sf.read(io.BytesIO(zf.read(members[0])), dtype="float64")
            assert sr == 16000 and x.ndim == 1, (env, sr, x.shape)
            write16(out / env / "ch01.wav", x)
    files = sorted(out.glob("*/ch01.wav"))
    inv["demand"] = {"archives": archives, "files": len(files), "hours": hours(files)}


def esc50(inv: dict) -> None:
    z = EXT / "dl_esc50" / "ESC-50-master.zip"
    out = EXT / "esc50"
    n = 0
    with zipfile.ZipFile(z) as zf:
        for m in zf.namelist():
            p = safe_name(m)
            if p.name == "esc50.csv" and "meta" in p.parts:
                (out).mkdir(parents=True, exist_ok=True)
                (out / "esc50.csv").write_bytes(zf.read(m))
            elif p.suffix == ".wav" and "audio" in p.parts:
                x, sr = sf.read(io.BytesIO(zf.read(m)), dtype="float64")
                if x.ndim > 1:
                    x = x[:, 0]
                assert sr == 44100, (m, sr)
                write16(out / "audio" / p.name, resample_poly(x, 160, 441))
                n += 1
    files = sorted((out / "audio").glob("*.wav"))
    assert n == len(files) == 2000, n
    inv["esc50"] = {"archives": {z.name: {"sha256": sha256(z), "bytes": z.stat().st_size}},
                    "files": len(files), "hours": hours(files), "resample": "44100 -> 16000 resample_poly(160, 441)"}


def musan(inv: dict) -> None:
    t = EXT / "dl_musan" / "musan.tar.gz"
    out = EXT / "musan"
    digest = sha256(t)
    with tarfile.open(t, "r:gz") as tf:
        for m in tf:
            p = safe_name(m.name)
            if m.isfile() and len(p.parts) >= 3 and p.parts[:2] == ("musan", "noise"):
                tf.extract(m, path=EXT, filter="data")  # -> data_ext/musan/noise/...
    files = sorted((out / "noise").rglob("*.wav"))
    srs = {sf.info(str(f)).samplerate for f in files}
    chans = {sf.info(str(f)).channels for f in files}
    assert srs == {16000} and chans == {1}, (srs, chans)
    inv["musan"] = {"archives": {t.name: {"sha256": digest, "bytes": t.stat().st_size}},
                    "files": len(files), "hours": hours(files)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", nargs="+", default=["demand", "esc50", "musan"])
    args = ap.parse_args(argv)
    inv_path = EXT / "inventory.json"
    inv = json.loads(inv_path.read_text()) if inv_path.is_file() else {}
    for c in args.corpus:
        {"demand": demand, "esc50": esc50, "musan": musan}[c](inv)
        inv_path.write_text(json.dumps(inv, indent=2))
        print(c, {k: v for k, v in inv[c].items() if k != "archives"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
