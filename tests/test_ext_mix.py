"""exp/r2-ext: ext_mix in nrc.data.remix.RemixData + nrc.data.extpool (synthetic package in a temp dir, CPU).

Run: python tests/test_ext_mix.py   (also collected by pytest if installed; no extra dependency needed).
The synthetic package uses real train_pool file names (splits/gen) with random audio, so the leakage guard runs on
the real split lists.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from nrc.data.extpool import SPLITS_DIR, ExtPool, ShardStore, ShardWriter, check_noise_rels  # noqa: E402
from nrc.data.remix import RemixData  # noqa: E402

N, L, TARGET = 12, 16000, 0.079130


def _lines(name):
    return [ln.strip() for ln in (SPLITS_DIR / name).read_text(encoding="utf-8").splitlines() if ln.strip()]


def _package(d: Path, extra_rel: str | None = None) -> Path:
    r = np.random.default_rng(5)
    w = ShardWriter(d / "librispeech", ["utt", "speaker", "rms"], shard_samples=5 * L)
    for k in range(9):
        x = (r.standard_normal(int(r.integers(L, 3 * L))) * r.uniform(500, 5000)).astype(np.int16)
        x[: L // 4] = 0  # a silent lead-in, so some crops are redrawn
        w.add(x, utt=f"u{k}", speaker=f"s{k % 3}", rms=float(np.sqrt(np.mean((x / 32768.0) ** 2))))
    w.close()
    rels = [_lines(f"{c}_train_pool.txt")[0] for c in ("demand", "esc50", "musan")]
    rels += [_lines("esc50_train_pool.txt")[1]]
    if extra_rel:
        rels.append(extra_rel)
    w = ShardWriter(d / "noise_pool", ["corpus", "rel", "status"], shard_samples=4 * L)
    for rel in rels:
        x = (r.standard_normal(int(r.integers(L // 2, 3 * L))) * r.uniform(200, 8000)).astype(np.int16)
        w.add(x, corpus=rel.split("/")[0], rel=rel, status="kept")
    w.close()
    return d


def _data(pool, **kw):
    r = np.random.default_rng(1)
    clean = (r.standard_normal((N, L)) * r.uniform(0.01, 0.2, (N, 1))).astype(np.float32)
    noise = (r.standard_normal((N, L)) * r.uniform(0.005, 0.3, (N, 1))).astype(np.float32)
    return RemixData(clean, clean + noise, rms_norm_target=TARGET, ext_mix={"pool": pool, **kw}, p_speed=0.8,
                     speed_range=(0.8, 1.2), p_eq=0.8, eq_gain_db=(-12, 12))


def _snr(c, m):
    c, n = c.astype(np.float64), m.astype(np.float64) - c.astype(np.float64)
    return 10 * np.log10(np.mean(c ** 2) / np.mean(n ** 2))


def test_shards_roundtrip():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        d = _package(Path(t))
        s = ShardStore(d / "librispeech")
        assert len(s) == 9 and len(set(s.shard)) > 1  # several shards
        assert all(s.get(k).dtype == np.int16 and s.get(k).size == s.length[k] for k in range(len(s)))


def test_snr_and_level_100_draws():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        pool = ExtPool(_package(Path(t)), min_clean=L)
        data = _data(pool)
        rng = np.random.default_rng(0)
        src, second, ext_noise = [], 0, 0
        for _ in range(100):
            info = {}
            c, m = data.sample(rng, info)
            assert c.shape == m.shape == (L,) and np.isfinite(c).all() and np.isfinite(m).all()
            assert abs(_snr(c, m) - info["snr_db"]) < 0.1, (_snr(c, m), info)
            assert -10 <= info["snr_db"] <= 20
            assert abs(np.sqrt(np.mean(m.astype(np.float64) ** 2)) - TARGET) < 1e-5
            src.append(info["clean_src"][0])
            ext_noise += info["noise"][0] == "ext"
            if info["noise2"] is not None:
                second += 1
                assert info["noise2"] != info["noise"] and -10 <= info["rel_db"] <= 0
                if info["clean_src"][0] == "ch":
                    assert info["noise2"] != info["clean_src"]
            if info["clean_src"][0] == "ch":
                assert info["noise"] != info["clean_src"]  # challenge noise never from the same file
        assert 30 <= src.count("libri") <= 70 and 15 <= second <= 45 and 45 <= ext_noise <= 75, \
            (src.count("libri"), second, ext_noise)


def test_reproducible():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        pool = ExtPool(_package(Path(t)), min_clean=L)
        a = _data(pool).batch(np.random.default_rng(3), 8)
        b = _data(pool).batch(np.random.default_rng(3), 8)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_leakage_guard():
    dev = _lines("esc50_gen_dev.txt")[0]
    test = _lines("musan_gen_test.txt")[0]
    excl = _lines("excluded.txt")[0]
    check_noise_rels([_lines("demand_train_pool.txt")[0]])
    for bad in (dev, test, excl, "esc50/audio/not-a-file.wav"):
        try:
            check_noise_rels([_lines("demand_train_pool.txt")[0], bad])
        except AssertionError:
            continue
        raise AssertionError(f"{bad} was accepted")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        d = _package(Path(t), extra_rel=dev)
        try:
            ExtPool(d, min_clean=L)
        except AssertionError:
            return
        raise AssertionError("ExtPool accepted a gen_dev noise file")


def test_options():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        pool = ExtPool(_package(Path(t)), min_clean=L)
        clean = np.zeros((N, L), np.float32) + 0.01
        for kw in ({"p_dm": 1.0}, {"ext_mix": {"pool": pool, "p_libri": 1.5}},
                   {"ext_mix": {"pool": pool, "snr": (0, 1)}}):
            kw.setdefault("ext_mix", {"pool": pool})
            try:
                RemixData(clean, clean * 2, rms_norm_target=TARGET, **kw)
            except ValueError:
                continue
            raise AssertionError(f"accepted {kw}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
