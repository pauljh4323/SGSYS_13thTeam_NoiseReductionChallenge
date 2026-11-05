"""Self-check: STFT->ISTFT identity reconstruction and output-length preservation."""
from __future__ import annotations

import sys

import numpy as np

import _common  # noqa: F401  (sys.path setup)
from nrc.methods import get_method
from nrc.stft import istft, stft


def main() -> int:
    rng = np.random.default_rng(0)
    wiener = get_method("wiener")
    ok = True
    for n in (48000, 47999):
        x = 0.1 * rng.standard_normal(n)
        _, _, z = stft(x)
        y = istft(z, n)
        err = float(np.max(np.abs(y - x)))
        w = wiener(x, 16000)
        checks = {
            f"identity len {n}: max err {err:.2e} < 1e-8": err < 1e-8,
            f"identity len {n}: length preserved ({y.size})": y.size == n,
            f"wiener len {n}: length preserved ({w.size}), finite": w.size == n
            and bool(np.isfinite(w).all()),
        }
        for label, passed in checks.items():
            print(("PASS " if passed else "FAIL ") + label)
            ok &= passed
    print("selfcheck:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
