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
        for est in ("initial", "mcra", "quantile"):
            lsa = get_method("mmse_lsa", noise_est=est)
            for kind, sig in (("noise", x), ("zeros", np.zeros(n))):
                m = lsa(sig, 16000)
                checks[f"mmse_lsa[{est}] {kind} len {n}: length preserved ({m.size}), finite"] = (
                    m.size == n and bool(np.isfinite(m).all()))
                if kind == "zeros":
                    checks[f"mmse_lsa[{est}] zeros len {n}: output all zero"] = not np.any(m)
        q = get_method("mmse_lsa", noise_est="quantile", q=0.1, bias=1.5, smooth_frames=5)
        for kind, sig in (("noise", x), ("zeros", np.zeros(n))):
            m = q(sig, 16000)
            checks[f"mmse_lsa[quantile q=.1 b=1.5 s=5] {kind} len {n}: length, finite"] = (
                m.size == n and bool(np.isfinite(m).all()))
        for label, passed in checks.items():
            print(("PASS " if passed else "FAIL ") + label)
            ok &= passed
    print("selfcheck:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
