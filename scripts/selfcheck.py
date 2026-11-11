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
        # dsp-h options: length / finite / all-zeros, defaults bit-identical, beta -> 0 == LSA.
        v5 = dict(n_fft=1024, hop=256, noise_fraction=0.10, noise_bias=1.25, alpha=0.94,
                  xi_min_db=-25, gmin_db=-30, xi_mode="hrnr", hrnr_rho=0.3, highpass_hz=100)
        base = get_method("mmse_lsa", **v5)(x, 16000)
        explicit = get_method("mmse_lsa", **v5, gain_mode="lsa", beta=0.0, pause_att_db=0.0,
                              noise_bias_low=None, noise_bias_high=None)(x, 16000)
        beta0 = get_method("mmse_lsa", **v5, gain_mode="beta", beta=1e-7)(x, 16000)
        checks[f"mmse_lsa[dsp-v5] len {n}: explicit new defaults bit-identical"] = bool(
            np.array_equal(base, explicit))
        checks[f"mmse_lsa[dsp-v5] len {n}: beta 1e-7 equals LSA"] = bool(np.array_equal(base, beta0))
        for label, opt in (("beta -0.5", dict(gain_mode="beta", beta=-0.5)),
                           ("beta 1", dict(gain_mode="beta", beta=1.0)),
                           ("pause 6/4/4", dict(pause_thr_db=6, pause_att_db=4, pause_release_frames=4)),
                           ("bias 1.25/1.5", dict(noise_bias_low=1.25, noise_bias_high=1.5))):
            f = get_method("mmse_lsa", **v5, **opt)
            for kind, sig in (("noise", x), ("zeros", np.zeros(n))):
                m = f(sig, 16000)
                checks[f"mmse_lsa[{label}] {kind} len {n}: length, finite"] = (
                    m.size == n and bool(np.isfinite(m).all()))
                if kind == "zeros":
                    checks[f"mmse_lsa[{label}] zeros len {n}: output all zero"] = not np.any(m)
        for label, passed in checks.items():
            print(("PASS " if passed else "FAIL ") + label)
            ok &= passed
    print("selfcheck:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
