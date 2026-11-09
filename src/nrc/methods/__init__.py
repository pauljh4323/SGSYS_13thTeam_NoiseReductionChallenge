"""Method registry. Every method maps (noisy waveform, sample rate) -> enhanced waveform.

Clean audio is never passed to a method.
"""
from __future__ import annotations

import functools
import inspect

from . import mmse_lsa, noisy, omlsa, wiener

_REGISTRY = {
    "mmse_lsa": mmse_lsa.enhance,
    "noisy": noisy.enhance,
    "omlsa": omlsa.enhance,
    "wiener": wiener.enhance,
}


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_method(name: str, **params):
    """Return callable f(noisy, sr) with params bound; rejects unknown params."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown method {name!r}; available: {available()}")
    fn = _REGISTRY[name]
    accepted = set(inspect.signature(fn).parameters) - {"noisy", "sr"}
    unknown = set(params) - accepted
    if unknown:
        raise TypeError(f"Unknown params for {name!r}: {sorted(unknown)}")
    bound = functools.partial(fn, **params)

    def method(x, sr=16000):
        return bound(x, sr=sr)

    method.__name__ = name
    return method
