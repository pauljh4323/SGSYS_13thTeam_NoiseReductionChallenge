"""Identity passthrough (reference point: unprocessed noisy input)."""
from __future__ import annotations

from ..io import as_waveform


def enhance(noisy, sr=16000):
    return as_waveform(noisy).copy()
