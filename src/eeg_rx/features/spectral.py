"""Spectral baseline features: band power, relative band power and frontal alpha asymmetry."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}


def band_powers(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """(channels, samples) -> (channels, bands) absolute power from a Welch spectrum (2 s segments)."""
    freqs, psd = welch(epoch, fs=sfreq, nperseg=min(epoch.shape[-1], int(2 * sfreq)), axis=-1)
    df = freqs[1] - freqs[0]
    out = np.empty((epoch.shape[0], len(BANDS)))
    for b, (lo, hi) in enumerate(BANDS.values()):
        sel = (freqs >= lo) & (freqs < hi)
        out[:, b] = psd[:, sel].sum(axis=1) * df
    return out


def bandpower_features(epoch: np.ndarray, sfreq: float, channels) -> np.ndarray:
    bp = band_powers(epoch, sfreq)
    log_abs = np.log10(bp + 1e-12)
    rel = bp / bp.sum(axis=1, keepdims=True)
    ch = list(channels)
    alpha = list(BANDS).index("alpha")
    asym = np.log(bp[ch.index("F4"), alpha] + 1e-12) - np.log(bp[ch.index("F3"), alpha] + 1e-12)
    return np.concatenate([log_abs.ravel(), rel.ravel(), [asym]])


def bandpower_names(channels) -> list[str]:
    names = [f"logpow_{c}_{b}" for c in channels for b in BANDS]
    names += [f"relpow_{c}_{b}" for c in channels for b in BANDS]
    return names + ["alpha_asymmetry_F4_F3"]
