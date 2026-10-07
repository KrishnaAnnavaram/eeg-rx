"""Synthetic resting-state EEG with subject fingerprints and an adjustable label effect.

Each subject has its own channel gains and its own alpha frequency (a "fingerprint"). An
epoch-wise split can learn these fingerprints and look accurate even when the label has no
effect. ``effect`` controls how much the responders differ: more frontal theta and a shift of
the frontal alpha asymmetry. ``effect=0`` gives data with no label signal at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import CHANNELS
from .io import Recording

FRONTAL = {"Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"}
POSTERIOR = {"P3", "Pz", "P4", "O1", "O2", "T5", "T6"}


@dataclass(frozen=True)
class SynthSpec:
    n_responders: int = 12
    n_nonresponders: int = 18
    seconds: float = 120.0
    sfreq: float = 256.0
    effect: float = 0.2
    fingerprint: float = 0.35
    artefact_rate: float = 0.05
    seed: int = 42


def _pink(rng, shape, sfreq):
    white = rng.normal(size=shape)
    spec = np.fft.rfft(white, axis=-1)
    f = np.fft.rfftfreq(shape[-1], 1 / sfreq)
    f[0] = f[1]
    return np.fft.irfft(spec / np.sqrt(f), n=shape[-1], axis=-1)


def make_subject(rng, label: int, sid: str, spec: SynthSpec) -> Recording:
    n = int(spec.seconds * spec.sfreq)
    t = np.arange(n) / spec.sfreq
    nch = len(CHANNELS)
    gains = np.exp(rng.normal(0, spec.fingerprint, nch))
    alpha_f = rng.uniform(9.0, 11.5)
    theta_f = rng.uniform(5.0, 7.0)
    noise = _pink(rng, (nch, n), spec.sfreq)
    noise *= 12.0 / noise.std(axis=1, keepdims=True)
    x = noise
    for c, name in enumerate(CHANNELS):
        alpha_amp = 14.0 if name in POSTERIOR else 6.0
        theta_amp = 5.0 if name in FRONTAL else 3.0
        if label == 1 and name in FRONTAL:
            theta_amp *= 1 + spec.effect
        if name == "F4":
            alpha_amp *= 1 + 0.5 * spec.effect * (1 if label else -1)
        mod = 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t + rng.uniform(0, 6.28))
        x[c] += alpha_amp * mod * np.sin(2 * np.pi * alpha_f * t + rng.uniform(0, 6.28))
        x[c] += theta_amp * np.sin(2 * np.pi * theta_f * t + rng.uniform(0, 6.28))
        x[c] *= gains[c]
    # Eye-blink artefacts on frontal channels in some 15 s blocks.
    block = int(15 * spec.sfreq)
    for b in range(n // block):
        if rng.uniform() < spec.artefact_rate:
            at = b * block + rng.integers(0, block - int(0.4 * spec.sfreq))
            blink = 450 * np.hanning(int(0.4 * spec.sfreq))
            for c, name in enumerate(CHANNELS):
                if name in FRONTAL:
                    x[c, at:at + len(blink)] += blink
    return Recording(sid, label, x, spec.sfreq).validate()


def generate(spec: SynthSpec = SynthSpec()) -> list[Recording]:
    rng = np.random.default_rng(spec.seed)
    recs = [make_subject(rng, 1, f"R{k + 1:02d}", spec) for k in range(spec.n_responders)]
    recs += [make_subject(rng, 0, f"NR{k + 1:02d}", spec) for k in range(spec.n_nonresponders)]
    return recs
