"""Filtering, re-referencing, epoching and artefact rejection.

Each epoch keeps its subject ID and label, so no later step can mix subjects by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from .io import Recording


@dataclass
class Epochs:
    data: np.ndarray  # (n_epochs, channels, samples)
    subjects: np.ndarray  # (n_epochs,) subject ID of each epoch
    labels: np.ndarray  # (n_epochs,) label of each epoch (the subject label)
    sfreq: float
    rejected: dict[str, int]

    def subject_table(self) -> dict[str, int]:
        """Subject ID -> label."""
        return {s: int(l) for s, l in zip(self.subjects, self.labels)}


def bandpass(x: np.ndarray, sfreq: float, l_freq: float, h_freq: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos") if l_freq > 0 else \
        butter(order, h_freq, btype="lowpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def notch(x: np.ndarray, sfreq: float, freq: float, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(freq, q, fs=sfreq)
    return sosfiltfilt(tf2sos(b, a), x, axis=-1)


def average_reference(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def epoch(rec: Recording, seconds: float) -> np.ndarray:
    n = int(round(seconds * rec.sfreq))
    k = rec.data.shape[1] // n
    return rec.data[:, :k * n].reshape(rec.data.shape[0], k, n).transpose(1, 0, 2)


def preprocess(recs: list[Recording], epoch_seconds: float = 15.0, l_freq: float = 0.5, h_freq: float = 45.0,
               line_freq: float | None = None, reject_uv: float = 300.0) -> Epochs:
    """Filter -> average reference -> non-overlapping epochs -> peak-to-peak rejection."""
    sfreqs = {r.sfreq for r in recs}
    if len(sfreqs) != 1:
        raise ValueError(f"all recordings need one sampling rate, got {sorted(sfreqs)}")
    sfreq = sfreqs.pop()
    data, subjects, labels, rejected = [], [], [], {}
    for r in recs:
        x = bandpass(r.data, sfreq, l_freq, h_freq)
        if line_freq:
            x = notch(x, sfreq, line_freq)
        ep = epoch(Recording(r.subject_id, r.label, average_reference(x), sfreq, r.channels), epoch_seconds)
        if reject_uv > 0 and len(ep):
            ptp = ep.max(axis=2) - ep.min(axis=2)
            keep = (ptp <= reject_uv).all(axis=1)
            rejected[r.subject_id] = int((~keep).sum())
            ep = ep[keep]
        if len(ep) == 0:
            continue
        data.append(ep)
        subjects += [r.subject_id] * len(ep)
        labels += [r.label] * len(ep)
    if not data:
        raise ValueError("no epoch survived. Check epoch_seconds and reject_uv")
    return Epochs(np.concatenate(data), np.array(subjects), np.array(labels), sfreq, rejected)
