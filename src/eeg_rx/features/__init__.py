"""Feature extraction for each epoch. Features are computed one epoch at a time, so they use no
information from other epochs or subjects. Scaling and selection happen later, inside the folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import CHANNELS
from ..preprocess import Epochs
from .lbp import lbp_features, lbp_names
from .spectral import bandpower_features, bandpower_names

KINDS = ("lbp", "bandpower")


@dataclass
class FeatureSet:
    X: np.ndarray
    names: list[str]
    subjects: np.ndarray
    labels: np.ndarray
    source: str = "local"  # "synthetic" or "local"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=self.X, names=np.array(self.names), subjects=self.subjects, labels=self.labels,
                            source=np.array(self.source))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSet":
        with np.load(path, allow_pickle=False) as d:
            source = str(d["source"]) if "source" in d.files else "local"
            return cls(d["X"], d["names"].tolist(), d["subjects"], d["labels"], source)


def parse_kinds(spec: str) -> list[str]:
    kinds = [k.strip() for k in spec.split("+") if k.strip()]
    bad = set(kinds) - set(KINDS)
    if bad or not kinds:
        raise ValueError(f"unknown feature kinds {sorted(bad)}. Use a '+' list of {KINDS}")
    return kinds


def extract(epochs: Epochs, spec: str = "lbp+bandpower", channels=CHANNELS, source: str = "local") -> FeatureSet:
    kinds = parse_kinds(spec)
    rows = []
    for ep in epochs.data:
        parts = []
        if "lbp" in kinds:
            parts.append(lbp_features(ep))
        if "bandpower" in kinds:
            parts.append(bandpower_features(ep, epochs.sfreq, channels))
        rows.append(np.concatenate(parts))
    names = (lbp_names(channels) if "lbp" in kinds else []) + (bandpower_names(channels) if "bandpower" in kinds else [])
    return FeatureSet(np.vstack(rows), names, epochs.subjects, epochs.labels, source)
