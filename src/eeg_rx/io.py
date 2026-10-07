"""Loaders that keep the subject ID with every recording.

Two input forms:

- a folder of ``.mat`` files named ``SSRI_R_<n>.mat`` (responder) and ``SSRI_NR_<n>.mat``
  (non-responder), each with a variable ``EEG`` of shape (samples, 19) at 256 Hz;
- a CSV manifest (``subject_id,label,path``) of EDF files, read with MNE (extra ``mne``).

The number of subjects and the number of samples come from the files. Nothing is hard-coded.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import CHANNELS

MAT_PATTERN = re.compile(r"^SSRI_(R|NR)_(\d+)\.mat$", re.IGNORECASE)


class DataError(ValueError):
    pass


@dataclass
class Recording:
    subject_id: str
    label: int  # 1 = responder, 0 = non-responder
    data: np.ndarray  # (channels, samples), microvolts
    sfreq: float
    channels: tuple[str, ...] = CHANNELS

    def validate(self, min_seconds: float = 1.0) -> "Recording":
        if self.label not in (0, 1):
            raise DataError(f"{self.subject_id}: label must be 0 or 1")
        if self.data.ndim != 2 or self.data.shape[0] != len(self.channels):
            raise DataError(f"{self.subject_id}: expected ({len(self.channels)}, samples), got {self.data.shape}")
        if self.data.shape[1] < min_seconds * self.sfreq:
            raise DataError(f"{self.subject_id}: recording shorter than {min_seconds} s")
        if not np.isfinite(self.data).all():
            raise DataError(f"{self.subject_id}: NaN or infinite samples")
        return self


def load_mat_folder(folder: str | Path, sfreq: float = 256.0, variable: str = "EEG") -> list[Recording]:
    from scipy.io import loadmat

    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found. See data/README.md.")
    recs = []
    for path in sorted(folder.iterdir()):
        m = MAT_PATTERN.match(path.name)
        if not m:
            continue
        mat = loadmat(path)
        if variable not in mat:
            raise DataError(f"{path.name}: no variable {variable!r}")
        x = np.asarray(mat[variable], dtype=float)
        if x.ndim == 2 and x.shape[1] == len(CHANNELS) and x.shape[0] != len(CHANNELS):
            x = x.T  # files store samples x channels
        group = m.group(1).upper()
        recs.append(Recording(f"{group}{int(m.group(2)):02d}", 1 if group == "R" else 0, x, sfreq).validate())
    if not recs:
        raise DataError(f"no SSRI_R_<n>.mat or SSRI_NR_<n>.mat files in {folder}")
    return recs


def load_edf_manifest(manifest: str | Path) -> list[Recording]:  # pragma: no cover - needs MNE
    """Read ``subject_id,label,path`` rows. The EDF channel names must contain the 19 channels."""
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError('MNE is not installed: pip install -e ".[mne]"') from exc
    base = Path(manifest).parent
    recs = []
    with open(manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = mne.io.read_raw_edf(base / row["path"], preload=True, verbose="ERROR")
            names = {n.upper().replace("EEG ", "").split("-")[0]: n for n in raw.ch_names}
            picks = [names[c.upper()] for c in CHANNELS]
            x = raw.get_data(picks=picks) * 1e6
            recs.append(Recording(row["subject_id"], int(row["label"]), x, float(raw.info["sfreq"])).validate())
    return recs


def save_mat_folder(recs: list[Recording], folder: str | Path) -> Path:
    """Write recordings in the ``.mat`` layout (samples x channels). Used for synthetic data."""
    from scipy.io import savemat

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    counters = {0: 0, 1: 0}
    for r in recs:
        counters[r.label] += 1
        name = f"SSRI_{'R' if r.label else 'NR'}_{counters[r.label]}.mat"
        savemat(folder / name, {"EEG": r.data.T.astype(np.float32)})
    return folder
