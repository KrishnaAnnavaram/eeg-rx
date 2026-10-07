"""5x5 matrix-pattern histogram, vectorised, with an element-wise threshold.

For each position of a sliding 25-sample window on one channel:

1. Reshape the 25 samples into a 5x5 matrix, column by column (5 blocks of 5 samples).
2. Calculate the standard deviation of each column (ddof = 1).
3. Take the difference of the column with the highest deviation and the column with the
   second-highest deviation. This gives 5 values.
4. The threshold vector is the column deviations divided by 2 ** 0.25.
5. Bit k is 1 if difference[k] >= threshold[k]. This is an ELEMENT-WISE comparison.
6. The code is sum(bit[k] * 2 ** k), a value from 0 to 31.

The histogram has 32 FIXED bins (codes 0..31) and is divided by the window count, so epochs of
different lengths give comparable features.
"""

from __future__ import annotations

import numpy as np

WINDOW = 25
N_CODES = 32
_WEIGHTS = 2 ** np.arange(5)


def pattern_codes(signal: np.ndarray) -> np.ndarray:
    """One code (0..31) for each window position of a 1-D signal."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size < WINDOW:
        raise ValueError(f"need a 1-D signal with at least {WINDOW} samples")
    win = np.lib.stride_tricks.sliding_window_view(signal, WINDOW)  # (n, 25)
    mats = win.reshape(-1, 5, 5).transpose(0, 2, 1)  # (n, row, column); column j = samples 5j..5j+4
    col_std = mats.std(axis=1, ddof=1)  # (n, 5)
    order = np.argsort(col_std, axis=1, kind="stable")
    top = np.take_along_axis(mats, order[:, None, -1:], axis=2)[..., 0]  # (n, 5)
    second = np.take_along_axis(mats, order[:, None, -2:-1], axis=2)[..., 0]
    diff = top - second
    threshold = col_std / 2 ** 0.25
    bits = (diff >= threshold).astype(int)
    return bits @ _WEIGHTS


def pattern_histogram(signal: np.ndarray) -> np.ndarray:
    codes = pattern_codes(signal)
    return np.bincount(codes, minlength=N_CODES) / codes.size


def lbp_features(epoch: np.ndarray) -> np.ndarray:
    """(channels, samples) -> (channels * 32,) normalised histograms."""
    return np.concatenate([pattern_histogram(ch) for ch in epoch])


def lbp_names(channels) -> list[str]:
    return [f"lbp_{ch}_{code:02d}" for ch in channels for code in range(N_CODES)]
