import numpy as np
import pytest

from eeg_rx import CHANNELS
from eeg_rx.features import FeatureSet, extract, parse_kinds
from eeg_rx.features.lbp import N_CODES, lbp_features, pattern_codes, pattern_histogram
from eeg_rx.features.spectral import BANDS, band_powers, bandpower_features, bandpower_names


def reference_code(window25):
    """A plain loop version of the documented rule, written independently of the module."""
    m = np.array(window25, dtype=float).reshape(5, 5, order="F")  # column j = samples 5j..5j+4
    sd = m.std(axis=0, ddof=1)
    order = np.argsort(sd, kind="stable")
    diff = m[:, order[-1]] - m[:, order[-2]]
    th = sd / 2 ** 0.25
    return sum(int(diff[k] >= th[k]) << k for k in range(5))


def test_vectorised_codes_match_the_loop_reference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    codes = pattern_codes(x)
    assert len(codes) == 400 - 25 + 1
    assert all(codes[i] == reference_code(x[i:i + 25]) for i in range(0, len(codes), 7))


def test_threshold_is_element_wise():
    """Problem 7: each bit compares one difference with one threshold, not with the whole vector."""
    w = np.zeros(25)
    w[15:20] = [-0.5, 0.5, -0.5, 0.5, -0.5]  # column 3: second-largest deviation
    w[20:25] = [0, 10, -10, 10, -10]  # column 4: largest deviation
    m = w.reshape(5, 5, order="F")
    sd = m.std(axis=0, ddof=1)
    diff = m[:, 4] - m[:, 3]
    th = sd / 2 ** 0.25
    # Element-wise, bit 0 is set. An "all thresholds" rule would clear it.
    assert diff[0] >= th[0] and not (diff[0] >= th).all()
    assert pattern_codes(w)[0] == 0b01011 == reference_code(w)


def test_histogram_has_fixed_bins_and_sums_to_one():
    rng = np.random.default_rng(1)
    for n in (30, 500, 3840):
        h = pattern_histogram(rng.normal(size=n))
        assert h.shape == (N_CODES,) and h.sum() == pytest.approx(1.0)
    assert lbp_features(rng.normal(size=(19, 300))).shape == (19 * 32,)
    with pytest.raises(ValueError):
        pattern_codes(np.zeros(10))


def test_band_power_finds_an_alpha_sine():
    sf = 256.0
    t = np.arange(int(10 * sf)) / sf
    x = np.tile(10 * np.sin(2 * np.pi * 10 * t), (19, 1))
    bp = band_powers(x, sf)
    assert np.argmax(bp[0]) == list(BANDS).index("alpha")
    x[CHANNELS.index("F4")] *= 2
    feats = bandpower_features(x, sf, CHANNELS)
    assert feats[-1] == pytest.approx(np.log(4), abs=0.01)  # F4 alpha power is 4 times F3
    assert len(feats) == len(bandpower_names(CHANNELS)) == 19 * 5 * 2 + 1


def test_extract_and_round_trip(fs, tmp_path):
    assert fs.X.shape == (len(fs.labels), 19 * 32 + 191)
    assert len(fs.names) == fs.X.shape[1] and fs.source == "synthetic"
    again = FeatureSet.load(fs.save(tmp_path / "f.npz"))
    assert np.array_equal(again.X, fs.X) and again.names == fs.names and again.source == "synthetic"
    with pytest.raises(ValueError):
        parse_kinds("lbp+wavelets")


def test_one_kind_only(recs):
    from eeg_rx.preprocess import preprocess

    f = extract(preprocess(recs[:2]), "bandpower")
    assert f.X.shape[1] == 191
