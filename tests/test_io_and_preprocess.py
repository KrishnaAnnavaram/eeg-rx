import numpy as np
import pytest

from eeg_rx.io import DataError, Recording, load_mat_folder, save_mat_folder
from eeg_rx.preprocess import average_reference, bandpass, epoch, preprocess
from eeg_rx.synthetic import SynthSpec, generate


def test_mat_folder_round_trip_counts_come_from_the_files(tmp_path):
    """Problem 8: the subject and sample counts are read from the files, not hard-coded."""
    recs = generate(SynthSpec(n_responders=3, n_nonresponders=5, seconds=31.0, seed=1))
    save_mat_folder(recs, tmp_path)
    (tmp_path / "notes.txt").write_text("ignored")
    loaded = load_mat_folder(tmp_path)
    assert len(loaded) == 8 and sum(r.label for r in loaded) == 3
    assert {r.data.shape for r in loaded} == {(19, int(31 * 256))}
    assert loaded[0].subject_id.startswith("NR")
    with pytest.raises(DataError):
        load_mat_folder(tmp_path / "..")  # a folder with no SSRI files
    with pytest.raises(FileNotFoundError):
        load_mat_folder(tmp_path / "missing")


def test_recording_validation():
    with pytest.raises(DataError):
        Recording("s", 1, np.zeros((18, 1000)), 256.0).validate()
    bad = np.zeros((19, 1000))
    bad[0, 0] = np.nan
    with pytest.raises(DataError):
        Recording("s", 1, bad, 256.0).validate()
    with pytest.raises(DataError):
        Recording("s", 2, np.zeros((19, 1000)), 256.0).validate()


def test_epochs_keep_subject_ids_and_reject_blinks():
    recs = generate(SynthSpec(n_responders=2, n_nonresponders=2, seconds=150.0, artefact_rate=0.5, seed=2))
    ep = preprocess(recs, epoch_seconds=15.0)
    assert ep.data.shape[1:] == (19, 15 * 256)
    assert sum(ep.rejected.values()) > 0
    assert len(ep.labels) + sum(ep.rejected.values()) == 4 * 10
    assert ep.subject_table() == {r.subject_id: r.label for r in recs}
    none = preprocess(recs, epoch_seconds=15.0, reject_uv=0)
    assert len(none.labels) == 40


def test_filter_and_reference():
    sf = 256.0
    t = np.arange(int(40 * sf)) / sf
    x = np.vstack([np.sin(2 * np.pi * 10 * t) + 5 * np.sin(2 * np.pi * 80 * t)] * 19)
    y = bandpass(x, sf, 0.5, 45.0)
    mid = y[:, 2560:-2560]
    assert np.abs(mid).max() < 1.1  # the 80 Hz part is gone
    assert np.abs(mid).max() > 0.9  # the 10 Hz part stays
    assert np.allclose(average_reference(x).mean(axis=0), 0)
    rec = Recording("s", 0, np.zeros((19, 1000)), 100.0)
    assert epoch(rec, 3.0).shape == (3, 19, 300)


def test_mixed_sampling_rates_are_refused():
    a = Recording("a", 0, np.random.default_rng(0).normal(size=(19, 5000)), 256.0)
    b = Recording("b", 1, np.random.default_rng(1).normal(size=(19, 5000)), 250.0)
    with pytest.raises(ValueError):
        preprocess([a, b], epoch_seconds=5.0)


def test_mne_loader_optional():
    pytest.importorskip("mne")
