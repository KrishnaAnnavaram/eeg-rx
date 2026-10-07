import json

import numpy as np
import pytest

from eeg_rx import DISCLAIMER
from eeg_rx.cli import main
from eeg_rx.config import Settings
from eeg_rx.features import extract
from eeg_rx.preprocess import preprocess
from eeg_rx.train import CARD_FILE, fit_final, load, predict_recording, save


def test_prediction_calls_the_trained_model(fs, recs, tmp_path):
    """Problems 1 and 9: the output comes from the model, with a research-only notice."""
    s = Settings(inner_folds=2)
    model, best = fit_final(fs, s)
    save(model, best, fs, s, tmp_path / "m")
    model2, meta = load(tmp_path / "m")
    rec = recs[0]
    out = predict_recording(model2, meta, rec)
    expected = model.predict_proba(extract(preprocess([rec]), meta["features"]).X)[:, 1].mean()
    assert out["responder_probability"] == pytest.approx(expected)
    assert out["disclaimer"] == DISCLAIMER
    card = (tmp_path / "m" / CARD_FILE).read_text()
    assert "RESEARCH USE ONLY" in card and "Not for diagnosis" in card


def test_settings_from_env_and_validation():
    s = Settings.from_env({"EEG_RX_SEED": "7", "EEG_RX_OUTER_FOLDS": "0", "EEG_RX_CLASSIFIER": "svm"})
    assert (s.seed, s.outer_folds, s.classifier) == (7, 0, "svm")
    with pytest.raises(ValueError):
        Settings(outer_folds=1).validate()
    with pytest.raises(ValueError):
        Settings(h_freq=200).validate()


def test_cli_end_to_end(tmp_path, capsys):
    data = tmp_path / "SSRI"
    assert main(["synth", "--out", str(data), "--seconds", "45", "--responders", "6", "--nonresponders", "6",
                 "--effect", "0.35"]) == 0
    feats = tmp_path / "f.npz"
    assert main(["features", "--data-dir", str(data), "--out", str(feats)]) == 0
    assert main(["evaluate", "--features-file", str(feats), "--permutations", "2", "--outer-folds", "3",
                 "--inner-folds", "2", "--out", str(tmp_path / "rep")]) == 0
    rep = json.loads((tmp_path / "rep" / "evaluation.json").read_text())
    assert rep["n_subjects"] == 12 and rep["disclaimer"] == DISCLAIMER and rep["synthetic"] is False
    assert main(["leakage-demo", "--features-file", str(feats), "--outer-folds", "3", "--inner-folds", "2"]) == 0
    assert main(["train", "--data-dir", str(data), "--out", str(tmp_path / "model"), "--inner-folds", "2"]) == 0
    capsys.readouterr()
    rec = sorted(data.glob("SSRI_R_*.mat"))[0]
    assert main(["predict", "--model-dir", str(tmp_path / "model"), "--recording", str(rec)]) == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert 0 <= result["responder_probability"] <= 1 and "RESEARCH USE ONLY" in out.err
    assert main(["evaluate", "--data-dir", str(tmp_path / "none")]) == 2


def test_synthetic_flag_reaches_the_report(tmp_path):
    from eeg_rx.synthetic import SynthSpec, generate
    from eeg_rx.features import FeatureSet

    f = extract(preprocess(generate(SynthSpec(n_responders=2, n_nonresponders=2, seconds=30))), "bandpower",
                source="synthetic")
    assert FeatureSet.load(f.save(tmp_path / "x.npz")).source == "synthetic"
    assert np.isfinite(f.X).all()
