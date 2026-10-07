"""Fit the final model on all subjects and predict for one new recording.

The prediction calls the trained model. It never shows a stored or typed-in number.
Model files use joblib (pickle). Load only model files that you made yourself.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from . import DISCLAIMER
from .config import Settings
from .evaluate import expected_warnings
from .features import FeatureSet, extract
from .io import Recording
from .model import make_pipeline, param_grid
from .preprocess import preprocess
from .report import model_card

MODEL_FILE = "model.joblib"
META_FILE = "model.json"
CARD_FILE = "MODEL_CARD.md"


def fit_final(fs: FeatureSet, s: Settings):
    pipe = make_pipeline(s.classifier, s.selector, s.seed)
    grid = param_grid(s.selector, fs.X.shape[1])
    if grid:
        search = GridSearchCV(pipe, grid, cv=StratifiedGroupKFold(s.inner_folds, shuffle=True, random_state=s.seed),
                              scoring="roc_auc", refit=True)
        with expected_warnings():
            search.fit(fs.X, fs.labels, groups=fs.subjects)
        return search.best_estimator_, search.best_params_
    return pipe.fit(fs.X, fs.labels), {}


def save(model, best_params: dict, fs: FeatureSet, s: Settings, out_dir: str | Path,
         evaluation: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / MODEL_FILE)
    meta = {
        "settings": asdict(s), "features": s.features, "classifier": s.classifier, "selector": s.selector,
        "best_params": best_params, "feature_names": fs.names,
        "n_subjects": len(set(fs.subjects)), "n_epochs": int(len(fs.labels)),
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (out / CARD_FILE).write_text(model_card(evaluation, meta), encoding="utf-8")
    return out


def load(model_dir: str | Path):
    d = Path(model_dir)
    meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
    return joblib.load(d / MODEL_FILE), meta


def predict_recording(model, meta: dict, rec: Recording) -> dict:
    s = Settings(**meta["settings"])
    epochs = preprocess([rec], s.epoch_seconds, s.l_freq, s.h_freq, reject_uv=s.reject_uv)
    fs = extract(epochs, meta["features"])
    if fs.names != meta["feature_names"]:
        raise ValueError("feature names differ from the trained model")
    p = model.predict_proba(fs.X)[:, 1]
    return {
        "subject_id": rec.subject_id,
        "epochs_used": int(len(p)),
        "epochs_rejected": int(epochs.rejected.get(rec.subject_id, 0)),
        "responder_probability": float(np.mean(p)),
        "epoch_probability_range": [float(p.min()), float(p.max())],
        "disclaimer": DISCLAIMER,
    }
