"""Command line: ``eeg-rx synth | features | evaluate | leakage-demo | train | predict``.

RESEARCH USE ONLY. No command gives a diagnosis or a treatment recommendation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import DISCLAIMER
from .config import Settings
from .evaluate import evaluate, leaky_epoch_cv, nested_cv
from .features import FeatureSet, extract
from .io import Recording, load_mat_folder, save_mat_folder
from .model import CLASSIFIERS, SELECTORS
from .preprocess import preprocess
from .report import write as write_report
from .synthetic import SynthSpec, generate
from .train import fit_final, load, predict_recording, save


def _settings(args) -> Settings:
    keys = ("data_dir", "out_dir", "seed", "classifier", "selector", "features", "outer_folds",
            "inner_folds", "permutations", "epoch_seconds")
    return Settings.from_env().merge(**{k: getattr(args, k, None) for k in keys})


def _recordings(args, s: Settings) -> tuple[list[Recording], bool]:
    if getattr(args, "synthetic", False):
        return generate(SynthSpec(seed=s.seed, effect=args.effect)), True
    return load_mat_folder(s.data_dir, s.sfreq), False


def _features(args, s: Settings) -> tuple[FeatureSet, bool]:
    if getattr(args, "features_file", None):
        fs = FeatureSet.load(args.features_file)
        return fs, fs.source == "synthetic"
    recs, synthetic = _recordings(args, s)
    epochs = preprocess(recs, s.epoch_seconds, s.l_freq, s.h_freq, reject_uv=s.reject_uv)
    dropped = sum(epochs.rejected.values())
    print(f"{len(recs)} subjects, {len(epochs.labels)} epochs kept, {dropped} rejected (peak-to-peak > "
          f"{s.reject_uv} uV)", file=sys.stderr)
    return extract(epochs, s.features, source="synthetic" if synthetic else "local"), synthetic


def cmd_synth(args) -> int:
    spec = SynthSpec(seed=args.seed, effect=args.effect, seconds=args.seconds,
                     n_responders=args.responders, n_nonresponders=args.nonresponders)
    folder = save_mat_folder(generate(spec), args.out)
    print(f"wrote {spec.n_responders} responder and {spec.n_nonresponders} non-responder files to {folder}")
    return 0


def cmd_features(args) -> int:
    s = _settings(args)
    fs, _ = _features(args, s)
    path = fs.save(args.out)
    print(f"{fs.X.shape[0]} epochs x {fs.X.shape[1]} features -> {path}")
    return 0


def cmd_evaluate(args) -> int:
    s = _settings(args)
    fs, synthetic = _features(args, s)
    result = evaluate(fs, s)
    j, m = write_report(result, args.out or s.out_dir, synthetic)
    print(m.read_text(encoding="utf-8"))
    print(f"wrote {j} and {m}")
    return 0


def cmd_leakage_demo(args) -> int:
    s = _settings(args)
    fs, _ = _features(args, s)
    leaky = leaky_epoch_cv(fs, s)
    sp = nested_cv(fs, s)
    from .evaluate import binary_metrics

    m = binary_metrics(sp.y, sp.p)
    print(f"label effect: {args.effect}")
    print(f"WRONG protocol (selection on all data, epoch-wise folds): epoch accuracy {leaky:.3f}")
    print(f"subject-wise nested protocol: balanced accuracy {m['balanced_accuracy']:.3f}, AUROC {m['auroc']:.3f}")
    return 0


def cmd_train(args) -> int:
    s = _settings(args)
    fs, synthetic = _features(args, s)
    evaluation = None
    if args.with_evaluation:
        evaluation = evaluate(fs, s, n_perm=0)
    model, best = fit_final(fs, s)
    out = save(model, best, fs, s, args.out, evaluation)
    print(f"trained on {len(set(fs.subjects))} subjects ({'synthetic' if synthetic else 'local'} data), "
          f"parameters {best}. Saved to {out}")
    return 0


def cmd_predict(args) -> int:
    from scipy.io import loadmat

    model, meta = load(args.model_dir)
    x = loadmat(args.recording)["EEG"]
    x = x.T if x.shape[0] != 19 else x
    rec = Recording(Path(args.recording).stem, 0, x.astype(float), meta["settings"]["sfreq"]).validate()
    result = predict_recording(model, meta, rec)
    print(json.dumps(result, indent=2))
    print(DISCLAIMER, file=sys.stderr)
    return 0


def _common(p, synthetic=True) -> None:
    p.add_argument("--data-dir", help="folder with SSRI_R_<n>.mat and SSRI_NR_<n>.mat (EEG_RX_DATA_DIR)")
    if synthetic:
        p.add_argument("--synthetic", action="store_true", help="use synthetic EEG")
        p.add_argument("--effect", type=float, default=0.2, help="label effect of the synthetic data")
    p.add_argument("--features-file", help="an .npz file from 'eeg-rx features'")
    p.add_argument("--features", help="'+' list of lbp, bandpower")
    p.add_argument("--classifier", choices=CLASSIFIERS)
    p.add_argument("--selector", choices=SELECTORS)
    p.add_argument("--outer-folds", type=int, help="0 = leave one subject out")
    p.add_argument("--inner-folds", type=int)
    p.add_argument("--permutations", type=int)
    p.add_argument("--epoch-seconds", type=float)
    p.add_argument("--seed", type=int)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="eeg-rx", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("synth", help="write synthetic .mat recordings")
    p.add_argument("--out", required=True)
    p.add_argument("--effect", type=float, default=0.2)
    p.add_argument("--seconds", type=float, default=120.0, help="recording length of each subject")
    p.add_argument("--responders", type=int, default=12)
    p.add_argument("--nonresponders", type=int, default=18)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("features", help="preprocess and extract features to an .npz file")
    _common(p)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_features)
    p = sub.add_parser("evaluate", help="nested subject-wise evaluation and report")
    _common(p)
    p.add_argument("--out", help="report folder (EEG_RX_OUT_DIR)")
    p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("leakage-demo", help="compare the wrong epoch-wise protocol with the subject-wise one")
    _common(p)
    p.set_defaults(fn=cmd_leakage_demo)
    p = sub.add_parser("train", help="fit the final model on all subjects")
    _common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--with-evaluation", action="store_true", help="add nested results to the model card")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("predict", help="responder probability for one .mat recording (research use only)")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--recording", required=True)
    p.set_defaults(fn=cmd_predict)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
