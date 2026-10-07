"""The evaluation report and the model card. Every number comes from the evaluation dict as it is.
No value is changed, rounded up or offset before it is written.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import DISCLAIMER


def _fmt(x) -> str:
    return "n/a" if x != x else f"{x:.3f}"  # NaN check


def markdown(result: dict, synthetic: bool) -> str:
    m, ci = result["subject_metrics"], result["ci95"]
    lines = [
        "# eeg-rx evaluation report", "",
        f"> {DISCLAIMER}", "",
        f"Data: {'SYNTHETIC EEG (not patient data)' if synthetic else 'local recordings'}. "
        f"{result['n_subjects']} subjects ({result['n_responders']} responders), {result['n_epochs']} epochs, "
        f"{result['n_features']} features ({result['features']}).", "",
        f"Protocol: {result['protocol']}, subject-level scores (mean epoch probability). "
        f"Classifier `{result['classifier']}`, selector `{result['selector']}`, seed {result['seed']}.", "",
        "| Subject-level metric | Value | 95% bootstrap interval |", "|---|---|---|",
        f"| AUROC | {_fmt(m['auroc'])} | [{_fmt(ci['auroc'][0])}, {_fmt(ci['auroc'][1])}] |",
        f"| Balanced accuracy | {_fmt(m['balanced_accuracy'])} | "
        f"[{_fmt(ci['balanced_accuracy'][0])}, {_fmt(ci['balanced_accuracy'][1])}] |",
    ]
    for k in ("sensitivity", "specificity", "ppv", "npv", "mcc", "brier"):
        lines.append(f"| {k.upper() if len(k) <= 3 else k.capitalize()} | {_fmt(m[k])} | |")
    lines += ["", f"Confusion (threshold 0.5): TP {m['tp']}, FN {m['fn']}, TN {m['tn']}, FP {m['fp']}.",
              f"Epoch-level AUROC (out-of-fold): {_fmt(result['epoch_auroc'])}."]
    if "permutation" in result:
        p = result["permutation"]
        lines.append(f"Permutation test ({p['n_permutations']} subject-label shuffles): p = {p['p_value']:.3f}, "
                     f"null AUROC mean {_fmt(p['null_auroc_mean'])}, 95th percentile {_fmt(p['null_auroc_95th'])}.")
    return "\n".join(lines) + "\n"


def model_card(result: dict | None, meta: dict) -> str:
    perf = "No nested evaluation was attached to this model." if result is None else (
        f"Nested subject-wise AUROC {_fmt(result['subject_metrics']['auroc'])} "
        f"(95% interval {_fmt(result['ci95']['auroc'][0])} to {_fmt(result['ci95']['auroc'][1])}) "
        f"on {result['n_subjects']} subjects.")
    return "\n".join([
        "# Model card: eeg-rx SSRI response classifier", "",
        f"> {DISCLAIMER}", "",
        "## Intended use",
        "Research on EEG biomarkers of antidepressant response. Not for diagnosis, triage or treatment choice.", "",
        "## Model",
        f"Features `{meta['features']}`, classifier `{meta['classifier']}`, selector `{meta['selector']}`, "
        f"selected parameters {meta.get('best_params', {})}. Trained on {meta['n_subjects']} subjects "
        f"({meta['n_epochs']} epochs).", "",
        "## Performance", perf, "",
        "## Limits",
        "- Small samples give wide intervals. Results do not transfer to other sites, devices or montages "
        "without external validation.",
        "- The subject probability is the mean epoch probability. It is not calibrated on an external set.",
        "- Dataset bias: age, sex, medication history and recording site can confound the label.",
        "",
    ])


def write(result: dict, out_dir: str | Path, synthetic: bool) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j, m = out / "evaluation.json", out / "evaluation.md"
    j.write_text(json.dumps({**result, "synthetic": synthetic, "disclaimer": DISCLAIMER}, indent=2), encoding="utf-8")
    m.write_text(markdown(result, synthetic), encoding="utf-8")
    return j, m
