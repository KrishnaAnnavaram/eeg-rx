"""Settings from environment variables, with CLI flags on top."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Settings:
    data_dir: str = "data/SSRI"
    out_dir: str = "out"
    seed: int = 42
    sfreq: float = 256.0
    epoch_seconds: float = 15.0
    l_freq: float = 0.5
    h_freq: float = 45.0
    reject_uv: float = 300.0  # peak-to-peak limit for one epoch; 0 turns the check off
    features: str = "lbp+bandpower"
    classifier: str = "logreg"
    selector: str = "anova"
    outer_folds: int = 5  # 0 = leave one subject out
    inner_folds: int = 3
    permutations: int = 50

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        types = {f.name: f.type for f in fields(cls)}
        values = {}
        for name, typ in types.items():
            raw = env.get("EEG_RX_" + name.upper())
            if raw:
                values[name] = int(raw) if typ in ("int", int) else float(raw) if typ in ("float", float) else raw
        return cls(**values).validate()

    def merge(self, **kw) -> "Settings":
        return replace(self, **{k: v for k, v in kw.items() if v is not None}).validate()

    def validate(self) -> "Settings":
        if self.epoch_seconds <= 0 or self.sfreq <= 0:
            raise ValueError("epoch_seconds and sfreq must be positive")
        if not 0 <= self.l_freq < self.h_freq < self.sfreq / 2:
            raise ValueError("need 0 <= l_freq < h_freq < sfreq / 2")
        if self.outer_folds == 1 or self.outer_folds < 0 or self.inner_folds < 2:
            raise ValueError("outer_folds must be 0 (LOSO) or >= 2, inner_folds >= 2")
        return self
